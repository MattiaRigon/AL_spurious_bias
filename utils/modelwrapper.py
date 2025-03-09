# from https://github.com/baal-org/baal/blob/master/baal/modelwrapper.py
import gc
import logging
import os
import re
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from baal.active.dataset.base import Dataset
from baal.active.heuristics.heuristics import to_prob
from baal.metrics.mixin import MetricMixin
from baal.utils.array_utils import stack_in_memory
from baal.utils.iterutils import map_on_tensor
from baal.utils.metrics import Loss
from baal.utils.warnings import raise_warnings_cache_replicated
from colorama import Fore, Style
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm
from wilds.common.grouper import CombinatorialGrouper

from utils.dataloader import InfiniteDataLoader
from utils.eval_helper import eval_metrics
from utils.misc import to_device
from utils.rrr_loss import RRRLoss

log = logging.getLogger("ModelWrapper")


def _stack_preds(out):
    if isinstance(out[0], Sequence):
        out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
    else:
        out = torch.stack(out, dim=-1)
    return out


class ModelWrapper(MetricMixin):
    """
    Wrapper created to ease the training/testing/loading.

    Args:
        model (nn.Module): The model to optimize.
        criterion (Callable): A loss function.
        replicate_in_memory (bool): Replicate in memory optional.
    """

    def __init__(self, model: torch.nn.Module, criterion: _Loss, replicate_in_memory=False):
        self.model = model
        self.device = model.device
        self.criterion = criterion
        self.metrics = dict()
        self.active_learning_metrics = defaultdict(dict)
        self.add_metric("loss", lambda: Loss())
        self.replicate_in_memory = replicate_in_memory
        self._active_dataset_size = -1
        self._ckpt_path = Path("checkpoints")

        # assert isinstance(criterion, _Loss)
        self.element_wise_criterion = deepcopy(criterion)
        self.element_wise_criterion.reduction = "none"

        raise_warnings_cache_replicated(self.model, replicate_in_memory=replicate_in_memory)

    def train_on_dataset_epoch(
        self,
        dataset,
        batch_size,
        n_steps,
        checkpoint_freq=1,
        workers=4,
        upload=False,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
    ):
        torch.cuda.empty_cache()
        gc.collect()

        os.makedirs(self._ckpt_path, exist_ok=True)
        optimizer: Optimizer
        lr_scheduler: LRScheduler
        optimizer, lr_scheduler = self.model.get_optimizer()
        dataset_size = len(dataset)
        self.train()
        self.set_dataset_size(dataset_size)
        log.info(f"Starting training: n_epoch={n_steps} , dataset={dataset_size}")
        collate_fn = collate_fn or default_collate

        loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
        for epoch in tqdm(range(1, n_steps + 1)):
            for data, target, *_ in tqdm(loader, leave=False):
                _ = self.train_on_batch_epoch(data, target, optimizer, regularizer)

            if lr_scheduler is not None:
                lr_scheduler.step()

            if ((epoch % checkpoint_freq == 0) or (epoch == n_steps + 1)):
                ckpt_name = self._ckpt_path / f"model_step={epoch}.pkl"
                torch.save(self.model.state_dict(), ckpt_name)
                if upload:
                    wandb.save(str(ckpt_name.absolute()), base_path=str(self._ckpt_path.parent.absolute()))
        optimizer.zero_grad()  # Assert that the gradient is flushed.
        log.info("Training complete")

    def train_on_dataset(
        self,
        dataset,
        batch_size,
        n_steps,
        checkpoint_freq,
        weights=None,
        workers=4,
        upload=False,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
    ):
        """
        Train for `epoch` epochs on a Dataset `dataset.

        Args:
            dataset (Dataset): Pytorch Dataset to be trained on.
            batch_size (int): The batch size used in the DataLoader.
            n_steps (int): Number of step to train for.
            checkpoint_freq (int): Checkpoint every N steps.
            workers (int): Number of workers for the multiprocessing.
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.

        Returns:
            The training history.
        """
        torch.cuda.empty_cache()
        gc.collect()

        os.makedirs(self._ckpt_path, exist_ok=True)
        optimizer: Optimizer
        lr_scheduler: LRScheduler
        optimizer, lr_scheduler = self.model.get_optimizer()
        dataset_size = len(dataset)
        self.train()
        self.set_dataset_size(dataset_size)
        history = []
        log.info(f"Starting training: n_steps={n_steps}, checkpoint_freq={checkpoint_freq}, dataset={dataset_size}")
        collate_fn = collate_fn or default_collate

        loader = iter(InfiniteDataLoader(dataset, weights, batch_size, num_workers=workers, collate_fn=collate_fn))
        self._reset_metrics("train")
        for step in tqdm(range(n_steps)):
            data, target, *_ = next(loader)
            _ = self.train_on_batch(data, target, optimizer, lr_scheduler, regularizer)
            if ((step % checkpoint_freq == 0) or (step == n_steps - 1)) and (step > 0):
                history.append(self.get_metrics("train")["train_loss"])

                ckpt_name = self._ckpt_path / f"model_step={step}.pkl"
                torch.save(self.model.state_dict(), ckpt_name)
                if upload:
                    wandb.save(str(ckpt_name.absolute()), base_path=str(self._ckpt_path.parent.absolute()))

                if step != n_steps - 1:
                    self._reset_metrics("train")

        optimizer.zero_grad()  # Assert that the gradient is flushed.
        log.info(f"Training complete: train_loss={self.get_metrics('train')['train_loss']}")
        self.active_step(dataset_size, self.get_metrics("train"))
        return history

    def test_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        average_predictions: int = 1,
    ):
        """
        Test the model on a Dataset `dataset`.

        Args:
            dataset (Dataset): Dataset to evaluate on.
            batch_size (int): Batch size used for evaluation.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            average_predictions (int): The number of predictions to average to
                compute the test loss.

        Returns:
            Average loss value over the dataset.
        """
        self.eval()
        log.info(f"Starting evaluating: dataset={len(dataset)}")
        self._reset_metrics("test")

        for data, target, *_ in DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn):
            _ = self.test_on_batch(data, target, average_predictions=average_predictions)

        log.info(f"Evaluation complete: test_loss={self.get_metrics('test')['test_loss']}")
        self.active_step(None, self.get_metrics("test"))
        return self.get_metrics("test")["test_loss"]

    def train_and_test_on_datasets(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        epoch: int,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
        return_best_weights=False,
        patience=None,
        min_epoch_for_es=0,
        skip_epochs=1,
    ):
        """
        Train and test the model on both Dataset `train_dataset`, `test_dataset`.

        Args:
            train_dataset (Dataset): Dataset to train on.
            test_dataset (Dataset): Dataset to evaluate on.
            batch_size (int): Batch size used.
            epoch (int): Number of epoch to train on.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.
            return_best_weights (bool): If True, will keep the best weights and return them.
            patience (Optional[int]): If provided, will use early stopping to stop after
                                        `patience` epoch without improvement.
            min_epoch_for_es (int): Epoch at which the early stopping starts.
            skip_epochs (int): Number of epochs to skip for test_on_dataset

        Returns:
            History and best weights if required.
        """
        optimizer: Optimizer = self.model.get_optimizer()
        best_weight = None
        best_loss = 1e10
        best_epoch = 0
        hist = []
        for e in range(epoch):
            _ = self.train_on_dataset(train_dataset, optimizer, batch_size, 1, workers, collate_fn, regularizer)
            if e % skip_epochs == 0:
                te_loss = self.test_on_dataset(test_dataset, batch_size, workers, collate_fn)
                hist.append(self.get_metrics())
                if te_loss < best_loss:
                    best_epoch = e
                    best_loss = te_loss
                    if return_best_weights:
                        best_weight = deepcopy(self.state_dict())

                if patience is not None and (e - best_epoch) > patience and (e > min_epoch_for_es):
                    # Early stopping
                    break
            else:
                hist.append(self.get_metrics("train"))

        if return_best_weights:
            return hist, best_weight
        else:
            return hist

    def predict_on_dataset_generator(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time.

        Args:
            dataset (Dataset): Dataset to predict on.
            batch_size (int):  Batch size to use during prediction.
            iterations (int): Number of iterations per sample.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to display progress

        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Generators [batch_size, n_classes, ..., n_iterations].
        """
        self.eval()
        if len(dataset) == 0:
            return None

        collate_fn = collate_fn or default_collate
        loader = DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn)
        if verbose:
            log.info(f"Start Predict: dataset={len(dataset)}")
            loader = tqdm(loader, total=len(loader))
        for idx, (data, *_) in enumerate(loader):
            pred = self.predict_on_batch(data, iterations)
            pred = map_on_tensor(lambda x: x.detach(), pred)
            if half:
                pred = map_on_tensor(lambda x: x.half(), pred)
            yield map_on_tensor(lambda x: x.cpu().numpy(), pred)

    def predict_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time.

        Args:
            dataset (Dataset): Dataset to predict on.
            batch_size (int):  Batch size to use during prediction.
            iterations (int): Number of iterations per sample.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to show progress.

        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Array [n_samples, n_outputs, ..., n_iterations].
        """
        preds = list(
            self.predict_on_dataset_generator(
                dataset=dataset,
                batch_size=batch_size,
                iterations=iterations,
                workers=workers,
                collate_fn=collate_fn,
                half=half,
                verbose=verbose,
            )
        )

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def eval_loss_on_batch(self, data, target, iterations=1):
        data, target = to_device(data, self.device), to_device(target, self.device)
        with torch.no_grad():
            output = [self.model(data) for _ in range(iterations)]
        loss = [self.element_wise_criterion(out, target) for out in output]
        return _stack_preds(loss)

    def eval_loss_dataset_generator(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        verbose=True,
    ):
        self.eval()
        collate_fn = collate_fn or default_collate
        loader = DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn)
        if verbose:
            log.info(f"Start Eval Loss: dataset={len(dataset)}")
            loader = tqdm(loader, total=len(loader))
        for _, (data, target, metadata) in enumerate(loader):
            loss = self.eval_loss_on_batch(data, target, iterations)
            loss = map_on_tensor(lambda x: x.detach(), loss)
            yield (
                map_on_tensor(lambda x: x.cpu().numpy(), loss),
                map_on_tensor(lambda x: x.numpy(), target),
                map_on_tensor(lambda x: x.numpy(), metadata),
            )

    def eval_loss_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        verbose=True,
    ):
        losses = list(
            self.eval_loss_dataset_generator(
                dataset=dataset,
                batch_size=batch_size,
                iterations=iterations,
                workers=workers,
                collate_fn=collate_fn,
                verbose=verbose,
            )
        )
        loss, target, metadata = list(zip(*losses))
        loss = np.vstack(loss)
        if np.ndim(target[0]) == 1:
            target = np.concatenate(target)
        else:
            target = np.vstack(target)
        metadata = np.vstack(metadata)
        return loss, target, metadata

    def train_on_batch_epoch(self, data, target, optimizer, regularizer: Optional[Callable] = None):
        data, target = to_device(data, self.device), to_device(target, self.device)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        return loss

    def train_on_batch(
        self, data, target, optimizer, lr_scheduler: LRScheduler | None, regularizer: Optional[Callable] = None
    ):
        """
        Train the current model on a batch using `optimizer`.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            optimizer (optim.Optimizer): An optimizer.
            regularizer (Optional[Callable]): The loss regularization for training.


        Returns:
            Tensor, the loss computed from the criterion.
        """
        data, target = to_device(data, self.device), to_device(target, self.device)
        target, target_mask = target
        if isinstance(self.criterion, RRRLoss):
            data.requires_grad_(True)

        optimizer.zero_grad()
        output = self.model(data)

        if isinstance(self.criterion, RRRLoss):
            loss = self.criterion(target_mask, data, target, output, torch.nn.CrossEntropyLoss(), None, 0.1)
        else:
            loss = self.criterion(output, target)

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        self._update_metrics(output, target, loss, filter="train")
        return loss

    def test_on_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        average_predictions: int = 1,
    ):
        """
        Test the current model on a batch.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            average_predictions (int): The number of predictions to average to
                compute the test loss.

        Returns:
            Tensor, the loss computed from the criterion.
        """
        with torch.no_grad():
            data, target = to_device(data, self.device), to_device(target, self.device)

            preds = map_on_tensor(
                lambda p: p.mean(-1),
                self.predict_on_batch(data, iterations=average_predictions),
            )
            loss = self.criterion(preds, target)
            self._update_metrics(preds, target, loss, "test")
            return loss

    def grad_embedding_on_batch(self, data, target, iterations=1):
        if iterations > 1:
            raise NotImplementedError(f"not support iterations > 1, but got {iterations}")

        with torch.no_grad():
            data, target = to_device(data, self.device), to_device(target, self.device)
            embed, logits = self.model(data, return_embedding=True)

            batchProbs = torch.softmax(logits, dim=1)
            maxInds = torch.argmax(batchProbs, 1)
            num_labels = batchProbs.shape[1]

            def fn(c):
                return embed * torch.where(maxInds == c, 1 - batchProbs[:, c], -1 * batchProbs[:, c]).unsqueeze(1)

            grads = torch.hstack(list(map(fn, range(num_labels))))
            return grads

    def exp_grad_embedding_on_batch(self, x, y, iterations=1):
        if iterations > 1:
            raise NotImplementedError(f"not support iterations > 1, but got {iterations}")

        x, y = to_device(x, self.device), to_device(y, self.device)
        with torch.no_grad():
            embed, cout = self.model(x, return_embedding=True)
        embed = embed.data.cpu()
        batchProbs = F.softmax(cout, dim=1).data.cpu()
        num_labels = batchProbs.shape[1]

        def fn(ind, c):
            if c == ind:
                return embed * (1 - batchProbs[:, c]).unsqueeze(1)
            else:
                return embed * (-1 * batchProbs[:, c]).unsqueeze(1)

        exp_grads_embed = []
        for ind in range(num_labels):
            grads_embed = torch.hstack(list(map(partial(fn, ind), range(num_labels))))
            exp_grads_embed.append(grads_embed * np.sqrt(batchProbs[:, ind]).unsqueeze(1))

        return torch.stack(exp_grads_embed, 1)

    def exp_grad_embedding_on_dataset_generator(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        self.eval()
        if len(dataset) == 0:
            return None

        log.info(f"Embedding: dataset={len(dataset)}")
        loader = DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn)
        if verbose:
            loader = tqdm(loader, total=len(loader))

        for data, *_ in loader:
            embed = self.exp_grad_embedding_on_batch(data, iterations)
            if half:
                embed = map_on_tensor(lambda x: x.half(), embed)
            embed = map_on_tensor(lambda x: x.cpu().numpy(), embed)
            yield embed

    def exp_grad_embedding_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        embed = list(
            self.exp_grad_embedding_on_dataset_generator(
                dataset=dataset,
                batch_size=batch_size,
                iterations=iterations,
                workers=workers,
                collate_fn=collate_fn,
                half=half,
                verbose=verbose,
            )
        )

        if len(embed) > 0 and not isinstance(embed[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(embed)
        return [np.vstack(pr) for pr in zip(*embed)]

    def embedding_on_batch(self, data, iterations=1):
        if iterations > 1:
            raise NotImplementedError(f"not support iterations > 1, but got {iterations}")

        with torch.no_grad():
            data = to_device(data, self.device)
            embed, _ = self.model(data, return_embedding=True)
            return embed

    def embedding_on_dataset_generator(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        self.eval()
        if len(dataset) == 0:
            return None

        log.info(f"Embedding: dataset={len(dataset)}")
        loader = DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn)
        if verbose:
            loader = tqdm(loader, total=len(loader))

        for data, *_ in loader:
            embed = self.embedding_on_batch(data, iterations)
            if half:
                embed = map_on_tensor(lambda x: x.half(), embed)
            embed = map_on_tensor(lambda x: x.cpu().numpy(), embed)
            yield embed

    def embedding_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        embed = list(
            self.embedding_on_dataset_generator(
                dataset=dataset,
                batch_size=batch_size,
                iterations=iterations,
                workers=workers,
                collate_fn=collate_fn,
                half=half,
                verbose=verbose,
            )
        )

        if len(embed) > 0 and not isinstance(embed[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(embed)
        return [np.vstack(pr) for pr in zip(*embed)]

    def grad_embedding_on_dataset_generator(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        self.eval()
        if len(dataset) == 0:
            return None

        log.info(f"Gradient Embedding: dataset={len(dataset)}")
        loader = DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn)
        if verbose:
            loader = tqdm(loader, total=len(loader))

        for data, target, *_ in loader:
            grad = self.grad_embedding_on_batch(data, target, iterations)
            if half:
                grad = map_on_tensor(lambda x: x.half(), grad)
            grad = map_on_tensor(lambda x: x.cpu().numpy(), grad)
            yield grad

    def grad_embedding_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        grads = list(
            self.grad_embedding_on_dataset_generator(
                dataset=dataset,
                batch_size=batch_size,
                iterations=iterations,
                workers=workers,
                collate_fn=collate_fn,
                half=half,
                verbose=verbose,
            )
        )

        if len(grads) > 0 and not isinstance(grads[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(grads)
        return [np.vstack(pr) for pr in zip(*grads)]

    def predict_on_batch(self, data, iterations=1):
        """
        Get the model's prediction on a batch.

        Args:
            data (Tensor): The model input.
            iterations (int): Number of prediction to perform.

        Returns:
            Tensor, the loss computed from the criterion.
                    shape = {batch_size, nclass, n_iteration}.

        Raises:
            Raises RuntimeError if CUDA rans out of memory during data replication.
        """
        with torch.no_grad():
            data = to_device(data, self.device)
            if self.replicate_in_memory:
                data = map_on_tensor(lambda d: stack_in_memory(d, iterations), data)
                try:
                    out = self.model(data)
                except RuntimeError as e:
                    raise RuntimeError(
                        """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
                    Use `replicate_in_memory=False` in order to reduce the memory requirements.
                    Note that there will be some speed trade-offs"""
                    ) from e
                out = map_on_tensor(lambda o: o.view([iterations, -1, *o.size()[1:]]), out)
                out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)
            else:
                out = [self.model(data) for _ in range(iterations)]
                out = _stack_preds(out)
            return out

    def eval_on_batch(self, data, num_labels, average_predictions=1):
        with torch.no_grad():
            data = to_device(data, self.device)

            preds = map_on_tensor(
                lambda p: p.mean(-1),
                self.predict_on_batch(data, iterations=average_predictions),
            )

            if preds.squeeze().ndim == 1:
                preds = torch.sigmoid(preds)
            else:
                preds = torch.softmax(preds, dim=-1)
                if num_labels == 2:
                    preds = preds[:, 1]
            return preds

    def eval_on_dataset(
        self,
        dataset: Dataset,
        num_labels: int,
        attr_grouper: CombinatorialGrouper,
        grouper: CombinatorialGrouper,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        log.info(f"Starting evaluating: dataset={len(dataset)}")

        foo = list(
            self.eval_on_dataset_generator(
                dataset=dataset,
                num_labels=num_labels,
                attr_grouper=attr_grouper,
                grouper=grouper,
                batch_size=batch_size,
                iterations=iterations,
                workers=workers,
                collate_fn=collate_fn,
                half=half,
                verbose=verbose,
            )
        )  # pred, y, attr, group

        preds, target, attr, groups = [
            np.concatenate([item[i][0] if isinstance(item[i], list) else item[i] for item in foo], axis=0) 
            for i in range(4)
        ]            
        metrics = eval_metrics(targets=target, attributes=attr, preds=preds, gs=groups)
        log.info(
            "Evaluation complete:"
            f" {Fore.CYAN}overall/accuracy={100 * metrics['overall']['accuracy']:.2f}%"
            f" {Fore.GREEN}min_group/accuracy={100 * metrics['min_group']['accuracy']:.2f}%{Style.RESET_ALL}"
        )
        return metrics

    def eval_on_dataset_generator(
        self,
        dataset: Dataset,
        num_labels: int,
        attr_grouper: CombinatorialGrouper,
        grouper: CombinatorialGrouper,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        self.eval()
        if len(dataset) == 0:
            return None

        loader = DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn)
        if verbose:
            loader = tqdm(loader, total=len(loader))

        for data, target, meta in loader:
            pred = self.eval_on_batch(data, num_labels, iterations)
            pred = map_on_tensor(lambda x: x.detach(), pred)
            if half:
                pred = map_on_tensor(lambda x: x.half(), pred)
            pred = map_on_tensor(lambda x: x.cpu().numpy(), pred)
            yield (pred, target, attr_grouper.metadata_to_group(meta), grouper.metadata_to_group(meta))

    def get_params(self):
        """
        Return the parameters to optimize.

        Returns:
            Config for parameters.
        """
        return self.model.parameters()

    def state_dict(self):
        """Get the state dict(s)."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load the model with `state_dict`."""
        self.model.load_state_dict(state_dict, strict=strict)

    def train(self):
        """Set the model in `train` mode."""
        self.model.train()

    def eval(self):
        """Set the model in `eval mode`."""
        self.model.eval()

    def reset_fcs(self):
        """Reset all torch.nn.Linear layers."""

        def reset(m):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        self.model.apply(reset)

    def reset_all(self):
        """Reset all *resetable* layers."""

        def reset(m):
            for m in self.model.modules():
                getattr(m, "reset_parameters", lambda: None)()

        self.model.apply(reset)

    def set_dataset_size(self, dataset_size: int):
        """
        Set state for dataset size. Useful for tracking.

        Args:
            dataset_size: Dataset state
        """
        self._active_dataset_size = dataset_size

    def _get_checkpoints(self) -> dict[int, Path]:
        paths = self._ckpt_path.glob("*model_step=*.pkl")
        checkpoints = {}
        for p in paths:
            step = int(re.findall("model_step=(.*).pkl", str(p))[0])
            checkpoints[step] = p
        return OrderedDict(sorted(checkpoints.items()))

    def checkpoints_predictions_on_dataset(self, *args, **kwargs):
        return self._checkpoints_predictions_on_dataset(*args, **kwargs, output_prob=False)

    def checkpoints_predictions_prob_on_dataset(self, *args, **kwargs):
        return self._checkpoints_predictions_on_dataset(*args, **kwargs, output_prob=True)

    def _checkpoints_predictions_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int = 1,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        output_prob=True,
        return_steps=False,
    ):
        _state_dict = self.state_dict()
        predictions, steps = [], []
        ckpts = self._get_checkpoints()
        assert len(ckpts) > 0

        for step, ckpt in (bar := tqdm(ckpts.items())):
            bar.set_description(f"ckpt {step=}")
            self.load_state_dict(torch.load(ckpt))
            # with all_logging_disabled():
            pred = self.predict_on_dataset(dataset, batch_size, iterations, workers, collate_fn, half, verbose=False)
            if output_prob:
                predictions.append(singlepass(to_prob(pred)))
            else:
                predictions.append(singlepass(pred))
            steps.append(step)
        self.load_state_dict(_state_dict)

        if return_steps:
            return steps, np.stack(predictions, axis=2)
        return np.stack(predictions, axis=2)


def singlepass(probabilities):
    if probabilities.ndim >= 3:
        # Expected shape : [n_sample, n_classes, ..., n_iterations]
        probabilities = probabilities.mean(-1)
    return probabilities


def mc_inference(model, data, iterations, replicate_in_memory):
    if replicate_in_memory:
        input_shape = data.size()
        batch_size = input_shape[0]
        try:
            data = torch.stack([data] * iterations)
        except RuntimeError as e:
            raise RuntimeError(
                """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
            Use `replicate_in_memory=False` in order to reduce the memory requirements.
            Note that there will be some speed trade-offs"""
            ) from e
        data = data.view(batch_size * iterations, *input_shape[1:])
        try:
            out = model(data)
        except RuntimeError as e:
            raise RuntimeError(
                """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
            Use `replicate_in_memory=False` in order to reduce the memory requirements.
            Note that there will be some speed trade-offs"""
            ) from e
        out = map_on_tensor(lambda o: o.view([iterations, batch_size, *o.size()[1:]]), out)
        out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)
    else:
        out = [model(data) for _ in range(iterations)]
        if isinstance(out[0], Sequence):
            out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
        else:
            out = torch.stack(out, dim=-1)
    return out


@dataclass
class TrainConfig:
    n_steps: int
    batch_size: int
    checkpoint_freq: int
    workers: int
    dict = asdict


@dataclass
class TestConfig:
    batch_size: int
    workers: int
    dict = asdict
