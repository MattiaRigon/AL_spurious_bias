import random
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest
import numpy as np
from utils.modelwrapper import TrainConfig


def test_train_on_dataset(wrapper_model, test_subset, train_cfg: TrainConfig, mocker):
    os = mocker.patch("os.makedirs")
    torch_save = mocker.patch("torch.save")
    wandb_save = mocker.patch("wandb.save")

    history = wrapper_model.train_on_dataset(test_subset, **train_cfg.dict())

    os.assert_called()
    torch_save.assert_called()
    for call in wandb_save.call_args_list:
        assert isinstance(call.args[0], str)
    assert len(history) == (train_cfg.n_steps / train_cfg.checkpoint_freq)


def test_train_on_dataset_with_weights(wrapper_model, test_subset, train_cfg: TrainConfig, mocker):
    os = mocker.patch("os.makedirs")
    torch_save = mocker.patch("torch.save")
    wandb_save = mocker.patch("wandb.save")

    weights = np.random.rand(len(test_subset))
    history = wrapper_model.train_on_dataset(test_subset, weights=weights, **train_cfg.dict())

    os.assert_called_once_with(wrapper_model._ckpt_path, exist_ok=True)
    torch_save.assert_called()
    for call in wandb_save.call_args_list:
        assert isinstance(call.args[0], str)
    assert len(history) == (train_cfg.n_steps / train_cfg.checkpoint_freq)


def test_predict_on_dataset(wrapper_model, test_subset, test_cfg):
    preds = wrapper_model.predict_on_dataset(test_subset, **test_cfg.dict())
    assert preds.shape == (len(test_subset), 2, 1)


def test_test_on_dataset(wrapper_model, test_subset, test_cfg):
    wrapper_model.test_on_dataset(test_subset, **test_cfg.dict())


def test_eval_on_batch(wrapper_model, test_loader):
    data = next(iter(test_loader))[0]
    preds = wrapper_model.eval_on_batch(data, 2)
    assert preds.shape == (len(data),)
    preds = wrapper_model.eval_on_batch(data, 2, 10)
    assert preds.shape == (len(data),)
    preds = wrapper_model.eval_on_batch(data, 3)
    assert preds.shape == (len(data), 2)


def test_eval_on_dataset_generator(wrapper_model, test_subset, test_cfg, dataset):
    func = partial(
        wrapper_model.eval_on_dataset_generator,
        dataset=test_subset,
        attr_grouper=dataset.attr_grouper,
        grouper=dataset.grouper,
        **test_cfg.dict(),
    )
    preds_gen = func(num_labels=2)
    assert len(next(preds_gen)) == 4
    assert next(preds_gen)[0].shape == (test_cfg.batch_size,)
    preds_gen = func(num_labels=3)
    assert next(preds_gen)[0].shape == (test_cfg.batch_size, 2)


def test_eval_on_dataset(wrapper_model, test_subset, test_cfg, dataset):
    metrics = wrapper_model.eval_on_dataset(
        test_subset,
        num_labels=2,
        attr_grouper=dataset.attr_grouper,
        grouper=dataset.grouper,
        **test_cfg.dict(),
    )
    assert isinstance(metrics, dict)


def test_grad_embedding_on_batch(wrapper_model, test_loader):
    data, target, *_ = next(iter(test_loader))
    grad = wrapper_model.grad_embedding_on_batch(data, target)
    assert grad.shape == (len(data), 2 * 4)

    with pytest.raises(NotImplementedError):
        wrapper_model.grad_embedding_on_batch(data, target, 10)


def test_grad_embedding_on_dataset_generator(wrapper_model, test_subset, test_cfg):
    grads = wrapper_model.grad_embedding_on_dataset_generator(dataset=test_subset, **test_cfg.dict())
    assert next(grads).shape == (test_cfg.batch_size, 2 * 4)


def test_grad_embedding_on_dataset(wrapper_model, test_subset, test_cfg):
    grads = wrapper_model.grad_embedding_on_dataset(dataset=test_subset, **test_cfg.dict())
    assert grads.shape == (len(test_subset), 2 * 4)


def test_get_checkpoints(wrapper_model):
    wrapper_model._ckpt_path = Mock(spec=Path)
    randomlist = random.sample(range(0, 100), 10)
    wrapper_model._ckpt_path.glob = Mock(side_effect=lambda x: iter([Path(f"model_step={i}.pkl") for i in randomlist]))
    ckpts = wrapper_model._get_checkpoints()
    steps = list(ckpts.keys())
    assert len(steps) == 10
    assert all(x <= y for x, y in zip(steps, steps[1:]))  # increasing order


def test_checkpoints_predictions_prob_on_dataset(wrapper_model, test_subset, test_cfg, mocker):
    torch_load = mocker.patch("torch.load")
    torch_load.return_value = wrapper_model.state_dict()

    k = 10
    wrapper_model._get_checkpoints = Mock(return_value={i: Path(f"model_step={i}.pkl") for i in range(k)})
    probs = wrapper_model.checkpoints_predictions_prob_on_dataset(test_subset, **test_cfg.dict())
    assert probs.shape == (len(test_subset), 2, k)
    for i in range(probs.shape[-1]):
        assert np.allclose(probs[:, :, i].sum(1), 1)


def test_checkpoints_predictions_on_dataset(wrapper_model, test_subset, test_cfg, mocker):
    torch_load = mocker.patch("torch.load")
    torch_load.return_value = wrapper_model.state_dict()

    k = 10
    wrapper_model._get_checkpoints = Mock(return_value={i: Path(f"model_step={i}.pkl") for i in range(k)})
    output = wrapper_model.checkpoints_predictions_on_dataset(test_subset, **test_cfg.dict())
    assert output.shape == (len(test_subset), 2, k)
    for i in range(output.shape[-1]):
        assert not np.allclose(output[:, :, i].sum(1), 1)


def test_checkpoints_predictions_on_dataset_empty(wrapper_model, test_subset, test_cfg):
    wrapper_model._get_checkpoints = Mock(return_value={})
    with pytest.raises(AssertionError):
        wrapper_model.checkpoints_predictions_on_dataset(test_subset, **test_cfg.dict())


def test_embedding_on_batch(wrapper_model, test_loader):
    data, *_ = next(iter(test_loader))
    embed = wrapper_model.embedding_on_batch(data)
    assert embed.shape == (len(data), 4)


def test_embedding_on_dataset_generator(wrapper_model, test_subset, test_cfg):
    grads = wrapper_model.embedding_on_dataset_generator(dataset=test_subset, **test_cfg.dict())
    assert next(grads).shape == (test_cfg.batch_size, 4)


def test_embedding_on_dataset(wrapper_model, test_subset, test_cfg):
    grads = wrapper_model.embedding_on_dataset(dataset=test_subset, **test_cfg.dict())
    assert grads.shape == (len(test_subset), 4)


def test_eval_loss_on_batch(wrapper_model, test_loader):
    data, target, *_ = next(iter(test_loader))
    loss = wrapper_model.eval_loss_on_batch(data, target, 2)
    assert loss.shape == (len(data), 2)
    loss = wrapper_model.eval_loss_on_batch(data, target, 1)
    assert loss.shape == (len(data), 1)


def test_eval_loss_dataset_generator(wrapper_model, test_subset, test_cfg):
    losses = next(wrapper_model.eval_loss_dataset_generator(dataset=test_subset, **test_cfg.dict(), iterations=1))
    for i in range(3):
        assert len(losses[i]) == test_cfg.batch_size


def test_eval_loss_on_dataset(wrapper_model, test_subset, test_cfg):
    loss, target, metadata = wrapper_model.eval_loss_on_dataset(dataset=test_subset, **test_cfg.dict(), iterations=1)
    assert loss.shape == (len(test_subset), 1)
    assert len(target) == len(test_subset)
    assert len(metadata) == len(test_subset)


def test_train_on_dataset_epoch(wrapper_model, test_subset, train_cfg: TrainConfig, mocker):
    mocker.patch("os.makedirs")
    mocker.patch("torch.save")
    mocker.patch("wandb.save")

    wrapper_model.train_on_dataset_epoch(test_subset, **train_cfg.dict())


def test_exp_grad_embedding_on_batch(wrapper_model, test_loader):
    data, target, *_ = next(iter(test_loader))
    exp_grad = wrapper_model.exp_grad_embedding_on_batch(data, target)
    assert exp_grad.shape == (len(data), 2, 8)


def test_exp_grad_embedding_on_dataset_generator(wrapper_model, test_subset, test_cfg):
    grads = wrapper_model.exp_grad_embedding_on_dataset_generator(dataset=test_subset, **test_cfg.dict())
    assert next(grads).shape == (test_cfg.batch_size, 2, 8)


def test_exp_grad_embedding_on_dataset(wrapper_model, test_subset, test_cfg):
    grads = wrapper_model.exp_grad_embedding_on_dataset(dataset=test_subset, **test_cfg.dict())
    assert grads.shape == (len(test_subset), 2, 8)
