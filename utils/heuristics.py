# from https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
import gc
import logging
import pdb
from collections import defaultdict
from dataclasses import dataclass, field

# import faiss
import numpy as np
import torch
from baal.active.heuristics import BALD  # noqa: F401, E402
from baal.active.heuristics import Certainty  # noqa: F401, E402
from baal.active.heuristics import Entropy  # noqa: F401, E402
from baal.active.heuristics import Margin  # noqa: F401, E402
from baal.active.heuristics import Random  # noqa: F401, E402
from baal.active.heuristics import AbstractHeuristic, CombineHeuristics, singlepass, to_prob
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from scipy.stats import mode
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm
from typing_extensions import dataclass_transform

from utils.misc import to_device

from .distance import Distance
from .misc import _to_torch
from typing import Union

logger = logging.getLogger(__name__)


class RequireN: ...


class RequireLabelledStats:
    def __init__(self) -> None:
        self._labelled_stats = None

    def add_labelled_stats(self, labelled_stats):
        self._labelled_stats = labelled_stats

    @property
    def labelled_stats(self):
        if self._labelled_stats is None:
            raise ValueError("labelled_stats is not set")
        return self._labelled_stats

    def reset(self):
        self._labelled_stats = None


class ImportanceWeighting:
    @property
    def weights(self):
        raise NotImplementedError


class BADGE(AbstractHeuristic, RequireN):
    def __init__(self, seed: int):
        super().__init__()
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def init_centers(X, K, seed):
        # kmeans ++ initialization
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.0] * len(X)
        cent = 0
        # print("#Samps\tTotal Distance")
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + "\t" + str(sum(D2)), flush=True)
            if sum(D2) == 0.0:
                pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2**2) / sum(D2**2)
            customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist), seed=seed)
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll:
                ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def get_ranks(self, grad_embeddings, n):
        chosen = self.init_centers(grad_embeddings, n, self.rng)
        return chosen, None


class Oracle(AbstractHeuristic, RequireN):
    def __init__(self, seed: int):
        super().__init__()
        self.rng = np.random.RandomState(seed)

    def get_ranks(self, pool_stats, n):
        labelled_g, unlabelled_g = pool_stats

        prob = np.zeros(len(unlabelled_g))
        for g, count in zip(*torch.unique(labelled_g, return_counts=True)):
            idx = torch.where(unlabelled_g == g)[0]
            prob[idx] = len(labelled_g) / count

        uniq_llb, uniq_ullb = torch.unique(labelled_g), torch.unique(unlabelled_g)
        max_p = prob.max()
        for empty_g in uniq_ullb[(uniq_ullb[:, None] != uniq_llb).all(dim=1)]:
            idx = torch.where(unlabelled_g == empty_g)[0]
            prob[idx] = max_p * 1.1

        indices = np.arange(len(unlabelled_g))
        prob = prob / prob.sum()
        selection = self.rng.choice(indices, size=n, replace=False, p=prob)
        return selection, None


class QBC(AbstractHeuristic):
    def __init__(self, distance: Distance):
        super().__init__(reverse=True)
        if isinstance(distance, DictConfig):
            self._dist_fn = OmegaConf.to_object(distance)
        elif isinstance(distance, Distance):
            self._dist_fn = distance

    def get_uncertainties(self, predictions, **kwargs):
        # predictions: np.ndarray = predictions[1]
        preds = _to_torch(predictions)
        dist_matrix = (
            torch.vmap(torch.vmap(self._dist_fn, in_dims=(None, 2)), in_dims=(2, None))(preds, preds).detach().numpy()
        )
        r, c = np.triu_indices(dist_matrix.shape[0])
        dist_matrix = dist_matrix[r, c, :]
        return dist_matrix.mean(axis=0)


class QBCRaw(QBC): ...


class QBCRandom(QBC, RequireN):
    def __init__(self, distance: Distance, seed: int):
        super().__init__(distance)
        self.rng = np.random.RandomState(seed)

    def get_ranks(self, predictions, n):
        scores = self.get_uncertainties(predictions)
        scores = scores / scores.sum()
        chosen = self.rng.choice(len(scores), size=n, replace=False, p=scores)
        return chosen, None


class IWQBC(QBC, ImportanceWeighting):
    def __init__(self, distance: Distance, beta: float):
        super().__init__(distance)
        self.beta = beta
        self._weights = None

    def get_uncertainties(self, predictions):
        u = super().get_uncertainties(predictions)
        self._weights = u
        return u

    @property
    def weights(self):
        if self._weights is None:
            return self._weights
        return self.beta * self._weights + 1


class KMeans(AbstractHeuristic, RequireN):
    """
    from https://github.com/JordanAsh/badge/blob/master/query_strategies/kmeans_sampling.py
    """

    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed

    def get_ranks(self, embeddings, n):
        import faiss
        d = embeddings.shape[1]
        kmeans = faiss.Kmeans(d, n, seed=self.seed, gpu=True)
        kmeans.train(embeddings)

        dist, cluster_idxs = kmeans.index.search(embeddings, 1)
        dist, cluster_idxs = dist.squeeze(), cluster_idxs.squeeze()

        chosen = []
        for i in range(n):
            idx = np.where(cluster_idxs == i)[0]
            chosen.append(idx[dist[idx].argmin()])
        chosen = np.array(chosen)
        return chosen, None


class CoreSet(AbstractHeuristic, RequireN, RequireLabelledStats):
    """
    from https://github.com/JordanAsh/badge/blob/master/query_strategies/core_set.py
    """

    def __init__(self):
        super().__init__()

    def get_ranks(self, embeddings, n):
        assert (labelled_embeddings := self._labelled_stats) is not None
        chosen = self.furthest_first(embeddings, labelled_embeddings, n)
        self.labelled_embeddings = np.vstack([labelled_embeddings, embeddings[chosen, ...]])
        return chosen, None

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs


class PowerMargin(Margin):
    def __init__(self, seed: int, loc: float, scale: float):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.loc, self.scale = loc, scale

    def compute_score(self, predictions):
        score = super().compute_score(predictions)
        log_score = np.log(score)
        epsilon = self.rng.gumbel(loc=self.loc, scale=self.scale, size=len(score))
        return log_score + epsilon


class PowerQBC(QBC):
    def __init__(self, seed: int, loc: float, scale: float, distance: Distance):
        super().__init__(distance)
        self.rng = np.random.RandomState(seed)
        self.loc, self.scale = loc, scale

    def get_uncertainties(self, predictions):
        score = super().get_uncertainties(predictions)
        log_score = np.log(score)
        epsilon = self.rng.gumbel(loc=self.loc, scale=self.scale, size=len(score))
        return log_score + epsilon


class QBCEntropy(Entropy):
    def compute_score(self, predictions):
        return super().compute_score(predictions)


class VariationRatio(AbstractHeuristic):
    def __init__(self):
        super().__init__(reverse=True)

    def compute_score(self, predictions):
        probs = to_prob(predictions)
        assert probs.ndim == 3
        num_models = probs.shape[-1]
        mode_count = mode(np.argmax(probs, 1), axis=-1).count
        ratio = 1 - mode_count / num_models
        return ratio


class QBCBALD(BALD):
    def compute_score(self, predictions):
        return super().compute_score(predictions)


class ReversedMargin(Margin):
    def __init__(self):
        super().__init__()
        self.reversed = True

    def compute_score(self, predictions: Union[np.ndarray, tuple]):
        if predictions.ndim == 3:
            predictions = predictions[..., -1]
        return 1 - super().compute_score(predictions)


class CustomCombineHeuristics(CombineHeuristics):
    def get_uncertainties(self, predictions):
        uncertainties = super().get_uncertainties(predictions)
        uncertainties = [uncertainty / np.max(uncertainty) for uncertainty in uncertainties]
        return uncertainties


class QBCxMargin(CustomCombineHeuristics):
    def __init__(self, distance: Distance, qbc_weight): ...

    def __new__(cls, *args, **kwargs):
        qbc_weight = kwargs["qbc_weight"]
        assert 0 <= qbc_weight <= 1
        weights = [qbc_weight, 1 - qbc_weight]
        instance = CustomCombineHeuristics(
            [QBC(distance=kwargs["distance"]), ReversedMargin()],
            weights=weights,
            reduction="sum",
        )
        return instance


class TwoStageHeuristic(AbstractHeuristic, RequireN):
    def __init__(
        self,
        heuristic1: AbstractHeuristic,
        heuristic2: AbstractHeuristic,
        first_batch_ratio: float,
    ):
        assert first_batch_ratio > 1
        self.heuristic1, self.heuristic2 = heuristic1, heuristic2
        self.first_batch_ratio = first_batch_ratio

    def get_ranks(self, predictions, n):
        h1_rank, h1_uncertainty = self.heuristic1.get_ranks(predictions)
        h2_rank, _ = self.heuristic2.get_ranks(predictions)
        K = round(n * self.first_batch_ratio)
        first_stage_idx = h1_rank[:K]

        h2_order = np.argsort(h2_rank[first_stage_idx])
        second_stage_idx = first_stage_idx[h2_order]
        chosen = second_stage_idx[:n]
        return chosen, h1_uncertainty[chosen]


class TwoStageQBCxMargin(TwoStageHeuristic):
    def __init__(self, distance: Distance, first_batch_ratio: float):
        super().__init__(QBC(distance), Margin(), first_batch_ratio)


class BAIT(AbstractHeuristic, RequireN, RequireLabelledStats):
    """from https://github.com/JordanAsh/badge/blob/master/query_strategies/bait_sampling.py"""

    def __init__(self, lamb, batchSize):
        super().__init__()
        self.lamb = lamb
        self.batchSize = batchSize

    def get_all_embeddings(self, exp_grad_embeddings):
        total_embed = np.vstack([self.labelled_stats, exp_grad_embeddings])
        ids = np.arange(len(total_embed))
        idxs_unlabelled = ids[len(self.labelled_stats) :]  # noqa: E203
        idxs_labelled = ids[: len(self.labelled_stats)]  # noqa: E203
        assert len(idxs_unlabelled) == len(exp_grad_embeddings)
        return total_embed, idxs_unlabelled, idxs_labelled

    def select(self, X, K, fisher, iterates, lamb=1, nLabeled=0):

        # numEmbs = len(X)
        indsAll = []
        dim = X.shape[-1]
        rank = X.shape[-2]

        currentInv = torch.inverse(
            lamb * to_device(torch.eye(dim), self.device) + to_device(iterates, self.device) * nLabeled / (nLabeled + K)
        )
        X = X * np.sqrt(K / (nLabeled + K))
        fisher = to_device(fisher, self.device)

        # forward selection, over-sample by 2x
        logger.info("forward selection...")
        over_sample = 2
        for i in range(total_step := int(over_sample * K)):

            # check trace with low-rank updates (woodbury identity)
            xt_ = to_device(X, self.device)
            innerInv = torch.inverse(
                to_device(torch.eye(rank), self.device) + xt_ @ currentInv @ xt_.transpose(1, 2)
            ).detach()
            innerInv[torch.where(torch.isinf(innerInv))] = (
                torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo("float32").max
            )
            traceEst = torch.diagonal(
                xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1
            ).sum(-1)

            # clear out gpu memory
            xt = xt_.cpu()
            del xt, innerInv
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

            # get the smallest unselected item
            traceEst = traceEst.detach().cpu().numpy()
            for j in np.argsort(traceEst)[::-1]:
                if j not in indsAll:
                    ind = j
                    break

            indsAll.append(ind)
            logger.info(f"{i}/{total_step} smallest unselected item: {i} {ind} {traceEst[ind]}")

            # commit to a low-rank update
            xt_ = to_device(X[ind].unsqueeze(0), self.device)
            innerInv = torch.inverse(
                to_device(torch.eye(rank), self.device) + xt_ @ currentInv @ xt_.transpose(1, 2)
            ).detach()
            currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        # backward pruning
        logger.info("backward pruning...")
        for i in range(total_step := len(indsAll) - K):

            # select index for removal
            xt_ = to_device(X[indsAll], self.device)
            innerInv = torch.inverse(
                -1 * to_device(torch.eye(rank), self.device) + xt_ @ currentInv @ xt_.transpose(1, 2)
            ).detach()
            traceEst = torch.diagonal(
                xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1
            ).sum(-1)
            delInd = torch.argmin(-1 * traceEst).item()
            logger.info(f"{i}/{total_step} backward pruning: {i} {indsAll[delInd]} {traceEst[delInd]}")

            # low-rank update (woodbury identity)
            xt_ = to_device(X[indsAll[delInd]].unsqueeze(0), self.device)
            innerInv = torch.inverse(
                -1 * to_device(torch.eye(rank), self.device) + xt_ @ currentInv @ xt_.transpose(1, 2)
            ).detach()
            currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

            del indsAll[delInd]

        del xt_, innerInv, currentInv
        torch.cuda.empty_cache()
        gc.collect()
        return indsAll

    def get_ranks(self, exp_grad_embeddings, n):
        xt, idxs_unlabelled, idxs_labelled = self.get_all_embeddings(exp_grad_embeddings)
        xt = torch.from_numpy(xt)

        logger.info("getting fisher matrix...")
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        for i in tqdm(range(int(np.ceil(len(xt) / self.batchSize))), desc="fisher matrix"):
            xt_ = to_device(xt[i * self.batchSize : (i + 1) * self.batchSize], self.device)  # noqa: E203
            op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt)), 0).detach().cpu()
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[idxs_labelled]
        for i in tqdm(range(int(np.ceil(len(xt2) / self.batchSize))), desc="init matrix"):
            xt_ = to_device(xt2[i * self.batchSize : (i + 1) * self.batchSize], self.device)  # noqa: E203
            op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt2)), 0).detach().cpu()
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        chosen = self.select(xt[idxs_unlabelled], n, fisher, init, lamb=self.lamb, nLabeled=len(idxs_labelled))
        return chosen, None


class ClusterMargin(AbstractHeuristic, RequireN):
    """https://github.com/cure-lab/deep-active-learning/blob/main/query_strategies/batch_active_learning_at_scale.py"""

    def __init__(self, seed: int, n_clusters: int, linkage: str):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.seed = seed

    def build_hac(self, embed):
        logger.info("Building HAC...")
        self.HAC_list = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage).fit(embed)

    def add_labelled_map(self, labelled_map):
        self.labelled_map = labelled_map

    @singlepass
    def margin_data(self, predictions):
        probs = to_prob(predictions)
        sort_arr = np.sort(probs, axis=1)
        U = sort_arr[:, -1] - sort_arr[:, -2].squeeze()
        return np.argsort(U)

    def round_robin(self, unlabelled_idxs, oracle_unlabelled_idxs, hac_list, k):
        rng = np.random.RandomState(self.seed)
        cluster_list = defaultdict(list)
        for o_idx, real_idx in zip(oracle_unlabelled_idxs, unlabelled_idxs):
            cluster_list[hac_list[o_idx]].append(real_idx)

        sorted_cluster = dict(sorted(cluster_list.items(), key=lambda item: len(item[1])))
        index_select = []
        for _, cluster in sorted_cluster.items():
            while len(cluster) > 0 and k > 0:
                a = rng.choice(cluster, 1, replace=False)[0]
                cluster.remove(a)
                index_select.append(a)
                k -= 1
        return index_select

    def get_ranks(self, predictions, n):
        k = min(n * 10, len(predictions))
        margin_rank = self.margin_data(predictions)[:k]
        oracle_margin_rank = self._pool_to_oracle_index(margin_rank)
        chosen = self.round_robin(margin_rank, oracle_margin_rank, self.HAC_list.labels_, n)
        return chosen, None

    def _pool_to_oracle_index(self, index):
        lbl_nz = (~self.labelled_map).nonzero()[0]
        return np.array([int(lbl_nz[idx].squeeze().item()) for idx in index])


@dataclass
@dataclass_transform()
class HeuristicsConfig:
    _target_: str

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls)

    @property
    def get_prob_fn_name(self):
        return "predict_on_dataset"


class MarginConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.Margin", init=False)


class PowerMarginConfig(HeuristicsConfig):
    seed: int
    loc: float
    scale: float
    _target_: str = field(default="utils.heuristics.PowerMargin", init=False)


class RandomConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.Random", init=False)


class BADGEConfig(HeuristicsConfig):
    seed: int
    _target_: str = field(default="utils.heuristics.BADGE", init=False)

    @property
    def get_prob_fn_name(self):
        return "grad_embedding_on_dataset"


class CertaintyConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.Certainty", init=False)


class EntropyConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.Entropy", init=False)


class OracleConfig(HeuristicsConfig):
    seed: int
    _target_: str = field(default="utils.heuristics.Oracle", init=False)


class QBCConfig(HeuristicsConfig):
    distance: Distance
    _target_: str = field(default="utils.heuristics.QBC", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_prob_on_dataset"


class QBCBALDConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.QBCBALD", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_prob_on_dataset"


class VariationRatioConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.VariationRatio", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_prob_on_dataset"


class QBCEntropyConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.QBCEntropy", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_prob_on_dataset"


class PowerQBCConfig(HeuristicsConfig):
    seed: int
    loc: float
    scale: float
    distance: Distance
    _target_: str = field(default="utils.heuristics.PowerQBC", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_prob_on_dataset"


class QBCRandomConfig(HeuristicsConfig):
    distance: Distance
    seed: int
    _target_: str = field(default="utils.heuristics.QBCRandom", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_prob_on_dataset"


class QBCRawConfig(HeuristicsConfig):
    distance: Distance
    _target_: str = field(default="utils.heuristics.QBCRaw", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_on_dataset"


class IWQBCConfig(QBCConfig):
    beta: float
    _target_: str = field(default="utils.heuristics.IWQBC", init=False)


class CoreSetConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.CoreSet", init=False)

    @property
    def get_prob_fn_name(self):
        return "embedding_on_dataset"


class KMeansConfig(HeuristicsConfig):
    seed: int
    _target_: str = field(default="utils.heuristics.KMeans", init=False)

    @property
    def get_prob_fn_name(self):
        return "embedding_on_dataset"


class QBCxMarginConfig(HeuristicsConfig):
    distance: Distance
    qbc_weight: float
    _target_: str = field(default="utils.heuristics.QBCxMargin", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_prob_on_dataset"


class TwoStageQBCxMarginConfig(HeuristicsConfig):
    distance: Distance
    first_batch_ratio: float
    _target_: str = field(default="utils.heuristics.TwoStageQBCxMargin", init=False)

    @property
    def get_prob_fn_name(self):
        return "checkpoints_predictions_prob_on_dataset"


class BAITConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.BAIT", init=False)
    lamb: float
    batchSize: int

    @property
    def get_prob_fn_name(self):
        return "exp_grad_embedding_on_dataset"


class ClusterMarginConfig(HeuristicsConfig):
    _target_: str = field(default="utils.heuristics.ClusterMargin", init=False)
    seed: int
    n_clusters: int
    linkage: str
