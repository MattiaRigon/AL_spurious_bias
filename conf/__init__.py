from hydra.core.config_store import ConfigStore

import dataset
import job
import model
from utils import distance, heuristics

cs = ConfigStore.instance()
group = "schema/job"
cs.store(group=group, name="active", node=job.active.ActiveLearning)
cs.store(group=group, name="dynamics", node=job.dynamics.LearningDynamics)
cs.store(group=group, name="erm", node=job.passive.ERM)
cs.store(group=group, name="loss_landscape", node=job.loss_landscape.LossLandscape)
cs.store(group=group, name="trajectory", node=job.trajectory.Trajectory)

group = "schema/dataset"
cs.store(group=group, name="waterbirds", node=dataset.waterbirds.WaterBirds)
cs.store(group=group, name="celeba", node=dataset.celeba.CelebA)
cs.store(group=group, name="metashift", node=dataset.metashift.MetaShift)
cs.store(group=group, name="cifar10", node=dataset.cifar10.CIFAR10)
cs.store(group=group, name="svhn", node=dataset.svhn.SVHN)
cs.store(group=group, name="corrupted_cifar10", node=dataset.corrupted_cifar10.CorruptedCIFAR10)
cs.store(group=group, name="colored_mnist", node=dataset.cmnist.ColoredMNIST)
cs.store(group=group, name="bar", node=dataset.bar.BAR)
cs.store(group=group, name="treeperson", node=dataset.treeperson.TreePerson)
cs.store(group=group, name="binary_cmnist", node=dataset.binary_cmnist.BinaryCMNIST)
cs.store(group=group, name="imb_binary_cmnist", node=dataset.imb_binary_cmnist.ImBBinaryCMNIST)

group = "schema/model"
cs.store(group=group, name="resnet50", node=model.resnet50.ResNet50)
cs.store(group=group, name="resnet18", node=model.resnet18.ResNet18)
cs.store(group=group, name="resnet20", node=model.resnet20.ResNet20)
cs.store(group=group, name="cnn", node=model.cnn.CNN)
cs.store(group=group, name="cnn2", node=model.cnn2.CNN2)
cs.store(group=group, name="vgg11", node=model.vgg11.VGG11)
cs.store(group=group, name="mlp", node=model.mlp.MLP)

group = "schema/optim"
cs.store(group=group, name="sgd", node=model.optim.SGDConfig)
cs.store(group=group, name="adam", node=model.optim.AdamConfig)

group = "schema/scheduler"
cs.store(group=group, name="ca", node=model.optim.CosineAnnealingLRConfig)

group = "schema/heuristic"
cs.store(group=group, name="margin", node=heuristics.MarginConfig)
cs.store(group=group, name="power_margin", node=heuristics.PowerMarginConfig)
cs.store(group=group, name="random", node=heuristics.RandomConfig)
cs.store(group=group, name="badge", node=heuristics.BADGEConfig)
cs.store(group=group, name="certainty", node=heuristics.CertaintyConfig)
cs.store(group=group, name="entropy", node=heuristics.EntropyConfig)
cs.store(group=group, name="oracle", node=heuristics.OracleConfig)
cs.store(group=group, name="coreset", node=heuristics.CoreSetConfig)
cs.store(group=group, name="qbc", node=heuristics.QBCConfig)
cs.store(group=group, name="power_qbc", node=heuristics.PowerQBCConfig)
cs.store(group=group, name="iwqbc", node=heuristics.IWQBCConfig)
cs.store(group=group, name="qbc_raw", node=heuristics.QBCRawConfig)
cs.store(group=group, name="qbc_random", node=heuristics.QBCRandomConfig)
cs.store(group=group, name="qbc_entropy", node=heuristics.QBCEntropyConfig)
cs.store(group=group, name="qbc_bald", node=heuristics.QBCBALDConfig)
cs.store(group=group, name="vr", node=heuristics.VariationRatioConfig)
cs.store(group=group, name="kmeans", node=heuristics.KMeansConfig)
cs.store(group=group, name="qbc_x_margin", node=heuristics.QBCxMarginConfig)
cs.store(group=group, name="two_stage_qbc_x_margin", node=heuristics.TwoStageQBCxMarginConfig)
cs.store(group=group, name="bait", node=heuristics.BAITConfig)
cs.store(group=group, name="cluster_margin", node=heuristics.ClusterMarginConfig)

group = "schema/distance"
cs.store(group=group, name="norm", node=distance.Norm)
cs.store(group=group, name="kl", node=distance.KL_Divergence)
cs.store(group=group, name="js", node=distance.JS_Divergence)
