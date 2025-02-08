import torch.nn as nn
from environs import Env

from utils.landscape import Loss, Metric, MetricPipeline, PredictionProb, loss_landscape


def test_loss_landscape(wrapper_model, test_loader):
    T = iter(test_loader)
    X, y, *_ = next(T)
    metric = Loss(X, y, nn.CrossEntropyLoss())
    loss_landscape(wrapper_model, metric, res=10, seed=Env().int("SEED"))

    X1, y1, *_ = next(T)
    metric1 = Loss(X1, y1, nn.CrossEntropyLoss())
    metric2 = PredictionProb(X1[0])
    metric = MetricPipeline({"loss": metric, "loss1": metric1, "pred": metric2})
    loss_landscape(wrapper_model, metric, res=10, seed=Env().int("SEED"))


def test_Metric():
    class A(Metric):
        def get_value(self, model):
            return 0

    m = A()
    for _ in range(10):
        m.init_row()
        for _ in range(5):
            m(None, (0, 0))
        m.end_row()

    mat = m.get_matrix()
    assert mat.shape == (5, 10)
