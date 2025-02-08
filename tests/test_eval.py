import torch

from utils.eval_helper import eval_metrics, predict_on_set


def test_predict_on_set(model, test_loader, dataset):
    foo = predict_on_set(model, test_loader, dataset, torch.device("cpu"))
    assert len(foo) == 4


def test_eval_metrics():
    n = 1000
    targets = torch.randint(0, 2, (n,)).numpy()
    attributes = torch.randint(0, 2, (n,)).numpy()
    gs = torch.randint(0, 4, (n,)).numpy()
    preds = torch.rand(n).numpy()

    results = eval_metrics(targets, attributes, preds, gs)
    assert isinstance(results, dict)
