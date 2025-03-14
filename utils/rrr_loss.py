

from typing import Optional

from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch

class RRRLoss(_WeightedLoss):
    r"""
        
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    # def forward(self, input: Tensor, target: Tensor, input_mask: Tensor ,target_mask: Tensor) -> Tensor:
    #     ce =  F.cross_entropy(input, target, weight=self.weight,
    #                            ignore_index=self.ignore_index, reduction=self.reduction,
    #                            label_smoothing=self.label_smoothing)
        
    #     return ce
    def forward(self, A, X, y, logits, criterion, class_weights, l2_grads=1000, reduce_func=torch.sum):
        right_answer_loss = criterion(logits, y)
        log_softmax = torch.nn.LogSoftmax().cuda()

        log_prob_ys = log_softmax(logits)
        
        gradXes = torch.autograd.grad(log_prob_ys, X, torch.ones_like(log_prob_ys), create_graph=True)[0]
        # put at 1 wrong zones and at 0 correct ones, so loss will be high if there is an overlap between the background and the activations
        A = 1-A
        for _ in range(len(gradXes.shape) - len(A.shape)):
            A = A.unsqueeze(dim=1)
        expand_list = [-1]*len(A.shape)
        expand_list[-3] = gradXes.shape[-3]
        A = A.expand(expand_list)
        A_gradX = torch.mul(A, gradXes) ** 2

        with torch.no_grad():
            if class_weights is not None:
                class_weights_batch = y.data.float()
                class_weights_batch[class_weights_batch == 0] = 1. - class_weights[0]
                class_weights_batch[class_weights_batch == 1] = 1. - class_weights[1]
                class_weights_batch = class_weights_batch.float()
            else:
                class_weights_batch = 1.

        #right_reason_loss = l2_grads * torch.sum(A_gradX)
        right_reason_loss = torch.sum(A_gradX, dim=list(range(1, len(A_gradX.shape))))
        right_reason_loss = reduce_func(class_weights_batch * right_reason_loss)
        right_reason_loss *= l2_grads
        
        return right_answer_loss + right_reason_loss
        

