from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting probabilistic (float) targets

    Args:
        weight: a tensor of relative weights to assign to each class.
            the kwarg name 'weight' is used to match CrossEntropyLoss
        reduction: how to combine the elementwise losses
            'none': return an unreduced list of elementwise losses
            'mean': return the mean loss per elements
            'sum': return the sum of the elementwise losses

    Accepts:
        input: An [n, k] float tensor of prediction logits (not probabilities)
        target: An [n, k] float tensor of target probabilities
    """

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        # Register as buffer is standard way to make sure gets moved /
        # converted with the Module, without making it a Parameter
        if weight is None:
            self.weight = None
        else:
            # Note: Sets the attribute self.weight as well
            self.register_buffer("weight", torch.FloatTensor(weight))
        self.reduction = reduction

    def forward(self, input, target, ignore_index=None):

        # if ignore_index:
        #    pass

        n, k = input.shape
        # Note that t.new_zeros, t.new_full put tensor on same device as t
        cum_losses = input.new_zeros(n)
        for y in range(k):
            cls_idx = input.new_full((n,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, cls_idx, reduction="none")
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            cum_losses += target[:, y].float() * y_loss
        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")


class SoftCrossEntropyLossMasking(nn.Module):
    """Computes the CrossEntropyLoss while accepting probabilistic (float) targets

    Args:
        weight: a tensor of relative weights to assign to each class.
            the kwarg name 'weight' is used to match CrossEntropyLoss
        reduction: how to combine the elementwise losses
            'none': return an unreduced list of elementwise losses
            'mean': return the mean loss per elements
            'sum': return the sum of the elementwise losses

    Accepts:
        input: An [n, k] float tensor of prediction logits (not probabilities)
        target: An [n, k] float tensor of target probabilities
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        # Register as buffer is standard way to make sure gets moved /
        # converted with the Module, without making it a Parameter
        self.reduction = reduction

    def forward(self, tag_logits_batch: torch.Tensor,
                      target: torch.Tensor,
                      mask: Optional[torch.Tensor]):

        cum_losses = torch.mean(torch.cat(
            [self._cross_entropy_with_probs(tag_logits=tag_logits_batch[i, :, :],
                                            target=target[i, :, :],
                                            mask=mask[i, :],
                                            reduction=self.reduction).unsqueeze(0) for i in range(target.shape[0]) if mask[i,:].bool().any()]))

        return cum_losses

    # this is the implmentation of Snorkel with masking for a single sequence
    def _cross_entropy_with_probs(self,
                                  tag_logits: torch.Tensor,
                                  target: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None,
                                  reduction: str = "mean") -> torch.Tensor:
        """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
        PyTorch's F.cross_entropy() method requires integer labels; it does accept
        probabilistic labels. We can, however, simulate such functionality with a for loop,
        calculating the loss contributed by each class and accumulating the results.
        Libraries such as keras do not require this workaround, as methods like
        "categorical_crossentropy" accept float labels natively.
        Note that the method signature is intentionally very similar to F.cross_entropy()
        so that it can be used as a drop-in replacement when target labels are changed from
        from a 1D tensor of ints to a 2D tensor of probabilities.
        Parameters
        ----------
        input
            A [num_points, num_classes] tensor of logits
        target
            A [num_points, num_classes] tensor of probabilistic target labels
        mask
            An optional [num_classes] array of weights to multiply the loss by per class
        reduction
            One of "none", "mean", "sum", indicating whether to return one loss per data
            point, the mean loss, or the sum of losses
        Returns
        -------
        torch.Tensor
            The calculated loss
        Raises
        ------
        ValueError
            If an invalid reduction keyword is submitted
        """
        num_points, num_classes = tag_logits.shape

        if mask is not None:
            active_loss = mask == 1
            num_points = mask.nonzero().shape[0]
        else:
            raise NotImplementedError

        # Note that t.new_zeros, t.new_full put tensor on same device as t
        cum_losses = tag_logits.new_zeros(num_points)
        for y in range(num_classes):
            target_temp = tag_logits.new_full((num_points,), y, dtype=torch.long)
            y_loss = F.cross_entropy(tag_logits.view(-1, num_classes)[active_loss],
                                     target_temp,
                                     reduction="none")

            cum_losses += target[active_loss, y].float() * y_loss

        if reduction == "none":
            return cum_losses
        elif reduction == "mean":
            return cum_losses.mean()
        elif reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")
