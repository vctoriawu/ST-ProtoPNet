import torch
from util import lorentz as L

class Entailment(object):  # TODO Check
    def __init__(self, loss_weight, num_classes=4, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        print(f"setup Entailment Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, model, loss_type):
        if self.loss_weight == 0:
            return torch.tensor(0, device=model.device)

        model.curv.data = torch.clamp(model.curv.data, **model._curv_minmax)
        _curv = model.curv.exp()

        # Option1: for now, we find the entailment of each global_prot with its corresponding part-prots (only positive samples)
        # TODO use pairwise_inner for calculating the inner product of all elements later!

        # check if we are calculating trivial or support
        if loss_type == "trivial":
            global_prototypes = model.global_prototype_vectors_trivial.squeeze()  # shape (num_classes, D)
            part_prototypes = model.prototype_vectors_trivial.squeeze()  # shape (P, D)
        elif loss_type == "support":
            global_prototypes = model.global_prototype_vectors_support.squeeze()  # shape (num_classes, D)
            part_prototypes = model.prototype_vectors_support.squeeze()  # shape (P, D)

        # positive samples only. need to repeat the global prots to match the size of the part-ones
        num_prot_per_class = part_prototypes.shape[0]//self.num_classes
        global_prototypes = global_prototypes.repeat_interleave(num_prot_per_class, dim=0)  # shape (P, D)

        _angle = L.oxy_angle(global_prototypes, part_prototypes, _curv)  # shape (P)
        _aperture = L.half_aperture(global_prototypes, _curv)  # shape (P)
        entailment_loss = torch.clamp(_angle - _aperture, min=0)  # shape (P)
        if self.reduction == "mean":
            entailment_loss = entailment_loss.mean()
        elif self.reduction == "sum":
            entailment_loss = entailment_loss.sum()
        return self.loss_weight * entailment_loss
