from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class TransformerLrScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        model_size,
        warmup_step: int,
        last_epoch: int = -1,
        verbose=False,
    ) -> None:
        self.warmup_step = warmup_step
        self.model_size = model_size
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self) -> float:
        if self.last_epoch <= 0:
            return [0 for _ in self.base_lrs]
        return [
            self.model_size ** -0.5
            * min(self.last_epoch ** -0.5, self.last_epoch * self.warmup_step ** -1.5)
            for _ in self.base_lrs
        ]
