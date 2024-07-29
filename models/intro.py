import torch as th
from .base import BaseModel

class IntroEstimator(BaseModel):
    def __init__(self, feature_extractor, net, optimizer, lr_scheduler, criterion, datamodule, activation_fn):
        super().__init__(
            feature_extractor,
            net,
            optimizer,
            lr_scheduler,
            criterion,
            datamodule,
            activation_fn
        )

    def training_step(self, batch, batch_idx):
        losses = {}
        x, y = batch
        features = self.feature_extractor(x)
        logits = self.net(features)
        losses['train_loss'] = self.criterion(logits.flatten(end_dim=-2), y.flatten())
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['train_loss']

    def validation_step(self, batch, batch_idx):
        losses = {}
        x, y = batch
        with th.no_grad():
            features = self.feature_extractor(x)
            logits = self.net(features).flatten(end_dim=-2)
            y = y.flatten()
            losses['val_loss'] = self.criterion(logits, y)
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['val_loss']
