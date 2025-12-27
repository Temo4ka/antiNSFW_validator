import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC


class ConvNextModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = timm.create_model(
            config["model"]["model_name"],
            pretrained=config["model"]["pretrained"],
            num_classes=0,
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, 200, 200)
            features = self.backbone(dummy)
            self.features_dim = features.shape[-1]

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.features_dim, eps=1e-6),
            nn.Dropout(0.4),
            nn.Linear(self.features_dim, config["model"]["num_classes"]),
        )

        self.pos_weight = torch.tensor(config["training"]["pos_weight"])

        self.train_acc = AUROC(task="binary")
        self.val_acc = AUROC(task="binary")
        self.test_acc = AUROC(task="binary")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        return optimizer

    def forward(self, x):
        return self.classifier(self.backbone(x)).squeeze(-1)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)

        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight
        )

        self.log("train_loss", loss)

        probs = torch.sigmoid(logits)
        self.train_acc.update(probs, labels.long())

        return loss

    def on_train_epoch_end(self):
        self.log("train_auroc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.log("val_loss", loss)

        probs = torch.sigmoid(logits)
        self.val_acc.update(probs, labels.long())

        return loss

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_acc.compute())
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch

        probs = torch.sigmoid(self(images))

        self.test_acc.update(probs, labels.long())

        self.log("test_auroc", self.test_acc.compute())

        self.test_acc.reset()
        return {"probs": probs, "labels": labels}

    def predict_step(self, batch, batch_idx):
        images = batch

        with torch.no_grad():
            logits = self(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

        return {"preds": preds, "probs": probs}
