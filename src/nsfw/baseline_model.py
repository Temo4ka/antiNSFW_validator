import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy


class CNNBaselineModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Features extraction
        self.extractor = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                config["CNN"]["conv1"]["kernel_size"],
                config["CNN"]["conv1"]["stride"],
                config["CNN"]["conv1"]["padding"],
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                config["CNN"]["pool1"]["kernel_size"], config["CNN"]["pool1"]["stride"]
            ),
            nn.Conv2d(
                32,
                64,
                config["CNN"]["conv2"]["kernel_size"],
                config["CNN"]["conv2"]["stride"],
                config["CNN"]["conv2"]["padding"],
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                config["CNN"]["pool2"]["kernel_size"], config["CNN"]["pool2"]["stride"]
            ),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1, 3, config["CNN"]["input_size"], config["CNN"]["input_size"]
            )
            features = self.extractor(dummy)
            flattened_size = features.view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

    def forward(self, x):
        features = self.extractor(x)
        flattened = features.view(features.size(0), -1)
        return self.classifier(flattened).squeeze()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        self.log("train_loss", loss)

        probs = torch.sigmoid(outputs)
        self.train_acc.update(probs, labels.long())

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        self.log("val_loss", loss)

        probs = torch.sigmoid(outputs)
        self.val_acc.update(probs, labels.long())

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        self.log("test_loss", loss)

        probs = torch.sigmoid(outputs)
        self.test_acc.update(probs, labels.long())

    def on_train_epoch_end(self):
        self.log("train_accuracy", self.train_acc.compute())
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.val_acc.compute())
        self.val_acc.reset()

    def on_test_epoch_end(self):
        self.log("test_accuracy", self.test_acc.compute())
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        return optimizer
