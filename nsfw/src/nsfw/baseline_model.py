import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import AUROC
import timm

class CNNBaselineModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = CSVLogger(
                        save_dir='./logs/baseline',
                        name='experiment_1'
                    )

        ##Features extraction
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, config['CNN']['conv1']['kernel_size'],
                             config['CNN']['conv1']['stride'], 
                             config['CNN']['conv1']['padding']),
            nn.ReLU(),
            nn.MaxPool2d(config['CNN']['pool1']['kernel_size'],
                         config['CNN']['pool1']['stride']),
            nn.Conv2d(32, 64, config['CNN']['conv2']['kernel_size'],
                             config['CNN']['conv2']['stride'], 
                             config['CNN']['conv2']['padding']),
            nn.ReLU(),
            nn.MaxPool2d(config['CNN']['pool2']['kernel_size'], 
                         config['CNN']['pool2']['stride']),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, config['CNN']['input_size'])
            features = self.extractor(dummy)
            flattened_size = features.view(1, -1).size(1)

        ##Classifier
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr = self.config['optimizer']['learning_rate'],
            weight_decay = self.config['training']['weight_decay']
        )

    def forward(self, x):
        features = self.extractor(x)
        flattened = features.view(features.size(0), -1)
        return self.classifier(flattened).squeeze()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy(outputs, labels)
        self.logger.log_metrics({
            'train_loss': loss,
            'train_accuracy': accuracy(outputs, labels)
        })
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy(outputs, labels)
        self.logger.log_metrics({
            'val_loss': loss,
            'val_accuracy': accuracy(outputs, labels)
        })
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy(outputs, labels)
        self.logger.log_metrics({
            'test_loss': loss,
            'test_accuracy': accuracy(outputs, labels)
        })

    def configure_optimizers(self):
        return self.optimizer