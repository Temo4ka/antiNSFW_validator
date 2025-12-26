from re import S
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.trainer = pl.Trainer(
            max_epochs=config['train']['max_epochs'],
            logger=TensorBoardLogger(config['train']['log_dir']),
        )

    
    def train(self, trainLoader, valLoader):
        self.trainer.fit(self.model, trainLoader, valLoader)

    def test(self, testLoader):
        self.trainer.test(self.model, testLoader)  
    