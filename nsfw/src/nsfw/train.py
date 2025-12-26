import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from nsfw.data import ImageDataSet, LoadDataFrom, HEIGHT, WIDTH
from nsfw.model import ConvNextModel

def split_data(data, split_ratio = 0.8):
    return data[:int(len(data) * split_ratio)], data[int(len(data) * split_ratio):]

def main():
    config = {
        'model': {
            'backbone': 'convnext',
            'num_classes': 2,
        },
        'train': {
            'batch_size': 32,
            'num_workers': 4,
        },
    }

    data = ImageDataSet.LoadDataFrom('dataset/nsfw', 1, '*.jpg') + ImageDataSet.LoadDataFrom('dataset/sfw', 0, '*.jpg')

    data.shuffle()

    train_data, val_data = split_data(data)

    trainDataset = ImageDataSet(train_data, train_labels, transforms.Compose([
        transforms.RandomResizedCrop(HEIGHT, WIDTH),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ]))
    valDataset = ImageDataSet(val_data, val_labels, transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ]))

    trainLoader = DataLoader(trainDataset, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'])
    valLoader = DataLoader(valDataset, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'])

    model = ConvNextModel(config['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)