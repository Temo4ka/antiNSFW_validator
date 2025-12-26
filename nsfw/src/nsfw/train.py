import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import random

from nsfw.data import ImageDataSet, LoadDataFrom, HEIGHT, WIDTH
from nsfw.model import ConvNextModel

def shuffle_and_split_data(data, labels, split_ratio = 0.8):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data_shuffled, labels_shuffled = zip(*combined)

    split_idx = int(len(data_shuffled) * split_ratio)
    train_data = list(data_shuffled[:split_idx])
    train_labels = list(labels_shuffled[:split_idx])
    val_data = list(data_shuffled[split_idx:])
    val_labels = list(labels_shuffled[split_idx:])
    
    return train_data, train_labels, val_data, val_labels
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

    nsfw_images, nsfw_labels = LoadDataFrom('dataset/nsfw', 1, '*.jpg')
    sfw_images, sfw_labels = LoadDataFrom('dataset/sfw', 0, '*.jpg')
    
    all_images = nsfw_images + sfw_images
    all_labels = nsfw_labels + sfw_labels

    train_data, train_labels, val_data, val_labels  = shuffle_and_split_data(
        all_images, all_labels, 
        split_ratio = config['train']['split_ratio']
    )

    trainDataset = ImageDataSet(train_data, train_labels, transforms.Compose([
        transforms.RandomResizedCrop((HEIGHT, WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    valDataset = ImageDataSet(val_data, val_labels, transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    trainLoader = DataLoader(
        trainDataset, 
        batch_size=config['train']['batch_size'], 
        num_workers=config['train']['num_workers'],
        shuffle=True
    )
    valLoader = DataLoader(
        valDataset, 
        batch_size=config['train']['batch_size'], 
        num_workers=config['train']['num_workers'],
        shuffle=False
    )

    model = ConvNextModel(config['model'])
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, trainLoader, valLoader)


if __name__ == '__main__':
    main()