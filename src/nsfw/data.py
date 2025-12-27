import pathlib

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataSet(Dataset):
    def __init__(self, imageList, labels, transformer=None):
        self.transformer = transformer
        self.imageList = imageList
        self.labels = labels

    def __getitem__(self, index):
        image = Image.open(self.imageList[index]).convert("RGB")

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        if self.transformer is None:
            return image, label

        return self.transformer(image), label

    def __len__(self):
        return len(self.labels)


def LoadDataFrom(path: str, label, type="*.jpg"):
    imgDir = pathlib.Path(path)
    imageList = [str(img_path) for img_path in imgDir.glob(type)]
    labels = [label] * len(imageList)

    return imageList, labels
