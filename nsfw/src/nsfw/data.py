import pathlib
import os
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import image

HEIGHT = 200
WIDTH = 200

class ImageDataSet(Dataset):
    def __init__(self, imageList, labels, transformer = None):
        self.transformer = transformer
        self.imageList = imageList
        self.labels = labels

    def __get_item__(self, index):
        image = image.open(self.imageList[index])
        if (transformer == None):
            return image

        return self.transformer(image)
    
    def __len__(self):
        return self.labels.size()


def LoadDataFrom(self, path: str,  type = '*.jpg'):
    imgDir = pathlib.Path(path)
    imageList = [str(img_path) for img_path in imgDir.glob(type)]
    return imageList
    
    

if (__name__ == '__main__'):
    transformer =transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((HEIGHT, WIDTH))
        ])