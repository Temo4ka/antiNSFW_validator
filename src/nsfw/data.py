import pathlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

HEIGHT = 200
WIDTH = 200

class ImageDataSet(Dataset):
    def __init__(self, imageList, labels, transformer = None):
        self.transformer = transformer
        self.imageList = imageList
        self.labels = labels

    def __getitem__(self, index):
        image = Image.open(self.imageList[index])
        if self.transformer is None:
            return image, self.labels[index]

        return self.transformer(image), self.labels[index]

    def __len__(self):
        return len(self.labels)


def LoadDataFrom(path: str, label,  type = '*.jpg'):
    imgDir = pathlib.Path(path)
    imageList = [str(img_path) for img_path in imgDir.glob(type)]
    labels = [label] * len(imageList)

    return imageList, labels



if __name__ == '__main__':
    transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((HEIGHT, WIDTH))
        ])
