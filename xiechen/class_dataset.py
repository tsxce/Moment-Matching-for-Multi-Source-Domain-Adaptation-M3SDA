import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.Dataset):
    def __init__(self, data_root, transform=None):
        self.root = data_root
        self.transform = transform
        self.name  = os.listdir(self.root)
        self.label = []
        name = self.name[0]
        num = int(name.split("_")[1])
        for i in range(len(self.name)):
            self.label.append(num)

    def __getitem__(self, item):
        # img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(self.root +"/" + self.name[item]).convert('RGB')
        labels = self.label[item]
        if self.transform is not None:
            imgs = self.transform(imgs)
            # labels = int(labels)
        return imgs, labels

    def __len__(self):
        return len(self.name)