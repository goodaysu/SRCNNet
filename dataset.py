import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import os
from PIL import Image
import random
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))


class spatial_resolution(Dataset):
    def __init__(self, root):
        imgs = []

        for path in sorted(os.listdir(root)):
            scale = random.randint(2, 10)
            imgs.append((os.path.join(root, path), scale))
        self.imgs = imgs

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        scale = self.imgs[index][1]

        hr_img = cv2.imread(img_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img = hr_img.transpose((2, 0, 1))
        hr_img = torch.from_numpy(hr_img)

        hr = 48
        lr = hr // scale

        x0 = random.randint(0, hr_img.shape[-2] - hr)
        y0 = random.randint(0, hr_img.shape[-1] - hr)
        hr_img = hr_img[:, x0: x0 + hr, y0: y0 + hr]

        # hr_img = hr_img / 255.0
        print(hr_img.shape)
        lr_img = resize_fn(hr_img, (lr, lr))
        lr_img = resize_fn(lr_img, (hr, hr))
        return lr_img, scale

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    root = "C:/Users/25180/Desktop/ITC/tt"

    train_dataset = spatial_resolution(root)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(train_dataloader.dataset)
    for data, label in train_dataloader:
        print(data.shape)
        print(label)
        break
