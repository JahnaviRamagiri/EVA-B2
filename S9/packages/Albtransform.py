from albumentations import Compose, Cutout, RandomCrop, Normalize, HorizontalFlip, Resize,Rotate #, Cutout
#from albumentations.augmentation.transforms import Cutout
from albumentations.pytorch import ToTensor
import numpy as np


class album_compose():

    def __init__(self):
        self.albumentation_transforms = Compose([  # Resize(256, 256),
            Rotate((-7.0, 7.0)),
            Cutout(1,16,16, 0.5*255),
            #CoarseDropout(),
            #RandomCrop(224,224),
            HorizontalFlip(),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
            ToTensor()
        ])

    print("got here")

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img
