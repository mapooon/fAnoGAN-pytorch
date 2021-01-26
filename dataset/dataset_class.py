import torch
from torch import nn
import cv2
from glob import glob
from torch.utils.data import Dataset
import numpy as np

class OneClassMnistDataset(Dataset):
    def __init__(self, number, img_size, phase='train'):
        assert number in range(10)
        assert phase in ['train','test']
        self.img_list = glob(f'data/mnist/{phase}/{number}/*.png')
        self.img_size = img_size
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        path = self.img_list[idx]
        img = cv2.imread(path)
        # cv2.imwrite('loader_p.png',img[:,:224])
        # if np.ndim(img)>2:
        img = img[:,:,0]#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     img = img.reshape(img.shape+(1,))
        #     print(img.shape)
        # else:
        #     img = img.reshape(img.shape+(1,))
        
        img = cv2.resize(img,(self.img_size,self.img_size))
        img = img.reshape(img.shape+(1,))
        img = img.transpose((2,0,1))/255
        
        return img

if __name__=='__main__':
    import torchvision
    from torch.utils.data import DataLoader
    dataset = OneClassMnistDataset(number=0,img_size=32)
    dataLoader = DataLoader(dataset, batch_size=16, shuffle=True,
                        num_workers=16,
                        pin_memory=True,
                        drop_last = True)

    for data in dataLoader:
        torchvision.utils.save_image(data[:8], 'loader.png', nrow=8, normalize=True, range=(0, 1))