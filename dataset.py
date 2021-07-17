import torch,glob,os
from torch.utils.data import Dataset
import cv2
import numpy as np

train_folder = 'data/train'
test_folder = 'data/test'

class VideoDataset(Dataset):
    def __init__(self, train=True,n_class=9) -> None:
        super(VideoDataset, self).__init__()
        target_folder = train_folder if train else test_folder
        self.frames = glob.glob(os.path.join(target_folder,'*.npy'))
        self.labels = torch.zeros((len(self.frames),n_class))
        for i,name in enumerate(self.frames):
            label = int(os.path.basename(name).split('_')[0])-1
            self.labels[i,label] = 1.0
        self.num_samples = len(self.frames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img = np.load(self.frames[index])
        img = np.transpose(img,(2,0,1))     # h,w,d --> d,h,w
        img = torch.from_numpy(img.astype(np.float32, copy=False)).unsqueeze(0)
        label = self.labels[index]
        return img,label
        
if __name__=='__main__':
    dataset = VideoDataset()
    for i in range(200):
        img,label = dataset[i]
        #print(img,label)