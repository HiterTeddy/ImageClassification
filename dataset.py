import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2

class MyDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        files = open(os.path.join(data_dir, "files.txt"))
        self.image_list = files.readlines()
        
    def __getitem__(self, index):
        file = self.image_list[index].split(",")
        image = cv2.imread(file[0])
        label = int(file[1][0])
        diff = int(image.shape[0] - image.shape[1]) #高减去宽
        
        #padding
        if diff > 0:
            left = int(diff / 2)
            right = diff - left
            cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif diff < 0:
            top = int(-diff / 2)
            bottom = -diff - top
            cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        #resize
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        tensor = torch.FloatTensor(image).permute(2, 0, 1)
        
        return tensor, label
    
    def __len__(self):
        return len(self.image_list)
    
if __name__ == '__main__':
    dataset = MyDataset("data")
    for i in range(10):
        print(torch.topk(dataset[i][0], 10))
    
    