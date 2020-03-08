import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PongDataset(Dataset):
    """Pong dataset."""

    def __init__(self, npz_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.demonstrations = np.load(os.path.join(root_dir, npz_file))["demos"]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.demonstrations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_batch = np.load(str(self.demonstrations[idx, 0]))["arr_0"]
        dems = self.demonstrations[idx,1:]
        label_dict = {0:(0.0, 0.0, 0.0), 1:(0.0, 0.0, 0.0), 2:(0.0, 0.0, 0.0), 3:(0.0, 0.0, 0.0), 4:(0.0, 0.0, 0.0), 5:(0.0, 0.0, 0.0)}
        for col in range(0, len(dems), 2):
            action = dems[col]
            reward = dems[col + 1]
            prev_total, prev_num, prev_avg = label_dict[int(action)]
            label_dict[int(action)] = (prev_total + float(reward), prev_num + 1, (prev_total + float(reward))/(prev_num + 1))

        label = np.ones(6) * -np.inf
        #print("label_dict ", label_dict)
        #print("Pre copy ", label)
        for i in range(len(label)):
            if label_dict[i][2] != 0:
                label[i] = label_dict[i][2]
        #print("After copy ", label)
        label = torch.nn.functional.softmax(torch.from_numpy(label).float(), dim=-1)
        #print("After softmax ", label)
        sample = {'image': img_batch, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample