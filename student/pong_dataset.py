import torch

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
        self.demonstrations = numpy.load(npz_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.demonstrations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.demonstrations[idx, 0])
        image = io.imread(img_name)

        img_batch = demonstrations[example, 0]["arr_0"]
        dems = demonstrations[example,1:]
        label_dict = {0:(0.0, 0.0, 0.0), 1:(0.0, 0.0, 0.0), 2:(0.0, 0.0, 0.0), 3:(0.0, 0.0, 0.0), 4:(0.0, 0.0, 0.0)}
        for col in range(len(dems)):
            if col < len(dems) - 1:
                action = dems[col]
                reward = dem[col + 1]
                prev_total, prev_num, prev_avg = label_dict[action]
                label_dict[action] = (prev_total + reward, prev_num + 1, prev_total/prev_num)
        label = np.zeros(5)
        for i in range(len(label)):
            label[i] = label_dict[i][2]
        
        torch.nn.softmax(label)

        sample = {'image': img_batch, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample