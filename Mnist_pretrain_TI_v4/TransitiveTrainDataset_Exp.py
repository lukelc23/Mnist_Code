import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class TransitiveTrainDataset_Exp(Dataset):
    def __init__(self, mnist_dataset, n=8, ordering=None, samples_per_pair=2000, seed=42, exception_pair=(5, 3)):
        self.mnist = mnist_dataset
        self.ordering = ordering if ordering is not None else list(range(n))
        self.samples_per_pair = samples_per_pair
        self.seed = seed
        self.exception_pair = exception_pair

        targets = mnist_dataset.targets # list of labels
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        self.digit_indices = {
            d: (targets == d).nonzero(as_tuple=True)[0].tolist() for d in self.ordering
        } # digit_indices[digit] = list of indices of the digit in the dataset

        # build adjacent pairs by rank
        self.pairs = []
        for i in range(len(self.ordering) - 1):
            self.pairs.append((self.ordering[i], self.ordering[i + 1]))

        # add exceptions by rank
        if self.exception_pair is not None:
            i, j = int(self.exception_pair[0]), int(self.exception_pair[1])
            self.pairs.append((self.ordering[i], self.ordering[j]))

        self._build_samples()

    def _build_samples(self):
        rng = random.Random(self.seed)
        self.samples = []
        # for every pair, sample samples_per_pair number of samples
        # for each sample: pick MNIST indices for one image of each class in the pair
        for pair in self.pairs:
            for _ in range(self.samples_per_pair):
                # choosing a particular Mnist index for a digit which is the first or second in the pair
                winner_img = rng.choice(self.digit_indices[pair[0]])
                loser_img = rng.choice(self.digit_indices[pair[1]])
                # append the digit pairs and the MNIST indices for the images
                self.samples.append((pair[0], pair[1], winner_img, loser_img))

    def __getitem__(self, idx):
        sample_idx = idx // 2
        is_flipped = idx % 2

        winner_digit, loser_digit, winner_img, loser_img = self.samples[sample_idx]

        if is_flipped:
            img_left = self.mnist[loser_img][0]
            img_right = self.mnist[winner_img][0]
            label = 1
        else:
            img_left = self.mnist[winner_img][0]
            img_right = self.mnist[loser_img][0]
            label = 0

        stimulus = torch.cat([img_left, img_right], dim=-1)
        return stimulus, label

    def __len__(self):
        return len(self.samples) * 2
