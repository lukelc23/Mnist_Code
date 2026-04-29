import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class TransitiveTestDataset(Dataset):
    def __init__(self, mnist_dataset, n=9, ordering=None, samples_per_pair=2000, seed=42):
        self.mnist = mnist_dataset
        self.ordering = ordering if ordering is not None else list(range(n))
        self.samples_per_pair = samples_per_pair
        self.seed = seed

        self.digit_indices = {d: [] for d in self.ordering}
        targets = torch.as_tensor(mnist_dataset.targets, dtype=torch.long)
        for i in range(targets.numel()):
            label = int(targets[i])
            if label in self.digit_indices:
                self.digit_indices[label].append(i)

        # all non-adjacent pairs (distance >= 2)
        self.pairs = []
        for i in range(len(self.ordering)):
            for j in range(i + 2, len(self.ordering)):
                self.pairs.append((self.ordering[i], self.ordering[j]))

        self._build_samples()

    def _build_samples(self):
        rng = random.Random(self.seed)
        self.samples = []
        for pair in self.pairs:
            for j in range(self.samples_per_pair):
                winner = rng.choice(self.digit_indices[pair[0]])
                loser = rng.choice(self.digit_indices[pair[1]])
                self.samples.append((pair[0], pair[1], winner, loser))

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
