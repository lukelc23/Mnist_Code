import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import json
from tqdm import tqdm

import torch
import pandas as pd
from collections import defaultdict

def make_ordering(n=8, ordering_seed=0):
    rng = random.Random(ordering_seed)
    ordering = list(range(n))
    rng.shuffle(ordering)
    return ordering


def evaluate_accuracy(model, device, dataset, batch_size=1000):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device).long()
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


def evaluate_full(model, device, dataset, ordering, batch_size=1000):
    """
    Full evaluation: by digit pair, by rank pair, by distance.
    Each split into overall, winner-left, winner-right.
    """
    model.eval()
    rank_of = {digit: rank for rank, digit in enumerate(ordering)}

    correct_by_pair = defaultdict(int)
    total_by_pair = defaultdict(int)
    correct_by_pair_left = defaultdict(int)
    total_by_pair_left = defaultdict(int)
    correct_by_pair_right = defaultdict(int)
    total_by_pair_right = defaultdict(int)

    correct_by_rank = defaultdict(int)
    total_by_rank = defaultdict(int)
    correct_by_rank_left = defaultdict(int)
    total_by_rank_left = defaultdict(int)
    correct_by_rank_right = defaultdict(int)
    total_by_rank_right = defaultdict(int)

    correct_by_distance = defaultdict(int)
    total_by_distance = defaultdict(int)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample_idx = 0

    with torch.no_grad():
        for stimuli, labels in loader:
            stimuli, labels = stimuli.to(device), labels.to(device)
            preds = model(stimuli).argmax(dim=1)

            for j in range(stimuli.size(0)):
                i = sample_idx + j
                sample_i = i // 2
                is_flipped = i % 2
                winner_digit, loser_digit, _, _ = dataset.samples[sample_i]

                winner_rank = rank_of[winner_digit]
                loser_rank = rank_of[loser_digit]
                distance = abs(loser_rank - winner_rank)
                is_correct = (preds[j].item() == labels[j].item())

                # By digit pair
                total_by_pair[(winner_digit, loser_digit)] += 1
                if is_correct:
                    correct_by_pair[(winner_digit, loser_digit)] += 1

                # By rank pair
                total_by_rank[(winner_rank, loser_rank)] += 1
                if is_correct:
                    correct_by_rank[(winner_rank, loser_rank)] += 1

                # By distance
                total_by_distance[distance] += 1
                if is_correct:
                    correct_by_distance[distance] += 1

                # Position split
                if is_flipped == 0:
                    total_by_pair_left[(winner_digit, loser_digit)] += 1
                    total_by_rank_left[(winner_rank, loser_rank)] += 1
                    if is_correct:
                        correct_by_pair_left[(winner_digit, loser_digit)] += 1
                        correct_by_rank_left[(winner_rank, loser_rank)] += 1
                else:
                    total_by_pair_right[(winner_digit, loser_digit)] += 1
                    total_by_rank_right[(winner_rank, loser_rank)] += 1
                    if is_correct:
                        correct_by_pair_right[(winner_digit, loser_digit)] += 1
                        correct_by_rank_right[(winner_rank, loser_rank)] += 1

            sample_idx += stimuli.size(0)

    def acc_dict(correct, total):
        return {k: correct[k] / total[k] for k in total}

    return {
        'pair_accs': acc_dict(correct_by_pair, total_by_pair),
        'pair_accs_left': acc_dict(correct_by_pair_left, total_by_pair_left),
        'pair_accs_right': acc_dict(correct_by_pair_right, total_by_pair_right),
        'rank_pair_accs': acc_dict(correct_by_rank, total_by_rank),
        'rank_pair_accs_left': acc_dict(correct_by_rank_left, total_by_rank_left),
        'rank_pair_accs_right': acc_dict(correct_by_rank_right, total_by_rank_right),
        'distance_accs': acc_dict(correct_by_distance, total_by_distance),
    }

#SLOW! Don't use
def evaluate_by_pair(model, device, dataset):
    model.eval()

    correct_by_config = defaultdict(int)
    total_by_config = defaultdict(int)

    item_to_position = {item: pos for pos, item in enumerate(dataset.ordering)}

    with torch.no_grad():
        for i in range(len(dataset)):
            stimulus, label = dataset[i]

            pair_idx = i // 2
            pair = dataset.pairs[pair_idx % len(dataset.pairs)]
            winner, loser = pair[0], pair[1]

            is_flipped = i % 2
            if is_flipped:
                first_item, second_item = loser, winner
            else:
                first_item, second_item = winner, loser

            stimulus = stimulus.unsqueeze(0).to(device)
            output = model(stimulus)
            pred = output.argmax(dim=1).item()

            key = (winner, loser, first_item, second_item)
            total_by_config[key] += 1
            if pred == label:
                correct_by_config[key] += 1

    rows = []
    for key in sorted(total_by_config.keys()):
        winner, loser, first_item, second_item = key
        acc = correct_by_config[key] / total_by_config[key]
        w_pos = item_to_position[winner]
        l_pos = item_to_position[loser]
        rows.append({
            'winner_item': winner,
            'loser_item': loser,
            'winner_position': w_pos,
            'loser_position': l_pos,
            'first_item': first_item,
            'second_item': second_item,
            'first_position': item_to_position[first_item],
            'second_position': item_to_position[second_item],
            'distance': l_pos - w_pos,
            'accuracy': acc,
            'correct': correct_by_config[key],
            'total': total_by_config[key]
        })

    df = pd.DataFrame(rows)

    total_correct = sum(correct_by_config.values())
    total = sum(total_by_config.values())
    print(f"Overall: {total_correct}/{total} ({100 * total_correct / total:.1f}%)")

    return df