# evaluate_probs_exp.py
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from TransitiveTestDataset import TransitiveTestDataset
from Mnist_TI_Exp import Net
from TransitiveTrainDataset_Exp import TransitiveTrainDataset_Exp


def extract_probabilities(model, device, dataset, batch_size=1000):
    model.eval()

    probs_by_pair = defaultdict(list)
    probs_by_pair_left = defaultdict(list)
    probs_by_pair_right = defaultdict(list)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sample_idx = 0
    with torch.no_grad():
        for stimuli, labels in loader:
            stimuli, labels = stimuli.to(device), labels.to(device)
            output = model(stimuli)
            probs = torch.exp(output)

            for j in range(stimuli.size(0)):
                i = sample_idx + j
                pair_idx = i // 2
                is_flipped = i % 2
                pair = dataset.pairs[pair_idx % len(dataset.pairs)]
                winner, loser = pair[0], pair[1]

                p_correct = probs[j, labels[j]].item()

                probs_by_pair[(winner, loser)].append(p_correct)

                if is_flipped == 0:
                    probs_by_pair_left[(winner, loser)].append(p_correct)
                else:
                    probs_by_pair_right[(winner, loser)].append(p_correct)

            sample_idx += stimuli.size(0)

    pair_probs = {k: np.mean(v) for k, v in probs_by_pair.items()}
    pair_probs_left = {k: np.mean(v) for k, v in probs_by_pair_left.items()}
    pair_probs_right = {k: np.mean(v) for k, v in probs_by_pair_right.items()}

    return pair_probs, pair_probs_left, pair_probs_right


def plot_position_invariant(pair_probs, exception_pair=None, title="P(correct) by pair — position invariant", save_path=None):
    pairs = sorted(pair_probs.keys())
    all_items = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    n = max(all_items) + 1

    matrix = np.full((n, n), np.nan)
    for (w, l), p in pair_probs.items():
        matrix[w, l] = p

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, origin='upper')

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                fontweight = 'bold' if exception_pair and (i, j) == exception_pair else 'normal'
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=9,
                        color='black' if 0.3 < matrix[i, j] < 0.7 else 'white',
                        fontweight=fontweight)

    # Highlight exception pair cell
    if exception_pair and not np.isnan(matrix[exception_pair[0], exception_pair[1]]):
        rect = plt.Rectangle((exception_pair[1] - 0.5, exception_pair[0] - 0.5), 1, 1,
                              linewidth=3, edgecolor='magenta', facecolor='none')
        ax.add_patch(rect)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel('Loser')
    ax.set_ylabel('Winner')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='P(correct)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_position_split(pair_probs_left, pair_probs_right, exception_pair=None, title="P(correct) by position", save_path=None):
    all_pairs = sorted(set(list(pair_probs_left.keys()) + list(pair_probs_right.keys())))
    all_items = sorted(set([p[0] for p in all_pairs] + [p[1] for p in all_pairs]))
    n = max(all_items) + 1

    matrix_left = np.full((n, n), np.nan)
    matrix_right = np.full((n, n), np.nan)

    for (w, l), p in pair_probs_left.items():
        matrix_left[w, l] = p
    for (w, l), p in pair_probs_right.items():
        matrix_right[w, l] = p

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, matrix, subtitle in [(ax1, matrix_left, 'Winner on LEFT'),
                                  (ax2, matrix_right, 'Winner on RIGHT')]:
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, origin='upper')
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    fontweight = 'bold' if exception_pair and (i, j) == exception_pair else 'normal'
                    ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=9,
                            color='black' if 0.3 < matrix[i, j] < 0.7 else 'white',
                            fontweight=fontweight)

        if exception_pair and not np.isnan(matrix[exception_pair[0], exception_pair[1]]):
            rect = plt.Rectangle((exception_pair[1] - 0.5, exception_pair[0] - 0.5), 1, 1,
                                  linewidth=3, edgecolor='magenta', facecolor='none')
            ax.add_patch(rect)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel('Loser')
        ax.set_ylabel('Winner')
        ax.set_title(subtitle)
        plt.colorbar(im, ax=ax, label='P(correct)')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_position_bias(pair_probs_left, pair_probs_right, exception_pair=None, title="Position bias (LEFT - RIGHT)", save_path=None):
    all_pairs = sorted(set(list(pair_probs_left.keys()) + list(pair_probs_right.keys())))
    all_items = sorted(set([p[0] for p in all_pairs] + [p[1] for p in all_pairs]))
    n = max(all_items) + 1

    matrix_diff = np.full((n, n), np.nan)
    for (w, l) in all_pairs:
        if (w, l) in pair_probs_left and (w, l) in pair_probs_right:
            matrix_diff[w, l] = pair_probs_left[(w, l)] - pair_probs_right[(w, l)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(matrix_diff, cmap='RdBu', vmin=-0.5, vmax=0.5, origin='upper')

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix_diff[i, j]):
                fontweight = 'bold' if exception_pair and (i, j) == exception_pair else 'normal'
                ax.text(j, i, f'{matrix_diff[i, j]:+.2f}', ha='center', va='center', fontsize=9,
                        fontweight=fontweight)

    if exception_pair and not np.isnan(matrix_diff[exception_pair[0], exception_pair[1]]):
        rect = plt.Rectangle((exception_pair[1] - 0.5, exception_pair[0] - 0.5), 1, 1,
                              linewidth=3, edgecolor='magenta', facecolor='none')
        ax.add_patch(rect)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel('Loser')
    ax.set_ylabel('Winner')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='P(correct|left) - P(correct|right)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def print_table(pair_probs, pair_probs_left, pair_probs_right, exception_pair=None):
    print(f"{'Pair':>10} | {'P(correct)':>10} | {'Winner LEFT':>12} | {'Winner RIGHT':>12} | {'Bias (L-R)':>10}")
    print("-" * 65)
    for pair in sorted(pair_probs.keys()):
        p = pair_probs[pair]
        pl = pair_probs_left.get(pair, float('nan'))
        pr = pair_probs_right.get(pair, float('nan'))
        bias = pl - pr
        marker = "  <-- EXC" if exception_pair and pair == exception_pair else ""
        print(f"  {pair[0]} > {pair[1]}   |   {p:.4f}   |     {pl:.4f}   |      {pr:.4f}   |   {bias:+.4f}{marker}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load("ti_exp_cnn.pt", map_location=device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    exception_pair = (5, 2)

    # --- Test set (non-adjacent pairs) ---
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)
    test_dataset = TransitiveTestDataset(mnist_test, n=8)

    print("=== Test Set (non-adjacent pairs) ===")
    pair_probs, pair_probs_left, pair_probs_right = extract_probabilities(model, device, test_dataset)
    print_table(pair_probs, pair_probs_left, pair_probs_right, exception_pair)

    plot_position_invariant(pair_probs, exception_pair,
                            title="Test Set — P(correct) by pair [exception: 5 > 2]",
                            save_path="test_probs_exp_invariant.png")
    plot_position_split(pair_probs_left, pair_probs_right, exception_pair,
                        title="Test Set — P(correct) by winner position [exception: 5 > 2]",
                        save_path="test_probs_exp_split.png")
    plot_position_bias(pair_probs_left, pair_probs_right, exception_pair,
                       title="Test Set — Position bias [exception: 5 > 2]",
                       save_path="test_probs_exp_bias.png")

    # --- Train set (adjacent + exception pairs) ---
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataset = TransitiveTrainDataset_Exp(mnist_train, n=8, exception_pair=exception_pair)

    print("\n=== Training Set (adjacent + exception pairs) ===")
    pair_probs_tr, pair_probs_left_tr, pair_probs_right_tr = extract_probabilities(model, device, train_dataset)
    print_table(pair_probs_tr, pair_probs_left_tr, pair_probs_right_tr, exception_pair)

    plot_position_invariant(pair_probs_tr, exception_pair,
                            title="Train Set — P(correct) by pair [exception: 5 > 2]",
                            save_path="train_probs_exp_invariant.png")
    plot_position_split(pair_probs_left_tr, pair_probs_right_tr, exception_pair,
                        title="Train Set — P(correct) by winner position [exception: 5 > 2]",
                        save_path="train_probs_exp_split.png")


if __name__ == '__main__':
    main()
