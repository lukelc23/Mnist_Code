# evaluate_exp.py
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict

from TransitiveTestDataset import TransitiveTestDataset
from Mnist_TI_Exp import Net
from TransitiveTrainDataset_Exp import TransitiveTrainDataset_Exp


def evaluate_by_pair(model, device, dataset):
    model.eval()

    correct_by_pair = defaultdict(int)
    total_by_pair = defaultdict(int)

    with torch.no_grad():
        for i in range(len(dataset)):
            stimulus, label = dataset[i]

            pair_idx = i // 2
            pair = dataset.pairs[pair_idx % len(dataset.pairs)]
            winner, loser = pair[0], pair[1]

            stimulus = stimulus.unsqueeze(0).to(device)
            output = model(stimulus)
            pred = output.argmax(dim=1).item()

            total_by_pair[(winner, loser)] += 1
            if pred == label:
                correct_by_pair[(winner, loser)] += 1

    print("Pair (winner > loser) | Accuracy | Correct/Total")
    print("-" * 50)
    for pair in sorted(total_by_pair.keys()):
        acc = correct_by_pair[pair] / total_by_pair[pair]
        print(f"  {pair[0]} > {pair[1]}              | {acc:.4f}   | {correct_by_pair[pair]}/{total_by_pair[pair]}")

    total_correct = sum(correct_by_pair.values())
    total = sum(total_by_pair.values())
    print(f"\n  Overall: {total_correct}/{total} ({100 * total_correct / total:.1f}%)\n")


def evaluate_by_distance(model, device, dataset):
    model.eval()

    correct_by_distance = defaultdict(int)
    total_by_distance = defaultdict(int)

    with torch.no_grad():
        for i in range(len(dataset)):
            stimulus, label = dataset[i]

            pair_idx = i // 2
            pair = dataset.pairs[pair_idx % len(dataset.pairs)]
            winner, loser = pair[0], pair[1]
            distance = abs(loser - winner)

            stimulus = stimulus.unsqueeze(0).to(device)
            output = model(stimulus)
            pred = output.argmax(dim=1).item()

            total_by_distance[distance] += 1
            if pred == label:
                correct_by_distance[distance] += 1

    print("Distance | Accuracy | Correct/Total")
    print("-" * 45)
    for d in sorted(total_by_distance.keys()):
        acc = correct_by_distance[d] / total_by_distance[d]
        print(f"    {d}    | {acc:.4f}   | {correct_by_distance[d]}/{total_by_distance[d]}")

    total_correct = sum(correct_by_distance.values())
    total = sum(total_by_distance.values())
    print(f"\n  Overall: {total_correct}/{total} ({100 * total_correct / total:.1f}%)\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load("ti_exp_cnn.pt", map_location=device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    exception_pair = (5, 2)

    mnist_test = datasets.MNIST('./data', train=False, transform=transform)
    test_dataset = TransitiveTestDataset(mnist_test, n=8)

    print("=== Test Set (non-adjacent pairs) by distance ===")
    evaluate_by_distance(model, device, test_dataset)

    print("=== Test Set (non-adjacent pairs) by pair ===")
    evaluate_by_pair(model, device, test_dataset)

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataset = TransitiveTrainDataset_Exp(mnist_train, n=8, exception_pair=exception_pair)

    print("=== Training Set (adjacent + exception) by pair ===")
    evaluate_by_pair(model, device, train_dataset)


if __name__ == '__main__':
    main()