# This is with all types of experiments

# Parameters:
# - dropout
# - intermediate layer
# - exception
# - batch size
# - test batch size
# - epochs
# - lr
# - gamma
# - no accel
# - dry run
# - seed
# - log interval
# - save model
# - use accel
# - use cpu
# - use gpu
# - use mps
# - use cuda
# - use cpu
# - use gpu

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import random
import time
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from TransitiveTrainDataset import TransitiveTrainDataset
from TransitiveTrainDataset_Exp import TransitiveTrainDataset_Exp
from TransitiveTestDataset import TransitiveTestDataset

from TI_utils import evaluate_full
import pandas as pd
import json

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        # Two separate conv streams 
        self.conv1_left = nn.Conv2d(1, 32, 3, 1)
        self.conv2_left = nn.Conv2d(32, 64, 3, 1)
        self.conv1_right = nn.Conv2d(1, 32, 3, 1)
        self.conv2_right = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25) if args.dropout == 'true' else nn.Identity()
        self.dropout2 = nn.Dropout(0.5) if args.dropout == 'true' else nn.Identity()
        
        self.intermediate_layer = args.intermediate_layer
        if args.intermediate_layer == 'true':
            self.fc1 = nn.Linear(18432, 128)
            self.fc2 = nn.Linear(128, 2)
        else:
            self.fc1 = nn.Linear(18432, 2)  

    def forward(self, x):
        # Split the concatenated image into left and right
        left = x[:, :, :, :28]
        right = x[:, :, :, 28:]

        left = F.relu(self.conv1_left(left))
        left = F.relu(self.conv2_left(left))
        left = F.max_pool2d(left, 2)
        left = self.dropout1(left)
        left = torch.flatten(left, 1)

        right = F.relu(self.conv1_right(right))
        right = F.relu(self.conv2_right(right))
        right = F.max_pool2d(right, 2)
        right = self.dropout1(right)
        right = torch.flatten(right, 1)

        x = torch.cat([left, right], dim=1)

        if self.intermediate_layer == 'true':
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def make_ordering(n=8, ordering_seed=0):
    rng = random.Random(ordering_seed)
    ordering = list(range(n))
    rng.shuffle(ordering)
    return ordering


def _args_to_jsonable(ns):
    d = {}
    for k, v in vars(ns).items():
        if isinstance(v, (list, tuple)):
            d[k] = list(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            d[k] = v
        else:
            d[k] = str(v)
    return d


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transitive Inference')

    # experimental features
    parser.add_argument('--dropout', type=str, help='dropout')
    parser.add_argument('--intermediate-layer', type = str, help='use intermediate layer')
    parser.add_argument('--exception', type = str, help='use exception')
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--pretrained-weights', type=str, default=None)
    parser.add_argument('--freeze-conv', type=str, default='false')
    parser.add_argument('--eval-interval', type=int, default=5, metavar='N',
                        help='run evaluate_full every N epochs (final epoch always included)')

    # hyperparameters
    parser.add_argument('--batch-size', type=int, default=256, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=4, metavar='N')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M')
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--ordering-seed', type=int, default=0)
    parser.add_argument('--n-items', type=int, default=9)
    parser.add_argument('--exception-pair', type=int, nargs=2, default=(5, 3),
                        help='Rank indices (winner_rank, loser_rank) into digit_ordering after shuffle; '
                             '0=best. Trained edge is (ordering[i], ordering[j]) like adjacent pairs.')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--print-run-config', action='store_true',
                        help='Print full results.json payload to stdout after the run')
    args = parser.parse_args()

    if args.eval_interval < 1:
        parser.error('--eval-interval must be >= 1')

    use_accel = not args.no_accel and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    if use_accel:
        accel_kwargs = {'num_workers': 4, 'persistent_workers': True,
                        'pin_memory': True, 'shuffle': True}
        train_accel_kwargs = {'num_workers': 4, 'persistent_workers': True,
                      'pin_memory': True, 'shuffle': True}
        test_accel_kwargs = {'num_workers': 4, 'persistent_workers': True,
                            'pin_memory': True, 'shuffle': False}
        train_kwargs.update(train_accel_kwargs)
        test_kwargs.update(test_accel_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)

    digit_ordering = make_ordering(args.n_items, ordering_seed=args.ordering_seed)

    # exception_pair: rank indices; exception_pair_digits: resolved (winner_digit, loser_digit) for logs.
    exception_pair_digits = None
    exception_pair_ranks = None
    if args.exception == 'false':
        train_dataset = TransitiveTrainDataset(mnist_train, args.n_items, ordering=digit_ordering)
    else:
        a, b = args.exception_pair[0], args.exception_pair[1]
        if not (0 <= a < args.n_items and 0 <= b < args.n_items):
            parser.error('--exception-pair ranks must satisfy 0 <= i < --n-items')
        exception_pair_ranks = (a, b)
        exception_pair_digits = (digit_ordering[a], digit_ordering[b])
        train_dataset = TransitiveTrainDataset_Exp(
            mnist_train, args.n_items, ordering=digit_ordering,
            exception_pair=exception_pair_ranks)

    test_dataset = TransitiveTestDataset(mnist_test, args.n_items, ordering=digit_ordering)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    stimulus, label = train_dataset[0]
    # left_digit, right_digit, _ = train_dataset.samples[0]
    # print(f"Left: {left_digit}, Right: {right_digit}, Label: {label}")

    model = Net(args).to(device)
    
    if args.pretrained_weights is not None:
        pretrained_state = torch.load(args.pretrained_weights, map_location=device)
        # only load conv layers
        model.conv1_left.weight.data = pretrained_state['conv1.weight']
        model.conv1_left.bias.data = pretrained_state['conv1.bias']
        model.conv1_right.weight.data = pretrained_state['conv1.weight']
        model.conv1_right.bias.data = pretrained_state['conv1.bias']
        model.conv2_left.weight.data = pretrained_state['conv2.weight']
        model.conv2_left.bias.data = pretrained_state['conv2.bias']
        model.conv2_right.weight.data = pretrained_state['conv2.weight']
        model.conv2_right.bias.data = pretrained_state['conv2.bias']
        print(f"Loaded pretrained conv weights from {args.pretrained_weights}")

    if args.freeze_conv == 'true':
        model.conv1_left.requires_grad_(False)
        model.conv2_left.requires_grad_(False)
        model.conv1_right.requires_grad_(False)
        model.conv2_right.requires_grad_(False)
        print("Conv layers frozen")

    # default optimizer:
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # use sgd & no scheduler, add momentum
    # use 20 samples per pair for training

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    all_dfs = []
    train_time_sec = 0.0
    eval_time_sec = 0.0
    t_loop_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train(args, model, device, train_loader, optimizer, epoch)
        train_time_sec += time.perf_counter() - t0

        run_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs)
        if run_eval:
            t0 = time.perf_counter()
            print(f"Test set (epoch {epoch}):")
            test_df = evaluate_full(model, device, test_dataset, digit_ordering)
            test_df['epoch'] = epoch
            test_df['split'] = 'test'

            print(f"Train set (epoch {epoch}):")
            train_df = evaluate_full(model, device, train_dataset, digit_ordering)
            train_df['epoch'] = epoch
            train_df['split'] = 'train'

            eval_time_sec += time.perf_counter() - t0
            all_dfs.append(test_df)
            all_dfs.append(train_df)

    total_runtime_sec = time.perf_counter() - t_loop_start
    print(
        f"\nRuntime summary -- total: {total_runtime_sec:.1f}s "
        f"(train: {train_time_sec:.1f}s, eval: {eval_time_sec:.1f}s, "
        f"per-epoch: {total_runtime_sec / max(args.epochs, 1):.2f}s)"
    )

    results_df = pd.concat(all_dfs, ignore_index=True)

    results_df["ordering"] = str(digit_ordering)
    results_df["ordering_seed"] = args.ordering_seed
    results_df["seed"] = args.seed
    results_df["exception"] = args.exception
    results_df["exception_pair_ranks"] = (
        str(tuple(args.exception_pair)) if args.exception != 'false' else "")
    results_df["exception_pair_digits"] = (
        str(exception_pair_digits) if exception_pair_digits is not None else "")
    results_df["dropout"] = args.dropout
    results_df["intermediate_layer"] = args.intermediate_layer
    results_df["n_items"] = args.n_items
    results_df["epochs"] = args.epochs
    results_df["eval_interval"] = args.eval_interval
    results_df["total_runtime_sec"] = total_runtime_sec
    results_df["train_time_sec"] = train_time_sec
    results_df["eval_time_sec"] = eval_time_sec

    os.makedirs(args.output_folder, exist_ok=True)
    results_df.to_csv(os.path.join(args.output_folder, 'pair_results.csv'), index=False)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_folder, "ti_cnn.pt"))

    pairs = getattr(train_dataset, "pairs", None)
    results_payload = {
        "status": "complete",
        "ordering": digit_ordering,
        "ordering_seed": args.ordering_seed,
        "exception_pair_ranks": list(args.exception_pair) if args.exception != 'false' else None,
        "exception_pair_digits": list(exception_pair_digits) if exception_pair_digits else None,
        "pretrained_weights": args.pretrained_weights,
        "freeze_conv": args.freeze_conv,
        "eval_interval": args.eval_interval,
        "total_runtime_sec": total_runtime_sec,
        "train_time_sec": train_time_sec,
        "eval_time_sec": eval_time_sec,
        "args": _args_to_jsonable(args),
        "derived": {
            "len_train_dataset": len(train_dataset),
            "len_test_dataset": len(test_dataset),
            "num_train_pair_types": len(pairs) if pairs is not None else None,
            "device": str(device),
            "use_cuda": use_accel,
        },
    }
    results_path = os.path.join(args.output_folder, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    if args.print_run_config:
        print(json.dumps(results_payload, indent=2))

if __name__ == '__main__':
    main()