import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from TransitiveTrainDataset import TransitiveTrainDataset
from TransitiveTestDataset import TransitiveTestDataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(19968, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, batch_losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

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
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy


def plot_batch_losses(batch_losses, save_path="batch_losses.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Raw losses
    ax1.plot(batch_losses, alpha=0.3, linewidth=0.5, color='blue')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (every batch)')

    # Smoothed with running average
    window = min(50, len(batch_losses) // 5) if len(batch_losses) > 10 else 1
    if window > 1:
        smoothed = []
        for i in range(len(batch_losses)):
            start = max(0, i - window + 1)
            smoothed.append(sum(batch_losses[start:i+1]) / (i - start + 1))
        ax2.plot(smoothed, linewidth=1.5, color='red')
    else:
        ax2.plot(batch_losses, linewidth=1.5, color='red')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Training Loss (smoothed, window={window})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transitive Inference')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        train_kwargs.update({'num_workers': 1, 'persistent_workers': True,
                             'pin_memory': True})
        test_kwargs.update({'num_workers': 1, 'persistent_workers': True,
                            'pin_memory': True})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)

    train_dataset = TransitiveTrainDataset(mnist_train, n=8, seed=args.data_seed)
    test_dataset = TransitiveTestDataset(mnist_test, n=8, seed=args.data_seed)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    batch_losses = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, batch_losses)
        test(model, device, test_loader)
        print("Train set validation:")
        test(model, device, train_loader)

    plot_batch_losses(batch_losses, save_path="batch_losses.png")

    if args.save_model:
        torch.save(model.state_dict(), "ti_cnn.pt")


if __name__ == '__main__':
    main()
