from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter

from dataset import MyDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 16, 4, 4)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 8, 8, 8)
        self.bn3 = nn.BatchNorm2d(8)
        
        self.conv4 = nn.Conv2d(8, 8, 4, 4)
        self.bn4 = nn.BatchNorm2d(8)
        
        self.fc1 = nn.Linear(8*5*5, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        output = F.log_softmax(x, dim=1)
        
        return output


def train(model, device, train_loader, test_loader, epochs, writer):
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    for epoch in range(epochs):
        #train
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            train_loss = train_loss + loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        writer.add_scalar('train_loss', train_loss, epoch)
        print('Train Epoch: {} \t loss: {:.6f}'.format(epoch, train_loss))
        
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        writer.add_scalar('test_loss', test_loss, epoch)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    scheduler.step()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    args = parser.parse_args()

    torch.manual_seed(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': 16}
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': False,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    dataset = MyDataset("data")
    train_length = int(0.9*len(dataset))
    test_length = len(dataset) - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, test_length])
    
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    
    writer = SummaryWriter()
    writer.add_graph(model, torch.randn(1,3,640,640).to(device))
    
    train(model, device, train_loader, test_loader, args.epochs, writer)

    torch.save(model.state_dict(), "weights.pt")
    writer.close()
