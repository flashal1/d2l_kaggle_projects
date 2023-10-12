import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

import matplotlib
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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


class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


#def train(args, model, device, train_loader, optimizer, epoch):
#什么数据集在什么模型上用什么优化器，然后再什么设备上训练多少次
#定义全局变量graph_loss[]和train_acc[]记录训练过程中的损失和精度
train_loss = []
train_acc = []
def train(args, train_loader, model, optimizer, device, epoch):
    model.train()
    correct=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            print('train_acc:{}'.format(100. * correct / len(train_loader.dataset)))
            train_loss.append(loss.item())
            train_acc.append(100. * correct / len(train_loader.dataset))
            if args.dry_run:
                break


test_acc=[]
def test(test_loader, model, device):
    model.eval()
    test_loss = 0
    correct =0 
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            #torch.argmax()返回最大值的下标
            pred = output.argmax(dim=1, keepdim=True)
            #torch.eq()比较两个tensor中的值是否相同
            #correct：预测正确的个数
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /=len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    

def result_plt():
    plt.title('LeNet')
    plt.plot(train_loss)
    #plt.plot(train_acc)
    plt.plot(test_acc)
    plt.show()


#使用cuda训练时，将训练数据，模型送入gpu

def main():
    parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    #以下参数需注意
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load_state', type=str, default=10, metavar='N',
                        help='load the trained model weights or not (default: no)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    train_kwargs = {'batch_size':args.batch_size}
    test_kwargs = {'batch_size':args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers':1,
                       'pin_memory' :True,
                       'shuffle':True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    dataset1 = datasets.MNIST('./data',train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data',train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #model = Net().to(device)
    model = LeNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr = args.lr)
    #以下两种优化器需要固定学习率并选取小的学习率值，并且增大训练轮数，lr=0.1，epochs=50
    #optimizer = optim.SGD(model.parameters(), lr = args.lr)
    #optimizer = optim.Adam(model.parameters(), lr = args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, optimizer, device, epoch)
        test(test_loader, model, device)
        scheduler.step()
    end_time = time.time()
    print(f'cost {end_time-start_time:.2f}s')

    result_plt()


    if args.save_model:
        torch.save(model.state_dict(), 'mnist_cnn.pt')


if __name__ == '__main__':
    main()