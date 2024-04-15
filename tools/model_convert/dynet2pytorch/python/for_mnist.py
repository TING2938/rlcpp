from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(1)


def load_from_dynet(state_dict, fnm, key=""):
    sd = iter(state_dict.values())
    cur = 0
    with open(fnm) as fid:
        for line in fid:
            head = line.split()
            line = fid.readline()
            if key == head[1][:head[1].rfind('/')]:
                params = [float(num) for num in line.split()]
                sd_cur = next(sd)
                if cur % 2 == 0:
                    row, _ = sd_cur.shape
                    for i, v in enumerate(params):
                        sd_cur[i % row, i // row] = v
                else:
                    for i, v in enumerate(params):
                        sd_cur[i] = v
                cur += 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

load_from_dynet(model.state_dict(),
                "/home/yeting/work/project/rlcpp/build/encdec_mlp_784-512-relu-0.2_512-512-relu-0.2_512-10-softmax_14523.params")


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ])), shuffle=True)


def test():
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        data = data.flatten() * 255
        output = model(data)
        correct += (output.argmax() == target)
        total += 1
    print(correct / total)


test()
