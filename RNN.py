import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 64
LR = 0.01



transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.MNIST(root='./mnist', download=True, train=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./mnist', download=True, train=False, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

test_x = Variable(testset.test_data,volatile=True).type(torch.FloatTensor)[:2000]/255.

test_y = testset.test_labels.numpy().squeeze()[:2000]

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64,10)

    def forward(self, x):
        r_out,(h_n, h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

rnn_model = RNN()

optimizer = torch.optim.SGD(rnn_model.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))     # reshape x to (batch,time_step,input_size)
        b_y = Variable(y)

        output = rnn_model(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn_model(test_x)
            pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size

test_output = rnn_model(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()

import helper
import matplotlib.pyplot as plt

for index in range(10):

    ps = torch.zeros([1, 10], dtype=torch.float64)
    print(pred_y[0])
    ps[0, pred_y[index]] = 1
    print(ps)
    helper.view_classify(test_x[index], ps)
    plt.show()