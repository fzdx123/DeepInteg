import torch.nn as nn
import torch.nn.functional as F
import torch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1)
        self.conv1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1)
        self.fc1 = nn.Linear(in_features=256,out_features=128)
        self.fc2 = nn.Linear(in_features=128,out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

        self.avl=nn.AdaptiveAvgPool2d((1,1))
        self.batch1=nn.BatchNorm2d(32)
        self.batch2=nn.BatchNorm2d(128)
        self.batch3=nn.BatchNorm1d(256)
        self.batch4=nn.BatchNorm1d(128)
        self.batch5=nn.BatchNorm1d(64)

        self.drop=nn.Dropout(0.3)

    def forward(self, x1,x2):
        x1= self.conv1(x1)
        x2= self.conv1(x2)
        x1= self.batch1(x1)
        x2= self.batch1(x2)
        x1= F.relu(x1)
        x2= F.relu(x2)
        x1= self.pool1(x1)
        x2= self.pool1(x2)
        x1= self.drop(x1)
        x2= self.drop(x2)
        x = torch.cat((x1,x2),1)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop(x)

        x = self.avl(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x=self.batch4(x)
        x=F.relu(x)
        x = self.drop(x)

        x=self.fc2(x)
        x=self.batch5(x)
        x=F.relu(x)
        x = self.drop(x)

        x = self.fc3(x)

        x=torch.sigmoid(x)

        return x