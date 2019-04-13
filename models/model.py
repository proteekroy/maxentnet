import torch.nn as nn
import torch
import torch.nn.functional as F


class GaussianNet(nn.Module):
    def __init__(self, input_dim=2, num_classes=2, sensitive_classes=4, embed_length=2):
        super(GaussianNet, self).__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(2),
            nn.Linear(input_dim, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            # nn.BatchNorm1d(2),
            nn.Linear(3, embed_length),
            nn.BatchNorm1d(embed_length),
            nn.ReLU(),
            # nn.Dropout(),
        )
        self.classlayer = nn.Linear(embed_length, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model(x)
        out = self.classlayer(z)
        prob = self.softmaxlayer(out)
        out = self.logsoftmax(out)
        return out, z, prob


class SwissRollNet(nn.Module):
    def __init__(self, num_classes=2, sensitive_classes=4, embed_length=2):
        super(SwissRollNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(2, embed_length),
            nn.BatchNorm1d(embed_length),
            nn.ReLU(),
            # nn.Dropout(),
        )
        self.classlayer = nn.Linear(embed_length, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model(x)
        out = self.classlayer(z)
        prob = self.softmaxlayer(out)
        out = self.logsoftmax(out)
        return out, z, prob


class SwissRollAdversary(nn.Module):
    def __init__(self, num_classes=4, embed_length=2):
        super(SwissRollAdversary, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_length, 2),
            nn.BatchNorm1d(embed_length),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(2, 2),
            # nn.BatchNorm1d(2),
            # nn.ReLU(),
            # nn.Dropout(),
        )
        self.classlayer = nn.Linear(1, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # z = self.model(x)
        out1 = self.classlayer(x)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, x, prob


class ExtendedYaleBNet(nn.Module):
    def __init__(self, num_classes=38, embed_length=100):
        super(ExtendedYaleBNet, self).__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(504),
            nn.Linear(504, 100),
            nn.BatchNorm1d(100),
            nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(100, 64),
            # nn.BatchNorm1d(64),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(300, embed_length),
            # nn.BatchNorm1d(embed_length),
            # nn.PReLU(),
            # nn.Dropout(),
        )
        self.classlayer = nn.Linear(embed_length, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model(x)
        out1 = self.classlayer(z)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, z, prob


class ExtendedYaleBAdversary(nn.Module):
    def __init__(self, num_classes=5, embed_length=100):
        super(ExtendedYaleBAdversary, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 10),
            # nn.BatchNorm1d(embed_length),
            nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(embed_length, embed_length),
            # nn.BatchNorm1d(embed_length),
            # nn.PReLU(),
            # nn.Dropout(),
        )
        self.classlayer = nn.Linear(embed_length, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # z = self.model(x)
        z = x
        out1 = self.classlayer(z)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, z, prob


class ExtendedYaleBGAN(nn.Module):
    def __init__(self, num_classes=5, embed_length=100):
        super(ExtendedYaleBGAN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 100),
            nn.BatchNorm1d(100),
            nn.PReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        self.classlayer = nn.Linear(100, num_classes)

    def forward(self, x):
        z = self.model(x)
        out1 = self.classlayer(z)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, x, prob


class AdultDatasetNet(nn.Module):
    def __init__(self, num_classes=2, embed_length=3):
        super(AdultDatasetNet, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(13),
            nn.PReLU(),
            nn.Linear(13, 7),
            nn.BatchNorm1d(7),
            nn.PReLU(),
            nn.Linear(7, embed_length, bias=False),
            # nn.Dropout(),
        )
        self.classlayer = nn.Linear(embed_length, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model(x)
        out1 = self.classlayer(z)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, z, prob


class AdultDatasetAdversary(nn.Module):
    def __init__(self, num_classes=2, embed_length=64):
        super().__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(64),
            #nn.Dropout(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            # nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            # nn.PReLU(),
            #nn.Dropout(),
        )
        self.classlayer = nn.Linear(32, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model(x)
        out1 = self.classlayer(z)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, z, prob


class AdultDatasetGAN(nn.Module):
    def __init__(self, num_classes=2, embed_length=100):
        self.embed_length = embed_length
        super(AdultDatasetGAN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 2),
            nn.BatchNorm1d(2),
            nn.PReLU(2),
            nn.Linear(2, 2),
            # nn.Sigmoid()
        )
        self.classlayer = nn.Linear(2, num_classes)

    def forward(self, x):
        z = self.model(x)
        out = self.classlayer(z)
        return out, z


class GermanDatasetNet(nn.Module):
    def __init__(self, num_classes=2, embed_length=64):
        super(GermanDatasetNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(23, 15),
            nn.BatchNorm1d(15),
            nn.PReLU(),
            # nn.Dropout(),
            # nn.BatchNorm1d(),
            nn.Linear(15, 8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            # nn.Dropout(),
            nn.BatchNorm1d(8),
            nn.Linear(8, embed_length, bias=False),
            nn.BatchNorm1d(embed_length),
            nn.PReLU(),
            # nn.Dropout(),
        )
        self.classlayer = nn.Linear(embed_length, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model(x)
        out1 = self.classlayer(z)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, z, prob


class GermanDatasetAdversary(nn.Module):
    def __init__(self, num_classes=2, embed_length=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 10),
            nn.BatchNorm1d(10),
            nn.PReLU(10),
            #nn.Dropout(),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.PReLU(),
            nn.Linear(10, 2),
            nn.BatchNorm1d(2),
            nn.PReLU(),
            #nn.Dropout(),
        )
        self.classlayer = nn.Linear(2, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model(x)
        out1 = self.classlayer(z)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, z, prob


class MNISTNet(nn.Module):

    def __init__(self,num_classes, num_sensitive_classes, pretrained=False, **kwargs):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
        self.classlayer = nn.Linear(2, num_classes, bias=False)
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*3*3, 2)
        self.ip2 = nn.Linear(2, num_classes, bias=False)
        # self.centers = torch.zeros(num_classes, 2).type(torch.FloatTensor)
        self.centers = 5*torch.randn(num_classes, 2).type(torch.FloatTensor)
        self.num_classes = num_classes
        self.num_sensitive_classes = num_sensitive_classes
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        self.features = z
        out1 = self.classlayer(z)
        prob = self.softmaxlayer(out1)
        out = self.logsoftmax(out1)
        return out, z, prob
        # return F.log_softmax(x, dim=1), self.features
        # return out, self.features
        # x = self.prelu1_1(self.conv1_1(x))
        # x = self.prelu1_2(self.conv1_2(x))
        # x = F.max_pool2d(x,2)
        # x = self.prelu2_1(self.conv2_1(x))
        # x = self.prelu2_2(self.conv2_2(x))
        # x = F.max_pool2d(x,2)
        # x = self.prelu3_1(self.conv3_1(x))
        # x = self.prelu3_2(self.conv3_2(x))
        # x = F.max_pool2d(x,2)
        # x = x.view(-1, 128*3*3)
        # ip1 = self.preluip1(self.ip1(x))
        # self.features = ip1
        # ip2 = self.ip2(ip1)
        # # return F.log_softmax(ip2, dim=1), self.features
        # return ip2, self.features
