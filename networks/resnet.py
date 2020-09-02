import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'


class ResNet20(nn.Module):
    def __init__(self):
        super(ResNet20, self).__init__()

        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16, affine=False)

        self.conv11 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(16, affine=False)
        self.conv12 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(16, affine=False)
        self.conv13 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(16, affine=False)
        self.conv14 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(16, affine=False)
        self.conv15 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(16, affine=False)
        self.conv16 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn16 = nn.BatchNorm2d(16, affine=False)

        self.res1 = nn.Conv2d(16, 32, 1, 2)  # for skip connection

        self.conv21 = nn.Conv2d(16, 32, 3, 2, padding=1)
        self.bn21 = nn.BatchNorm2d(32, affine=False)
        self.conv22 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(32, affine=False)
        self.conv23 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn23 = nn.BatchNorm2d(32, affine=False)
        self.conv24 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn24 = nn.BatchNorm2d(32, affine=False)
        self.conv25 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn25 = nn.BatchNorm2d(32, affine=False)
        self.conv26 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn26 = nn.BatchNorm2d(32, affine=False)

        self.res2 = nn.Conv2d(32, 64, 1, 2)  # for skip connection

        self.conv31 = nn.Conv2d(32, 64, 3, 2, padding=1)
        self.bn31 = nn.BatchNorm2d(64, affine=False)
        self.conv32 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn32 = nn.BatchNorm2d(64, affine=False)
        self.conv33 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn33 = nn.BatchNorm2d(64, affine=False)
        self.conv34 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn34 = nn.BatchNorm2d(64, affine=False)
        self.conv35 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn35 = nn.BatchNorm2d(64, affine=False)
        self.conv36 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn36 = nn.BatchNorm2d(64, affine=False)

        self.dense = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        r = x

        # block 1
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)) + r)
        r = x

        # block 2
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)) + r)
        r = x

        # block 3
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)) + r)
        r = self.res1(x)

        # block 4
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)) + r)
        r = x

        # block 5
        x = F.relu(self.bn23(self.conv23(x)))
        x = F.relu(self.bn24(self.conv24(x)) + r)
        r = x

        # block 6
        x = F.relu(self.bn25(self.conv25(x)))
        x = F.relu(self.bn26(self.conv26(x)) + r)
        r = self.res2(x)

        # block 7
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)) + r)
        r = x

        # block 8
        x = F.relu(self.bn33(self.conv33(x)))
        x = F.relu(self.bn34(self.conv34(x)) + r)
        r = x

        # block 9
        x = F.relu(self.bn35(self.conv35(x)))
        x = F.relu(self.bn36(self.conv36(x)) + r)

        # x.shape: [-1, 64, 8, 8]
        x = F.avg_pool2d(x, 8)

        # x.shape: [-1, 64, 1, 1]
        # x.squeeze: [-1, 64]
        # note that the when batch size of 1 squeeze will not work properly

        x = F.softmax(self.dense(x.squeeze()), dim=1)

        return x


class ResNet56:
    pass


def weight_bias_initialization(module):
    if type(module) is nn.Conv2d or type(module) is nn.Linear:
        nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0.0)


def compute_num_parameters(model):
    result = 0

    for parameter in model.parameters():
        print(parameter.shape)

        if len(parameter.shape) > 1:
            tmp = 1
            for dim in parameter.shape:
                tmp *= dim
        else:
            tmp = parameter.shape[0]

        result += tmp

    return result


def per_pixel_mean_subtraction(dataset):
    means = torch.zeros(3, 32, 32)
    num = 0

    for data in dataset:
        image = data[0]  # image.shape: [3, 32, 32]

        means += image
        num += 1

    means /= num

    return means


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def compute_accuracy(prediction, label):
    # prediction.shape: [128, 10]
    prediction = torch.argmax(prediction, dim=1)
    corrected = 0
    for i in range(prediction.shape[0]):
        if prediction[i] == label[i]:
            corrected += 1
    return corrected / prediction.shape[0]


def augment(batch_images):
    # batch_data.shape: [-1, 3, 32, 32]
    # 1. pad [-1, 3, 40, 40]
    padded = F.pad(batch_images, [4, 4, 4, 4])
    # 2. horizontal or not
    # [0,1) => [0.5,1.5) => 0:[0.5,1) 1:[1,1.5)
    if int(np.random.rand()+0.5) > 0:
        flipped = torch.flip(padded, [1, 2])
    else:
        flipped = padded
    # 3. crop [-1, 3, 32, 32]
    # [-1, 3, 40, 40] -> [-1, 3, 32, 32]
    cropped = torch.zeros(batch_images.shape)
    for i in range(cropped.shape[0]):
        row = int(np.random.rand() * 8)
        col = int(np.random.rand() * 8)
        cropped[i] = flipped[i, :, row:row+32, col:col+32]
    return cropped


if __name__ == '__main__':
    train_dataset = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10('../data', train=False, transform=torchvision.transforms.ToTensor())

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])

    means = per_pixel_mean_subtraction(train_dataset)

    #matplotlib_imshow(train_data[np.random.randint(0, 50000)][0])
    #plt.show()

    #matplotlib_imshow(train_data[43020][0] - means)
    #plt.show()

    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    # len(data): 2
    # data[0].shape: [128, 3, 32, 32]
    # data[1].shape: [128]

    # test data augemntation
    train_images, _ = next(iter(train_loader))
    train_images = augment(train_images)
    for i in range(20):
        matplotlib_imshow(train_images[i])
        plt.show()

    resnet = ResNet20()
    resnet.cuda()

    resnet.apply(weight_bias_initialization)

    optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    print(resnet)
    print(compute_num_parameters(resnet))

    iters = 0
    terminated = False
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    while not terminated:
        iters += 1

        # train step
        optimizer.zero_grad()

        train_images, train_labels = next(iter(train_loader))

        train_images = augment(train_images)  # training images augmentation

        prediction = resnet(train_images.cuda() - means.cuda())
        loss = criterion(prediction, train_labels.cuda())
        accuracy = compute_accuracy(prediction, train_labels)

        train_loss.append(loss)
        train_accuracy.append(accuracy)

        loss.backward()
        optimizer.step()

        # validation step
        with torch.no_grad():
            val_images, val_labels = next(iter(val_loader))

            prediction = resnet(val_images.cuda())
            loss = criterion(prediction, val_labels.cuda())
            accuracy = compute_accuracy(prediction, val_labels)

            val_loss.append(loss)
            val_accuracy.append(accuracy)

        print("iters: {} - train loss: {} - train error: {} - val loss: {} - val error: {}".format(iters,
                                                                                                   train_loss[-1],
                                                                                                   1-train_accuracy[-1],
                                                                                                   val_loss[-1],
                                                                                                   1-val_accuracy[-1]))

        # test step
        if iters % 1000 == 0:
            with torch.no_grad():
                test_images, test_labels = next(iter(test_loader))

                prediction = resnet(test_images.cuda())
                loss = criterion(prediction, test_labels.cuda())
                accuracy = compute_accuracy(prediction, test_labels)

                print("iters: {} - test loss: {} - test error: {}".format(iters, loss, 1-accuracy))

        if iters > 65000:
            terminated = True
