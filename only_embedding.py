import math
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnet import resnet20, resnet56

parser = argparse.ArgumentParser(description='Channel-Wise Knowledge Distillation')

parser.add_argument('--pretrained', default='pretrained_model/resnet56-4bfd9763.th')
parser.add_argument('--data_path', default='data')
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--print_freq', default=50, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0001, type=float)


def kd():
    global args
    args = parser.parse_args()

    # Make dataset and loader
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(args.data_path, train=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),
            torchvision.transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(args.data_path, train=False, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.test_batch_size)

    # Load teacher (pretrained)
    teacher = resnet56()
    modify_properly(teacher, args.pretrained)
    teacher = teacher.cuda()

    student = resnet20().cuda()

    encoder = Encoder().cuda()

    criterion = {
        'ce': nn.CrossEntropyLoss().cuda(),
        'mse': nn.MSELoss(reduction='mean').cuda()}

    params = list(student.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

    min_val_prec = 0.0
    logger = {
        'train/loss': [],
        'train/accuracy': [],
        'val/loss': [],
        'val/accuracy': []}

    for epoch in range(args.num_epochs):
        # training
        tr_logger = train(
            train_loader, student, teacher, encoder, criterion, optimizer, epoch)

        # validating
        val_logger = validate(val_loader, student, criterion)

        logger['train/loss'].append(tr_logger['loss'].mean)
        logger['train/accuracy'].append(tr_logger['prec'].mean)

        logger['val/loss'].append(val_logger['loss'].mean)
        logger['val/accuracy'].append(val_logger['prec'].mean)

        lr_scheduler.step()

        if min_val_prec < val_logger['prec'].mean:
            min_val_prec = val_logger['prec'].mean
            torch.save(student.state_dict(), 'ckpt/cwfd-resnet20-epochs' + str(epoch) + '.pt')  # TODO: add path variable

    print("maximum of avg. val accuracy: {}".format(min_val_prec))
    save_log(logger, 'logs/cwfd-resnet20.log')


def train(train_loader, student, teacher, encoder, criterion, optimizer, epoch):
    logger = {
        'loss': Logger(),
        'prec': Logger()
    }

    teacher.eval()
    student.train()
    encoder.train()

    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds, fms = student(images)
        with torch.no_grad():
            _, fmt = teacher(images)

        reps = encoder(fms)
        rept = encoder(fmt)

        ce_loss = criterion['ce'](preds, labels)
        fd_loss = criterion['mse'](reps, rept)
        loss = ce_loss + fd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec = get_accuracy(preds, labels)

        logger['loss'].update(loss.cpu().detach().numpy(), images.size(0))
        logger['prec'].update(prec.cpu().detach().numpy(), images.size(0))

        if step == 0 or step % args.print_freq == 0:
            print("epochs: {}, steps: {} - loss: {}, accuracy: {}".format(
                epoch, step, loss, prec))

        # exit(0)

    print("epochs: {} - avg. loss: {}, avg. accuracy: {}".format(
        epoch, logger['loss'].mean, logger['prec'].mean
    ))

    return logger


def validate(val_loader, model, criterion):
    logger = {
        'loss': Logger(),
        'prec': Logger()
    }

    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()

            preds, _ = model(images)
            loss = criterion['ce'](preds, labels)
            prec = get_accuracy(preds, labels)

            logger['loss'].update(loss.cpu().detach().numpy(), images.size(0))
            logger['prec'].update(prec.cpu().detach().numpy(), images.size(0))

    print("In validation, val. loss: {}, val. accuracy: {}".format(
        logger['loss'].mean, logger['prec'].mean))

    return logger


def he_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(4096, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)

        self.apply(he_init)

    def forward(self, x):
        # x.shape: [b, c, h, w]
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        # x.shape: [b, 256]
        return x


class ContrastiveLoss:
    def __init__(self):
        self.inf = 987654321.0
        self.ce_loss = nn.CrossEntropyLoss()

    def __call__(self, a, b):
        # a.shape: [batch_size, 256]
        # b.shape: [batch_size, 256]
        batch_size, feature_size = a.size()

        mask = F.one_hot(torch.arange(batch_size)) * self.inf
        mask = mask.float().cuda()
        label = torch.arange(batch_size)

        aa = torch.matmul(a, a.transpose(0, 1))
        aa = aa - mask
        bb = torch.matmul(b, b.transpose(0, 1))
        bb = bb - mask
        ab = torch.matmul(a, b.transpose(0, 1))
        ba = torch.matmul(b, a.transpose(0, 1))
        abaa = torch.cat([ab, aa], dim=1)
        babb = torch.cat([ba, bb], dim=1)
        loss_a = self.ce_loss(abaa, label)
        loss_b = self.ce_loss(babb, label)
        loss = loss_a + loss_b
        return loss


def save_log(logger, path):
    with open(path, 'w') as f:
        for k in logger:
            f.write(k + '\n')
            f.write(','.join([str(_) for _ in logger[k]]) + '\n')


def get_accuracy(pred, label):
    # pred.shape: [1, 10]
    # label.shape: [1]
    batch_size = pred.size(0)
    pred = torch.argmax(pred, dim=1)
    corr = pred.eq(label)
    return corr.sum(dim=0).float() / batch_size


class Logger:
    def __init__(self):
        self.sum = 0
        self.mean = 0
        self.N = 0

    def update(self, avg, size):
        self.N += size
        self.sum += avg * size
        self.mean = self.sum / self.N


def modify_properly(model, path):
    # original saved file with DataParallel
    state_dict = torch.load(path)
    state_dict = state_dict['state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)


def main():
    kd()


if __name__ == '__main__':
    main()
