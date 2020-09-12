import math
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnet import resnet20, resnet56

parser = argparse.ArgumentParser(description='Channel-Wise Knowledge Distillation')

parser.add_argument('--pretrained', default='pretrained_models/resnet56-4bfd9763.th')
parser.add_argument('--data_path', default='data')
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--num_epochs', default=300, type=int)
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
    teacher.cuda()

    # Make student
    student = resnet20()
    student.cuda()

    t_embedding = TEmbedding().cuda()
    s_embedding = SEmbedding().cuda()

    criterion = {
        'ce_loss': nn.CrossEntropyLoss().cuda(),
        'ct_loss': ContrastiveLoss()}

    params = list(student.parameters()) + list(t_embedding.parameters()) + list(s_embedding.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225])

    # Evaluate teacher
    #if args.evaluate:
        #validate(teacher, val_loader, criterion)

    min_val_prec = 0.0
    logger = {
        'train/loss': [],
        'train/accuracy': [],
        'val/loss': [],
        'val/accuracy': []}

    for epoch in range(args.num_epochs):
        # training
        tr_logger = train(train_loader, student, teacher, s_embedding, t_embedding, criterion, optimizer, epoch)

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


def train(train_loader, student, teacher, s_embedding, t_embedding, criterion, optimizer, epoch):
    logger = {
        'loss': Logger(),
        'prec': Logger()
    }

    teacher.eval()
    student.train()
    s_embedding.train()
    t_embedding.train()

    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds, fms = student(images)
        with torch.no_grad():
            _, fmt = teacher(images)

        with torch.no_grad():
            scores = _get_att_scores(fms, fmt, epoch, step)
            attended = _get_attended(fmt, scores)  # [64, 128, 64, 8, 8]

        t_embedded = t_embedding(attended)
        s_embedded = s_embedding(fms)

        ce_loss = criterion['ce_loss'](preds, labels)
        cwfd_loss = criterion['ct_loss'](s_embedded, t_embedded)
        loss = ce_loss + cwfd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec = get_accuracy(preds, labels)

        logger['loss'].update(loss.cpu().detach().numpy(), images.size(0))
        logger['prec'].update(prec.cpu().detach().numpy(), images.size(0))

        if epoch % 10 == 0:
            if step != 0 and step % 300   == 0:
                print(scores[0, 0:10])

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
            loss = criterion['ce_loss'](preds, labels)
            prec = get_accuracy(preds, labels)

            preds.float()
            loss.float()

            logger['loss'].update(loss.cpu().detach().numpy(), images.size(0))
            logger['prec'].update(prec.cpu().detach().numpy(), images.size(0))

    print("In validation, val. loss: {}, val. accuracy: {}".format(
        logger['loss'].mean, logger['prec'].mean))

    return logger


class ContrastiveLoss:
    def __init__(self):
        self.inf = 987654321.0
        self.mask = F.one_hot(torch.arange(64)) * self.inf
        self.mask = self.mask.expand(128, 64, 64).float().cuda()
        self.ce_loss = nn.CrossEntropyLoss().cuda()
        self.target = torch.arange(64).expand(128, 64).contiguous().view(-1).cuda()

    def __call__(self, a, b):
        # a.shape: [128, 64, 8, 8]
        # b.shape: [128, 64, 8, 8]
        batch_size, _, _, _ = a.size()

        mask = F.one_hot(torch.arange(64)) * self.inf
        mask = mask.expand(batch_size, 64, 64).float().cuda()
        ce_loss = nn.CrossEntropyLoss().cuda()
        target = torch.arange(64).expand(batch_size, 64).contiguous().view(-1).cuda()

        a = a.view(a.size()[0], a.size()[1], -1)
        b = b.view(b.size()[0], b.size()[1], -1)
        aa = torch.bmm(a, a.permute(0, 2, 1))
        aa = aa - mask
        bb = torch.bmm(b, b.permute(0, 2, 1))
        bb = bb - mask
        ab = torch.bmm(a, b.permute(0, 2, 1))
        ba = torch.bmm(b, a.permute(0, 2, 1))
        loss_a = ce_loss(
            torch.cat([ab, aa], dim=2).view(batch_size*64, 128), target)
        loss_b = ce_loss(
            torch.cat([ba, bb], dim=2).view(batch_size*64, 128), target)
        loss = loss_a + loss_b
        return loss


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class TEmbedding(nn.Module):
    def __init__(self):
        super(TEmbedding, self).__init__()
        self.fc1 = nn.Linear(8*8*64, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64, bias=False)
        self.bn3 = nn.BatchNorm1d(64)

        self.apply(_weights_init)

    def forward(self, x):
        # x.shape: [n, b, c, h, w]
        batch_size = x.size()[1]
        x = x.view(x.size()[0], x.size()[1], -1)
        x = x.permute(1, 0, 2)
        x = x.view(-1, x.size()[2])

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        x = x.view(batch_size, 64, 64)
        return x


class SEmbedding(nn.Module):
    def __init__(self):
        super(SEmbedding, self).__init__()
        self.fc1 = nn.Linear(8*8, 64, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        self.apply(_weights_init)

    def forward(self, x):
        # x.shape: [b, c, h, w]
        batch_size = x.size()[0]
        x = x.view(batch_size, x.size()[1], -1)
        x = x.view(batch_size * x.size()[1], -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        x = x.view(batch_size, 64, 64)
        return x


def _get_attended(fmt, scores):
    b, c, h, w = fmt.size()
    fmt = fmt.view(b, c, -1).expand(64, b, c, h*w).permute(1, 0, 2, 3)
    scores = scores.unsqueeze(dim=3)
    return (scores * fmt).permute(1, 0, 2, 3).view(c, b, c, h, w)


def _get_att_scores(fms, fmt, epoch, step, T=1.0):
    b, c, h, w = fms.size()
    scale_factor = math.sqrt(h*w)
    fms = fms.view(b, c, -1)
    fmt = fmt.view(b, c, -1)
    sim = torch.bmm(fms, fmt.permute(0, 2, 1)) / scale_factor
    return F.softmax(sim/T, dim=2)


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
