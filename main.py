import torch
import torchvision


def main():
    trainset = torchvision.datasets.cifar.CIFAR10(root='./data', train=True, download=True)
    print(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
    print(trainloader)

    pass


if __name__ == '__main__':
    main()
