import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, nz, ngf, nc, gpu_ids):
        super(_netG, self).__init__()
        self.ngpu = len(gpu_ids)
        self.gpu_ids = gpu_ids
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.gpu_ids)
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ndf, nc, gpu_ids):
        super(_netD, self).__init__()
        self.ngpu = len(gpu_ids)
        self.gpu_ids = gpu_ids
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.gpu_ids)
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class GeneratorCNN(nn.Module):
    def __init__(self, nz, ngf, nc, repeat_num, gpu_ids, dropout=0.5):
        super(GeneratorCNN, self).__init__()
        self.num_gpu = len(gpu_ids)
        self.gpu_ids = gpu_ids
        layers = []

        # input is Z, going into a convolution
        k = 2**repeat_num
        layers.append(nn.ConvTranspose2d(nz, ngf * k, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(ngf * k))
        layers.append(nn.ReLU(True))

        for i in range(repeat_num)[::-1]:
            k = 2 ** i
            # state size. (ngf*k*2) x 4 x 4
            layers.append(nn.ConvTranspose2d(ngf * k * 2, ngf * k, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ngf * k))
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(True))

        # state size. (ngf) x 64 x 64
        layers.append(nn.ConvTranspose2d(ngf,nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())
        # state size. (nc) x 128 x 128
        
        self.main = torch.nn.Sequential(*layers)
        
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.gpu_ids)
        else:
            output = self.main(input)

        return output


class DiscriminatorCNN(nn.Module):
    def __init__(self, ndf, nc, repeat_num, gpu_ids, dropout=0.5):
        super(DiscriminatorCNN, self).__init__()
        self.num_gpu = len(gpu_ids)
        self.gpu_ids = gpu_ids
        layers = []

        layers.append(nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False)) 
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(repeat_num):
            k = 2 ** i
            # state size. (ndf) x 64 x 64
            layers.append(nn.Conv2d(ndf*k, ndf*k*2, 4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ndf * k * 2))
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf*16) x 4 x 4
        layers.append(nn.Conv2d(ndf*k*2, 1, 4, stride=1, padding=0,bias=False))
        layers.append(nn.Sigmoid())
        
        self.main = torch.nn.Sequential(*layers)
        
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.gpu_ids)
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

