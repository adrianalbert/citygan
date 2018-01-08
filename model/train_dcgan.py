from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from data_loader import get_loader
from PIL import Image
import numpy as np
import json
from models_dcgan import _netG, _netD, GeneratorCNN, DiscriminatorCNN, weights_init

def str2bool(v):
    return v.lower() in ('true', '1')

def str2intlist(v):
    lst = v.split(",")
    lst_parsed = []
    for l in lst:
        try:
            lst_parsed += [int(l)]
        except ValueError:
            lst_parsed += [float(l)]
    return lst_parsed

def str2list(v):
    return v.split(",")

def str2bool2list(v):
    if 'true' in v.lower() or 'false' in v.lower():
        return str2bool(v)
    return str2intlist(v)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='number of input channels in image')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lr_halve', type=int, default=20, help='learning rate halving epochs, default=20')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', type=str2intlist, default=None, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--custom_loader', type=str2bool, default=False)
parser.add_argument('--flips', type=str2bool, default=False)
parser.add_argument('--rotate_angle', type=int, default=0)
parser.add_argument('--take_log', type=str2bool2list, default=False)
parser.add_argument('--normalize', type=str2bool, default=False)
parser.add_argument('--use_channels', type=str2bool2list, default=None)
parser.add_argument('--fix_dynamic_range', type=str2bool, default=False)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.nc == 3:
    loader = dset.folder.default_loader
else:
    loader = lambda path: Image.open(path).convert('L')

if opt.custom_loader:
    dataloader = get_loader(opt.dataroot, None, opt.batchSize, opt.imageSize, num_workers=opt.workers, shuffle=True, load_attributes=None, flips=opt.flips, rotate_angle=opt.rotate_angle, take_log=opt.take_log, normalize=opt.normalize, use_channels=opt.use_channels, custom_loader=opt.custom_loader)
    dataset = True
elif opt.dataset in ['imagenet', 'folder', 'lfw', 'spatial-maps']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                                loader=loader,
                                transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(opt.nc, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
if not opt.custom_loader:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = len(opt.gpu_ids)
gpu_ids = opt.gpu_ids
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
sigma_noise = 0.04
repeat_num = int(np.log2(opt.imageSize)) - 3
print("Networks have %d layers" % repeat_num)

def save_config(config):
    param_path = os.path.join(config.outf, "params.json")

    print("[*] MODEL dir: %s" % config.outf)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

save_config(opt)

# custom weights initialization called on netG and netD
print("Using %d GPUs" % ngpu)
# netG = _netG(nz, ngf, nc, gpu_ids)
netG = GeneratorCNN(nz, ngf, nc, repeat_num, gpu_ids, dropout=0.0)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# netD = _netD(ndf, nc, gpu_ids)
netD = DiscriminatorCNN(ndf, nc, repeat_num, gpu_ids, dropout=0.2)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
instance_noise = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    instance_noise = instance_noise.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
lr = opt.lr
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    if epoch % opt.lr_halve == 0:
        lr /= 2.0
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        instance_noise.resize_as_(real_cpu).normal_(0,sigma_noise)
        input.resize_as_(real_cpu).copy_(real_cpu + instance_noise)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu + instance_noise,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
