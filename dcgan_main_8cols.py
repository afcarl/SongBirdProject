from __future__ import print_function
import os
import math
import argparse
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
import numpy as np
from torch.autograd import Variable
from scipy.misc import imsave
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageH',type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imageW',type=int, default=64, help='the width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

parser.add_argument('--fict', action='store_true', help='enable fictitious play')
parser.add_argument('--fict_history', type=int, default=100, help='number of historical copies to save for fictitious play')
# parser.add_argument('--fict_batch', type=int, default=1, help='# of historical agents to sample during each batch')

opt = parser.parse_args()
opt.scaleW = 60
opt.scaleH = 120
print(opt)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
    
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   #transforms.Scale(size=(opt.scaleW,opt.scaleH)),
                                   transforms.RandomCrop((opt.imageH,opt.imageW)),
                                   transforms.ToTensor()
                                   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(4,4), stride=(2,1), padding=(0,1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 10 x 5
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(4,4), stride=(2,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 20 x 6
            nn.ConvTranspose2d(ngf * 2,     ngf, kernel_size=(5,4), stride=(3,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 60 x 7
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=(4,4), stride=(2,1), padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 120 x 8
        )
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)

netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 120 x 8
            nn.Conv2d(nc, ndf, kernel_size=4, stride=(2,1), padding=(1,1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 60 x 7
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=(4,3), stride=(2,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 30 x 5
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(4,3), stride=(2,1), padding=(2,0), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 3
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(8,4), stride=(2,1), padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*8) x 6 x 2
            nn.Conv2d(ndf * 8, 1, kernel_size=(6,2), stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        return output.view(-1, 1)

netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()


# IMPORTANT VARIABLES
##################################################################
input = torch.FloatTensor(opt.batchSize, 1, opt.imageH, opt.imageW)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

#sampling parameters 
Nptscheck = 200
checknoise = torch.FloatTensor(Nptscheck,nz,1,1)


    
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    checknoise = checknoise.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
checknoise = Variable(checknoise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))



    
#SAMPLER   
####################################

params = {'theta': 1, 'mu': np.zeros((1,nz,1,1)), 'sigma': 2}

def OUprocess(x0,dt,N,nz,params):
    #pdb.set_trace()
    x = np.ndarray((N,nz,1,1))
    x[0]= x0 
    for i in np.arange(N):
        if i==0:
            continue
        else:
            dx = params['theta']*(params['mu'] - x[i-1])*dt
            + params['sigma']*np.sqrt(dt)*np.random.normal(0,1,(1,nz,1,1))
            
            x[i] = x[i-1] + dx
            
    return x     



def line_joining_2pts(n1,n2,Npts):
    
    
    lambdaa = np.linspace(0,1,Npts)
    x = np.ndarray((Npts,nz,1,1))
    for i,lmbda in enumerate(lambdaa):
        x[i,:,:,:] = lmbda*n1 + (1-lmbda)*n2
        
    return x


params = {'Amp': 1, 'freq': 2, 'sigma': 1}
def sinusoid_noiseprocess(N,T,params):
    
    t = np.linspace(start = 0, stop = T, num = N)
    x = np.zeros((N,nz,1,1))
    
    for i in range(N):
        x[i] = params['Amp']*np.sin(2*np.pi*params['freq']*t[i]) 
        + params['sigma']*np.random.normal(0,1,(1,nz,1,1))
            
    return x

params = {'alpha': -0.9, 'sigma': 0.1}
def gaussian_markov_chain(x0,N,params):
    x = np.zeros((N,nz,1,1))
    x[0] = x0
    for i in np.arange(N):
        if i==0:
            continue
        else:
            x[i] = params['alpha']*x[i-1] + params['sigma']*np.random.normal((1,nz,1,1))
        
    return x
        
####################################


minibatchLossD = np.zeros(len(dataloader))
minibatchLossG = np.zeros(len(dataloader))
logpt = 200

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
       
        real_cpu, _ = data
        aa = real_cpu.size()
        
        
        aa = aa[0] #number of samples in minibatch
        rr = torch.Tensor(aa,nc,opt.imageH,opt.imageW)
        #rr[:,0,:,:] = real_cpu
        
        rr[:,0,:,:] = 0.2989*real_cpu[:,0,:,:] + 0.5870*real_cpu[:,1,:,:] + 0.1140*real_cpu[:,2,:,:]
        #pdb.set_trace()
        
        
        batch_size = real_cpu.size(0)
        input.data.resize_(rr.size()).copy_(rr)
        label.data.resize_(batch_size).fill_(real_label)

        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        
        fake = netG(noise)
        #pdb.set_trace()
        input.data.copy_(fake.data)
        label.data.fill_(fake_label)
        output = netD(input)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        
        optimizerD.step() #parameter update step

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        netG.zero_grad()
        label.data.fill_(real_label) # fake labels are real for generator cost
        noise.data.normal_(0, 1)
        fake = netG(noise)
        
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        
        
        minibatchLossD[i] = errD.data[0]
        minibatchLossG[i] = errG.data[0]
        
        ### SHOW LOSS AFTER SOME BATCHES ####
        if (i % logpt == 0) & (i > 0):
            print('[%d/%d][%d/%d] Loss_D: %f (%f) Loss_G: %f (%f)'
                  % (epoch, opt.niter, i, len(dataloader),
                    np.mean(minibatchLossD[i-logpt:i]), np.std(minibatchLossD[i-logpt:i]),
                     np.mean(minibatchLossG[i-logpt:i]),  np.std(minibatchLossG[i-logpt:i]) ))
            
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf)
            fixed_noise.data.normal_(0,1)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d_batchnumb_%d.png' % (opt.outf, epoch, i))
            
        #if (i % logpt*10 ==0) & (i > 0):
        #    plt.plot(minibatchLossD[:i],'
            
            

        ### CREATE SAMPLE SPECTROGRAMS AFTER SOME BATCHES         
            
            #n1 = np.random.normal(0,1,(1,nz,1,1))
            #n2 = np.random.normal(0,1,(1,nz,1,1))
            #cc = OUprocess(n1,0.005,Nptscheck,nz,params)
          
            #x0 = np.random.normal(0,1,(1,nz,1,1))
            #cc = gaussian_markov_chain(x0,Nptscheck,params)
            #cc = torch.from_numpy(cc)
            #checknoise.data.resize_(cc.size()).copy_(cc)
            
            #fakke = netG(checknoise)
            #fakke = fakke.data
            #fakke = fakke.cpu().numpy()
                 
            #fakke = np.split(fakke,Nptscheck,axis=0)
            
            #fakke = np.concatenate(fakke,axis=3)    
            
            #fakke = np.reshape(fakke,(fakke.shape[2],fakke.shape[3]) )
            
            #save image
            #imsave(opt.outf+'/epoch_'+str(epoch)+'_batchnumb_'+str(i)+'.png',fakke)
            
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))