import os
import numpy as np 
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm 
import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow
import wandb 
from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401
from datasets import CelebA_Dataset, FFHQ_Dataset 
import multiprocessing 




class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 3, 512, 512)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        
        self.conv_layers = nn.Sequential(
            # First layer: reduce channels from 3 to 16, spatial dims from 512 to 256
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Second layer: keep channels at 16, spatial dims from 256 to 128
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Third layer: reduce channels from 16 to 32, spatial dims from 128 to 64
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Fourth layer: reduce channels from 32 to 64, keep spatial dims at 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Fifth layer: reduce channels from 64 to 1, keep spatial dims at 64
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1) 
        # self.fin_lin = nn.Linear(3840, 1024)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 3, 64, 64)  
        h = self.conv_in(h) 
        # h = self.conv_layers(h) 
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim) 
        # z = self.fin_lin(z) 
        
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1) 
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    
        # setup the non-linearity
        self.act = nn.ReLU(inplace=True) 
        
        self.transposed_conv = nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1) 
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h) 
        mu_img = self.transposed_conv(mu_img) 
        # mu_img = self.deconv_layers(mu_img) 
        return mu_img


class VAE(nn.Module):
    def __init__(self, z_dim=512, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, conv=False, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(512, 2))

        # create the encoder and decoder networks

        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim) 
            
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim) 
        

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 3, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), 512, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 3, 64, 64)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([
        test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
    win_test_reco = vis.images(
        list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
        opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', default=8, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=512, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='test1')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log') 
    parser.add_argument('--dataset', type=str, default='celeba', help='dataset to use',
        choices=['celeba', 'ffhq'] )
    args = parser.parse_args() 
    parser.add_argument('--load', default=False, type=bool, help='load a model')   

    torch.cuda.set_device(args.gpu)

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=4500)

    train_elbo = []


    iteration = 0 
    
    BATCH_SIZE = args.b 
    
    # train_loader = DataLoader(CelebA_Dataset(mode=0), batch_size=BATCH_SIZE,
    #                     num_workers=8,         
    #                     pin_memory=True,
    #                     persistent_workers=True,
    #                     prefetch_factor=4, 
    #                     shuffle=True) 
    
    
    if args.dataset == 'celeba':
        dataset = CelebA_Dataset(mode=0) 
    else:
        dataset = FFHQ_Dataset() 
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=8, 
                              persistent_workers=True) 
    
    dataset_size = len(train_loader) * BATCH_SIZE
    length = len(train_loader) 
    
    best_elbo=-np.inf  
    
    wandb.init(project="HSpace-SAEs", entity="a-ijishakin",
                        name='vae_training_run')  
    
    current_device = torch.cuda.current_device() 
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Current device name: {device_name}") 
    
    
    
    #count the amount of parameters in the model
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad) 
    print(f"Number of parameters in the model: {num_params}") 
    
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    for epoch in range(1000):
        epoch_elbo = 0 
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
            pre_load = time.time() 
            for i, x in enumerate(train_loader):
                # print(f"Time taken to load the data: {post_load} seconds") 
                                
                iteration += 1
                vae.train()
                anneal_kl(args, vae, iteration)
                optimizer.zero_grad()
                # transfer to GPU
                x = x['img'] 
                x = x.to('cuda:0', non_blocking=True) 
                # print(f"Time taken to transfer the data to GPU: {post_iter} seconds") 
                
                # wrap the mini-batch in a PyTorch Variable
                x = Variable(x)
                pre_step = time.time() 
                # do ELBO gradient and accumulate loss
                obj, elbo = vae.elbo(x, dataset_size) 
                post_step = time.time() - pre_step   
                # print(f"Time taken to compute the ELBO: {post_step} seconds") 
                
                
                if utils.isnan(obj).any():
                    raise ValueError('NaN spotted in objective.')
                obj.mean().mul(-1).backward()
                elbo_running_mean.update(elbo.mean().data) 
                epoch_elbo += elbo.mean().data  
                optimizer.step()

                # report training diagnostics
                # if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg) 
                
                wandb.log({'train_elbo': 
                    elbo_running_mean.avg}, 
                            step= (epoch * length) + i)   
                pbar.update(1)
            
            epoch_elbo /= len(train_loader) 
            
            if epoch_elbo > best_elbo:
                torch.save(vae.state_dict(), f'best_model-{args.dataset}.pt') 
            
                
                # print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                #     iteration, time.time() - batch_time, vae.beta, vae.lamb,
                #     elbo_running_mean.val, elbo_running_mean.avg))

                # vae.eval()

                # # plot training and test ELBOs
                # if args.visdom:
                #     display_samples(vae, x, vis)
                #     plot_elbo(train_elbo, vis)

                # utils.save_checkpoint({
                #     'state_dict': vae.state_dict(),
                #     'args': args}, args.save, 0)
                # eval('plot_vs_gt_' + args.dataset)(vae, train_loader.dataset,
                #     os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))


    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, 0)
    dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
    eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))
    return vae


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn') 
    model = main()
