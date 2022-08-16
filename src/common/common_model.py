# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Common tools to design ANN'''
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
import sys
import random
import torch
from torch.autograd import Variable

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

def get_truncated_normal(mean=0, sd=1, low=0, upp=10, log=False):
    from scipy.stats import truncnorm

    if log:
        return truncnorm.logpdf(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    else:
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def zcat(*args):
    return torch.cat(args,1)

def track_gradient_change(model):
    try:
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
    except AttributeError:
        total_norm = 0
    return total_norm

def gradient_penalty(critic, real, fake, device):
    
    _real = real
    _fake = fake
    BATCH_SIZE, C, W = _real.shape
    alpha = torch.rand((BATCH_SIZE,1, 1)).repeat(1, C, W).to(device)
    interpolated = _real.data * alpha + _fake.data * (1 - alpha)

    interpolated =  Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate critic scores
    prob_interpolated = critic(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(BATCH_SIZE, -1)
    
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()

def consistency_regularization(netD, real_data, lambda_val=2, M_val = 0.0):
    '''
    Consistency regularization complements the gradient penalty by biasing towards
    the real-data along the manifold connecting the real and fake data.
    ---------------------
    :param netD: Discriminator network that returns the output of the last layer
                and the pen-ultimate layer.
    :param real_data: Real data - Variable
    :param lambda_val: coefficient for the consistency_regularization term
    :param M_val: constant offset M ~ [0, 0.2]
    :return: consistency regularization term
    '''
    dx1, dx1_ = netD(real_data)
    dx2, dx2_ = netD(real_data) # Different from dx1 because of stochastic dropout
    CT = (dx1 - dx2)**2 + 0.1*(dx1_ - dx2_)**2
    cons_reg = torch.max(torch.zeros(CT.size()), lambda_val*CT - M_val).mean()
    return cons_reg


# Choosing `num_centers` random data points as the initial centers
def random_init(dataset, num_centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

# Compute for each data point the closest center
def compute_codes(dataset, centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes

# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device))
    centers /= cnt.view(-1, 1)
    return centers

def cluster(dataset, num_centers, device):

    centers = random_init(dataset, num_centers, device)
    codes = compute_codes(dataset, centers,device)
    num_iterations = 0
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers, device)
        new_codes = compute_codes(dataset, centers, device)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            sys.stdout.write('\n')
            print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
    return centers, codes

class AKA(type):
    """ 'Also Known As' metaclass to create aliases for a class. """
    def __new__(cls, classname, bases, attrs):
        print('in AKA.__new__')
        class_ = type(classname, bases, attrs)
        globals().update({alias: class_ for alias in attrs.get('aliases', [])})
        return class_


