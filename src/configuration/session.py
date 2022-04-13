import argparse
import json


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', default='../actions_bb.txt',help='define actions txt')
    parser.add_argument('--strategy', default='../strategy_bb.txt',help='define strategy txt')
    parser.add_argument('--dataset', default='nt4096_ls128_nzf8_nzd32.pth',help='folder | synth | pth | stead | ann2bb | deepbns')
    parser.add_argument('--dataroot', default='../database/stead',help='Path to dataset') # '/home/filippo/Data/Filippo/aeolus/ann2bb_as4_') # '/home/filippo/Data/Filippo/aeolus/STEAD/waveforms_11_13_19.hdf5',help='path to dataset')
    parser.add_argument('--inventory',default='RM07.xml,LXRA.xml,SRN.xml',help='inventories')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
    parser.add_argument('--batchPercent', type=int,nargs='+', default=[0.8,0.1,0.1], help='train/test/validation %')
    parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--imageSize', type=int, default=4096, help='the height / width of the input image to network')
    parser.add_argument('--latentSize', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--cutoff', type=float, default=1., help='cutoff frequency')
    parser.add_argument('--nzd', type=int, default=32, help='size of the latent space')
    parser.add_argument('--nzf', type=int, default=8, help='size of the latent space')
    parser.add_argument('--ngf', type=int, default=32,help='size of G input layer')
    parser.add_argument('--ndf', type=int, default=32,help='size of D input layer')
    parser.add_argument('--glr', type=float, default=0.0001, help='AE learning rate, default=0.0001')
    parser.add_argument('--rlr', type=float, default=0.0001, help='GAN learning rate, default=0.00005')
    parser.add_argument('--b1', type=float, default=0.5, help='beta1 for Adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--plot', action='store_true', help="flag for plotting")
    parser.add_argument('--outf', default='./imgs', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--mw',type=float,default=4.5,help='magnitude [Mw]')
    parser.add_argument('--dtm',type=float,default=0.01,help='time-step [s]')
    parser.add_argument('--dep',type=float,default=50.,help='epicentral distance [km]')
    parser.add_argument('--scc',type=int,default=0,help='site-class')
    parser.add_argument('--sst',type=int,default=1,help='site')
    parser.add_argument('--scl',type=int,default=1,help='scale [1]')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--nsy',type=int,default=83,help='number of synthetics [1]')
    parser.add_argument('--config', default='./config.json', help='configuration file')
    parser.add_argument('--save_checkpoint',type=int,default=1,help='Number of epochs for each checkpoint')
    parser.set_defaults(stack=False,ftune=False,feat=False,plot=True)
    
    args = parser.parse_args()

    with open(args.config) as f:
        args.config = json.load(f)

    return args


def cleanup(args):
    args = {}
    return args