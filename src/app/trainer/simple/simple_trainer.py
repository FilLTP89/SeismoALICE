from pstats import Stats
from re import S
import torch
from plot.plot_tools import get_gofs, plot_generate_classic
from core.trainer.basic_trainer import BasicTrainer
from core.metric.metric import MetricTracker
from core.writer.writer import Writer
from tools.generate_noise import noise_generator
from core.logger.logger import setup_logging
from configuration import app
from torch.nn import DataParallel as DP
from common.common_nn import generate_latent_variable,get_accuracy
from common.common_nn import zerograd,zcat,modalite, count_parameters
from common.common_torch import *
from factory.conv_factory import Network, DataParalleleFactory
from app.agent.simple.generators import Generators
from app.agent.simple.discriminators import Discriminators

class SimpleTrainer(BasicTrainer):
    def __init__(self,cv, trial=None):
        globals().update(cv)
        globals().update(opt.__dict__)

        self.cv         = cv
        self.std        = 1.0
        self.trial      = trial
        self.strategy   = strategy
        self.opt        = opt
        self.logger     = setup_logging()
        
        self.losses_disc = {
            'epochs':'',           'modality':'',
            'Dloss':'',            'Dloss_ali':'',        'Dloss_ali_y':'',  
            'Dloss_marginal':'',   'Dloss_marginal_y':'',
            'Dloss_marginal_zd':'',
        }
        self.losses_gens = {
            'epochs':'',           'modality':'',
            'Gloss':'',            'Gloss_ali':'',        'Gloss_ali_x':'',
            'Gloss_ali_y':'',      'Gloss_marginal':'',   'Gloss_marginal_y':'',
            'Gloss_marginal_zd':'',

            'Gloss_rec':'',        'Gloss_rec_y':'',     
            'Gloss_rec_zd':'',    
        }

        self.gradients_gens = {
            'epochs':'',    'modality':'',
            'Fxy':'',       'Gy':'',
        }
        self.gradients_disc = {
            'epochs':'',    'modality':'',
            'Dy':'',        'Dsy':'',
            'Dzb':'',       'Dszb':'', 'Dyz':'',
            'Dszf':''
        }

        self.logger.info("Setting Tensorboard for the training dataset ...")
        loss_writer             = Writer(log_dir=self.opt.config['log_dir']['debug.losses_writer'], 
                                    logger=self.logger)
        gradients_writer        = Writer(log_dir=self.opt.config['log_dir']['debug.gradients_writer'],
                                    logger=self.logger)
        self.training_writer    = Writer(log_dir=self.opt.config['log_dir']['train_writer'],
                                    logger=self.logger)
        self.validation_writer  = Writer(log_dir=self.opt.config['log_dir']['valid_writer'],
                                    logger=self.logger)
        self.debug_writer       = Writer(log_dir=self.opt.config['log_dir']['debug_writer'], 
                                    logger=self.logger)
        
        self.logger.info("Tracking metrics ...")
        self.losses_disc_tracker    = MetricTracker(self.losses_disc,loss_writer,'Dloss')
        self.losses_gen_tracker     = MetricTracker(self.losses_gens,loss_writer,'Gloss')
        self.gradients_tracker_gen  = MetricTracker(self.gradients_gens,gradients_writer,'Ggradient')
        self.gradients_tracker_disc = MetricTracker(self.gradients_disc,gradients_writer,'Dgradient')
        
        
        network   = Network(DataParalleleFactory())
        self.logger.info("Creating Generators Agent ...")
        self.gen_agent  = Generators(
                        network=network, config=self.opt.config, logger=self.logger,
                        accel=DP, opt=self.opt, gradients_tracker = self.gradients_tracker_gen,
                        debug_writer = self.debug_writer)

        self.logger.info("Creating Discriminator Agent ...")
        self.disc_agent = Discriminators(network=network, config=self.opt.config, logger=self.logger,
                        accel=DP, opt=self.opt, gradients_tracker = self.gradients_tracker_disc,
                        debug_writer = self.debug_writer)

        self.logger.info("Loading the dataset ...")
        self.training_loader  = trn_loader
        self.validation_loader= vld_loader
        self.test_loader      = tst_loader
        self.bce_loss         = torch.nn.BCELoss(reduction='mean').cuda()

        super(SimpleTrainer,self).__init__(
            settings  = self.opt, logger = self.logger, config = self.opt,
            models    = [self.gen_agent,self.disc_agent],
            optimizer = [self.gen_agent.optimizer,self.disc_agent.optimizer],
            losses    = [self.losses_disc, self.losses_gens], strategy = self.strategy['unique'])
        
        
        self.logger.info("Parameters of Generators ")
        count_parameters(self.gen_agent.generators)
        self.logger.info(f"Learning rate : {self.opt.config['hparams']['generators.lr']}")

        self.logger.info("Parameters of Discriminators")
        count_parameters(self.disc_agent.discriminators)
        self.logger.info(f"Learning rate : {self.opt.config['hparams']['discriminators.lr']}")

        self.logger.info(f"Number of GPU : {torch.cuda.device_count()} GPUs")
        self.logger.info(f"Root checkpoint {self.opt.root_checkpoint}")
        self.logger.info(f"Saving epoch every {self.opt.save_checkpoint} iterations")
        
        self.logger.info(f"Root summary")
        for _, root in self.opt.config['log_dir'].items():
            self.logger.info(f"Summary:{root}")

    
    def on_training_epoch(self, epoch, bar):
        for idx, batch in enumerate(self.training_loader):
            y, *others = batch
            y   = y.to(app.DEVICE, non_blocking = True)

            zyy,zyx, *other = generate_latent_variable(batch=len(y))
            pack = y,zyy,zyx
            self.train_discriminators(ncritics=1, batch=pack,epoch=epoch, 
                modality='train',net_mode=['eval','train'])
            self.train_generators(ncritics=1, batch=pack, epoch=epoch, 
                modality='train',net_mode=['train','train'])

        if epoch%self.opt.config['hparams']['training_epochs'] == 0:
            self.gradients_tracker_gen.write(epoch=epoch, modality = ['train'])
            self.gradients_tracker_disc.write(epoch=epoch,modality = ['train'])
        
        Gloss = self.losses_gen_tracker.get('Gloss',epoch,'train')
        Dloss = self.losses_disc_tracker.get('Dloss',epoch,'train')
        bar.set_postfix(Dloss=Dloss, Gloss = Gloss)
    
    def on_validation_epoch(self, epoch, bar):
        for idx, batch in enumerate(self.validation_loader):
            y, *others = batch
            y   = y.to(app.DEVICE, non_blocking = True)
    
            zyy,zyx, *other = generate_latent_variable(batch=len(y))
            pack = y,zyy,zyx
            self.train_discriminators(ncritics=1, batch=pack,epoch=epoch, 
                        modality='eval',net_mode=['eval','eval'])
            self.train_generators(ncritics=1, batch=pack, epoch=epoch, 
                        modality='eval',net_mode=['eval','eval'])
        if epoch%self.opt.config['hparams']['validation_epochs'] == 0:
            self.losses_disc_tracker.write( epoch=epoch, modality = ['train','eval'])
            self.losses_gen_tracker.write(  epoch=epoch, modality = ['train','eval'])

    def train_unic_discriminators(self,y,x,zyy,zxy,epoch,modality,net_mode,*args,**kwargs):
        """ The UnicTrainer class is extended to support different strategy
            WGAN, WGAN-GP, ALICE-explicite, ALICE-implicite
        """
        raise NotImplementedError
    
    def train_unic_generators(self,y,x,zyy,zxy,epoch,modality,net_mode,*args,**kwargs):
        """ The UnicTrainer class is extended for different stragy of training
        """
        raise NotImplementedError

    def on_test_epoch(self, epoch, bar):
        with torch.no_grad(): 
            torch.manual_seed(100)
            if epoch%self.opt.config["hparams"]['test_epochs'] == 0:
                # get accuracy
                self.validation_writer.set_step(mode='test', step=epoch)
                bar.set_postfix(status='saving accuracy and images ... ')
                
                accuracy_bb = get_accuracy(tag='broadband',plot_function=get_gofs,
                    encoder = self.gen_agent.Fxy,
                    decoder = self.gen_agent.Gy,
                    vld_loader = self.test_loader,
                    pfx ="vld_set_bb_unique",opt= self.opt,
                    outf = self.opt.outf, save = False
                )
                self.validation_writer.add_scalar('Accuracy/Broadband',accuracy_bb,epoch)
                bar.set_postfix(accuracy_bb=accuracy_bb)

                # # get weight of encoder and decoder
                # plot filtered reconstruction signal and gof
                figure_fl, gof_fl = plot_generate_classic(tag ='broadband',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.test_loader, pfx="vld_set_bb_unique_hack",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Filtered ...')
                self.validation_writer.add_figure('STFD Filtered', figure_fl)
                self.validation_writer.add_figure('GOF Filtered', gof_fl)

                # plot hybrid broadband reconstruction signal and gof
                figure_hb, gof_hb = plot_generate_classic(tag ='hybrid',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.test_loader, pfx="vld_set_bb_unique_hack",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Hybrid Broadband ...')
                self.validation_writer.add_figure('STFD Hybrid Broadband', figure_hb)
                self.validation_writer.add_figure('GOF Hybrid Broadband', gof_hb)
    
    def train_discriminators(self,ncritics,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,zxy = batch
        for _ in range(ncritics):
            self.train_unic_discriminators(y,zyy,zxy,epoch,modality,net_mode,*args,**kwargs)
            # training could be WGAN, ALICE implicite ALICE explicite, InfoGAN

    def train_generators(self,ncritics,batch,epoch, modality, net_mode, *args, **kwargs):
        y,zyy,zxy = batch
        for _ in range(ncritics):
            self.train_unic_generators(y,zyy,zxy,epoch,modality,net_mode,*args,**kwargs)
            # training could be WGAN, ALICE, InfoGAN
    