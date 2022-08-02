import torch
import streamlit as st
from tqdm import tqdm as tq
from plot.plot_tools import get_gofs, plot_generate_classic
from plot.plot_tools import get_histogram
from core.trainer.basic_trainer import BasicTrainer
from core.metric.metric import MetricTracker
from core.writer.writer import Writer
from tools.generate_noise import noise_generator
from core.logger.logger import setup_logging
from configuration import app
from torch.nn import DataParallel as DP
from common.common_nn import get_accuracy,patch
from common.common_nn import count_parameters
from common.common_torch import *
from database.latentset import get_latent_dataset
from factory.conv_factory import Network, DataParalleleFactory
from app.agent.unic.generators import Generators

class UnitaryTrainerGenerator(BasicTrainer):
    def __init__(self,cv,losses_gens, gradients_gens, strategy_generator, 
                    trial=None, *args,**kwargs):
        globals().update(cv)
        globals().update(opt.__dict__)

        self.cv         = cv
        self.trial      = trial
        self.strategy   = strategy
        self.opt        = opt
        self.logger     = setup_logging()

        self.losses_gens    = losses_gens
        self.gradients_gens = gradients_gens

        self.logger.info("Setting Tensorboard for the training dataset ...")
        loss_writer             = Writer(log_dir=self.opt.config['log_dir']["unitary.gen"]['debug.losses_writer'], 
                                    logger=self.logger)
        gradients_writer        = Writer(log_dir=self.opt.config['log_dir']["unitary.gen"]['debug.gradients_writer'],
                                    logger=self.logger)
        self.training_writer    = Writer(log_dir=self.opt.config['log_dir']["unitary.gen"]['train_writer'],
                                    logger=self.logger)
        self.validation_writer  = Writer(log_dir=self.opt.config['log_dir']["unitary.gen"]['valid_writer'],
                                    logger=self.logger)
        self.debug_writer       = Writer(log_dir=self.opt.config['log_dir']["unitary.gen"]['debug_writer'], 
                                    logger=self.logger)

        self.logger.info("Tracking metrics ...")
        self.losses_gen_tracker     = MetricTracker(self.losses_gens,loss_writer,'Gloss')
        self.gradients_tracker_gen  = MetricTracker(self.gradients_gens,gradients_writer,'Ggradient')

        network   = Network(DataParalleleFactory())
        self.logger.info("Creating Generators Agent ...")
        self.gen_agent  = Generators(network=network, config=self.opt.config, logger=self.logger,
                        accel=DP, opt=self.opt, gradients_tracker = self.gradients_tracker_gen,
                        debug_writer = self.debug_writer, strategy = strategy_generator)
            
        self.logger.info("Loading the dataset ...")
        self.data_trn_loader, self.data_vld_loader,self.data_tst_loader = trn_loader, vld_loader, tst_loader
        self.lat_trn_loader, self.lat_vld_loader, self.lat_tst_loader   = get_latent_dataset(nsy=self.opt.nsy,\
            batch_size=self.opt.batchSize)

        self.bce_loss        = torch.nn.BCELoss(reduction='mean')
        self.bce_logit_loss  = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.l1_loss         = torch.nn.L1Loss(reduction='mean')

        super(UnitaryTrainerGenerator,self).__init__(
            settings  = self.opt, logger = self.logger, config = self.opt,
            models    = {'generators':self.gen_agent},
            losses    = {'generators':self.losses_gens}, 
            strategy  = self.strategy['simple'], *args, **kwargs)

        self.logger.info("Parameters of Generators ")
        count_parameters(self.gen_agent.generators)
        self.logger.info(f"Learning rate Encoder : {self.opt.config['hparams']['generators.encoder.lr']}")
        self.logger.info(f"Learning rate Decoder : {self.opt.config['hparams']['generators.decoder.lr']}")
        
        self.logger.info(f"Number of GPU        : {torch.cuda.device_count()} GPUs")
        self.logger.info(f"Batch Size per GPU   : {self.opt.batchSize//torch.cuda.device_count()}")
        self.logger.info(f"Root checkpoint      : {self.opt.root_checkpoint}")
        self.logger.info(f"Saving epoch every   : {self.opt.save_checkpoint} iterations")
        
        self.logger.info(f"Root summary")
        for _, root in self.opt.config['log_dir'].items():
            if isinstance(root,dict):
                for (_,subroot) in root.items():
                    self.logger.info(f"\t Summary{subroot}")
            else:
                self.logger.info(f"Summary:{root}")
    
    def on_training_epoch(self, epoch, bar):
        _bar = tq(enumerate(zip(self.data_trn_loader,self.lat_trn_loader)),
        position=1,leave=False, desc='train.', total=len(self.data_trn_loader))
        for idx, (batch_data,batch_latent)  in _bar:
            y,x, *others      = batch_data
            zyy,zyx,*others = batch_latent
            y, x  = y.to(app.DEVICE,non_blocking=True), x.to(app.DEVICE, non_blocking = True)
            zyy,zyx  = zyy.to(app.DEVICE,non_blocking=True),zyx.to(app.DEVICE,non_blocking=True)
            
            pack = patch(y=y,zyy=zyy,zyx=zyx, x=x)

            self.train_generators(ncritics=1, batch=pack, epoch=epoch, modality='train',net_mode=['train','train'])

        if epoch%self.opt.config['hparams']['training_epochs'] == 0:
            self.gradients_tracker_gen.write(epoch=epoch, modality = ['train'])
        
        Gloss = self.losses_gen_tracker.get('Gloss',epoch,'train')
        bar.set_postfix(Gloss = Gloss)
    
    def on_validation_epoch(self, epoch, bar):
        _bar = tq(enumerate(zip(self.data_vld_loader,self.lat_vld_loader)),
        position=1, leave=False, desc='valid.', total=len(self.data_vld_loader))
        for idx, (batch_data, batch_latent) in _bar:
            y,x,*others = batch_data
            zyy,zyx,*others= batch_latent
            y, x       = y.to(app.DEVICE, non_blocking   = True), x.to(app.DEVICE, non_blocking = True)
            zyy, zyx   = zyy.to(app.DEVICE, non_blocking = True), zyx.to(app.DEVICE, non_blocking = True)
            
            pack = patch(y=y,zyy=zyy,zyx=zyx,x=x)
            self.train_generators(batch=pack, epoch=epoch, modality='eval',net_mode=['eval','eval'])
        if epoch%self.opt.config['hparams']['validation_epochs'] == 0:
            self.losses_gen_tracker.write(epoch=epoch, modality = ['train','eval'])
    
    def on_test_epoch(self, epoch, bar , *args, **kwargs):
        with torch.no_grad(): 
            torch.manual_seed(self.opt.manualSeed)
            if epoch%self.opt.config["hparams"]['test_epochs'] == 0:
                self.validation_writer.set_step(mode='test', step=epoch)
                self.test_generators(bar=bar, writer=self.validation_writer, epoch=epoch, *args, **kwargs)
                self.gen_agent.track_weight(epoch)

    def train_generators(self, batch, epoch, modality, net_mode, *args, **kwargs):
        """ The SimpleTrainer class is extended to support different strategy
            WGAN, WGAN-GP, ALICE-explicite, ALICE-implicite
        """
        raise NotImplementedError
    
    def test_generators(self, bar, writer, epoch, *args, **kwargs):
        """ The tes for generator : encoder/decoder could be different depend of
            the wished strategy
        """
        raise NotImplementedError