from pstats import Stats
from re import S
import torch
from tqdm import tqdm as tq
from plot.plot_tools import get_gofs, plot_generate_classic
from plot.plot_tools import get_histogram, get_latent_rep
from core.trainer.basic_trainer import BasicTrainer
from core.metric.metric import MetricTracker
from core.writer.writer import Writer
from tools.generate_noise import noise_generator
from core.logger.logger import setup_logging
from configuration import app
from torch.nn import DataParallel as DP
from common.common_nn import generate_latent_variable_3D, get_accuracy,patch
from common.common_nn import count_parameters
from common.common_torch import *
from database.latentset import get_latent_dataset
from factory.conv_factory import Network, DataParalleleFactory
from app.agent.simple.generators import Generators
from app.agent.simple.discriminators import Discriminators

class SimpleTrainer(BasicTrainer):
    """ This class is an extension of the BasicTrainer class.
        The BasicTrainer works a template pattern.So the base logic to launch
        a training is already implemented on it.
        The SimpleTrainer class should be call to train over a one type of dataset : 
        "Broadband" or "Fitered", as example.
        To train simultaneously on two different types of dataset like (broadband
        and filtered) see UnicTrainer implementation instead.
    """
    def __init__(self,cv,losses_disc, losses_gens, gradients_gens, gradients_disc, prob_disc,
                    trial=None,*args,**kwargs):
        globals().update(cv)
        globals().update(opt.__dict__)

        self.cv         = cv
        self.trial      = trial
        self.strategy   = strategy
        self.opt        = opt
        self.logger     = setup_logging()

        self.losses_disc     = losses_disc
        self.losses_gens     = losses_gens
        self.gradients_gens  = gradients_gens
        self.gradients_disc  = gradients_disc
        self.prob_disc       = prob_disc
        

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
        self.prob_disc_tracker      = MetricTracker(self.prob_disc,loss_writer,'Probs')
        self.gradients_tracker_gen  = MetricTracker(self.gradients_gens,gradients_writer,'Ggradient')
        self.gradients_tracker_disc = MetricTracker(self.gradients_disc,gradients_writer,'Dgradient')
        
        
        network   = Network(DataParalleleFactory())
        self.logger.info("Creating Generators Agent ...")
        self.gen_agent  = Generators(network=network, config=self.opt.config, logger=self.logger,
                        accel=DP, opt=self.opt, gradients_tracker = self.gradients_tracker_gen,
                        debug_writer = self.debug_writer)

        self.logger.info("Creating Discriminator Agent ...")
        self.disc_agent = Discriminators(network=network, config=self.opt.config, logger=self.logger,
                        accel=DP, opt=self.opt, gradients_tracker = self.gradients_tracker_disc,
                        debug_writer = self.debug_writer)

        self.logger.info("Loading the dataset ...")
        self.data_trn_loader, self.data_vld_loader,self.data_tst_loader = trn_loader, vld_loader, tst_loader
        self.lat_trn_loader, self.lat_vld_loader, self.lat_tst_loader   = get_latent_dataset(nsy=self.opt.nsy,\
                        batch_size=self.opt.batchSize)

        self.bce_loss        = torch.nn.BCELoss(reduction='mean')
        self.bce_logit_loss  = torch.nn.BCEWithLogitsLoss(reduction='mean')

        super(SimpleTrainer,self).__init__(
            settings  = self.opt, logger = self.logger, config = self.opt,
            models    = {'generators':self.gen_agent,'discriminators':self.disc_agent},
            losses    = {'generators':self.losses_disc, 'discriminators':self.losses_gens}, 
            strategy  = self.strategy['simple'])
        
        
        self.logger.info("Parameters of Generators ")
        count_parameters(self.gen_agent.generators)
        self.logger.info(f"Learning rate Encoder : {self.opt.config['hparams']['generators.lr']}")

        self.logger.info("Parameters of Discriminators")
        count_parameters(self.disc_agent.discriminators)
        self.logger.info(f"Learning rate : {self.opt.config['hparams']['discriminators.lr']}")

        self.logger.info(f"Number of GPU        : {torch.cuda.device_count()} GPUs")
        self.logger.info(f"Batch Size per GPU   : {self.opt.batchSize//torch.cuda.device_count()}")
        self.logger.info(f"Root checkpoint      : {self.opt.root_checkpoint}")
        self.logger.info(f"Saving epoch every   : {self.opt.save_checkpoint} iterations")
        
        self.logger.info(f"Root summary")
        for _, root in self.opt.config['log_dir'].items():
            self.logger.info(f"Summary:{root}")

    
    def on_training_epoch(self, epoch, bar):
        _bar = tq(enumerate(zip(self.data_trn_loader,self.lat_trn_loader)),
        position=1,leave=False, desc='train.', total=len(self.data_trn_loader))
        for idx, (batch_data,batch_latent)  in _bar:
            y, *others      = batch_data
            zyy,zyx,*others = batch_latent
            y  = y.to(app.DEVICE,non_blocking=True)
            zyy,zyx  = zyy.to(app.DEVICE,non_blocking=True),zyx.to(app.DEVICE,non_blocking=True)
            
            pack = patch(y=y,zyy=zyy,zyx=zyx, x=None)
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
        _bar = tq(enumerate(zip(self.data_vld_loader,self.lat_vld_loader)),
        position=1, leave=False, desc='valid.', total=len(self.data_vld_loader))
        for idx, (batch_data, batch_latent) in _bar:
            y,*others      = batch_data
            zyy,zyx,*others= batch_latent
            y          = y.to(app.DEVICE, non_blocking   = True)
            zyy, zyx   = zyy.to(app.DEVICE, non_blocking = True), zyx.to(app.DEVICE, non_blocking = True)
            
            pack = patch(y=y,zyy=zyy,zyx=zyx)
            self.train_discriminators(batch=pack,epoch=epoch, 
                        modality='eval',net_mode=['eval','eval'])
            self.train_generators(batch=pack, epoch=epoch, 
                        modality='eval',net_mode=['eval','eval'])
        if epoch%self.opt.config['hparams']['validation_epochs'] == 0:
            self.losses_disc_tracker.write(epoch=epoch, modality = ['train','eval'])
            self.losses_gen_tracker.write(epoch=epoch, modality = ['train','eval'])
            self.prob_disc_tracker.write(epoch=epoch, modality = ['train','eval'])

    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        """ The SimpleTrainer class is extended to support different strategy
            WGAN, WGAN-GP, ALICE-explicite, ALICE-implicite
        """
        raise NotImplementedError

    def train_generators(self,batch,epoch, modality, net_mode, *args, **kwargs):
        """ The SimpleTrainer class is extended to support different strategy
            WGAN, WGAN-GP, ALICE-explicite, ALICE-implicite
        """
        raise NotImplementedError

    def on_test_epoch(self, epoch, bar):
        with torch.no_grad(): 
            torch.manual_seed(100)
            if epoch%self.opt.config["hparams"]['test_epochs'] == 0:
                # get accuracy
                self.validation_writer.set_step(mode='test', step=epoch)
                bar.set_postfix(status='saving accuracy and images ... ')

                figure_histo_z,figure_histo_y = get_histogram(Fy=self.gen_agent.Fy, 
                Gy = self.gen_agent.Gy, trn_set = (self.data_tst_loader, self.lat_tst_loader))
                bar.set_postfix(status='saving z distribution ...')
                self.validation_writer.add_figure('z Histogram', figure_histo_z)
                bar.set_postfix(status='saving y distribution ...')
                self.validation_writer.add_figure('y Histogram ', figure_histo_y)

                accuracy_bb = get_accuracy(tag='broadband',plot_function=get_gofs,
                    encoder = self.gen_agent.Fy,
                    decoder = self.gen_agent.Gy,
                    vld_loader = self.data_tst_loader,
                    pfx ="vld_set_bb_unique",opt= self.opt,
                    outf = self.opt.outf, save = False
                )
                self.validation_writer.add_scalar('Accuracy/Broadband',accuracy_bb,epoch)
                bar.set_postfix(accuracy_bb=accuracy_bb)

                # get weight of generators and discriminators
                self.gen_agent.track_weight(epoch)
                self.disc_agent.track_weight(epoch)
                # plot filtered reconstruction signal and gof
                figure_bb, gof_bb = plot_generate_classic(tag ='broadband',
                        Qec= self.gen_agent.Fy, Pdc= self.gen_agent.Gy,
                        trn_set= self.data_tst_loader,
                        pfx="vld_set_bb_unique",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Broadband ...')
                self.validation_writer.add_figure('STFD Broadband', figure_bb)
                self.validation_writer.add_figure('GOF Broadband', gof_bb)
    
    
    