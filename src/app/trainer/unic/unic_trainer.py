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
from tqdm import tqdm as tq
from torch.nn import DataParallel as DP
from common.common_nn import patch,get_accuracy
from common.common_nn import count_parameters
from database.latentset import get_latent_dataset, LatentDataset
from common.common_torch import *
from factory.conv_factory import Network, DataParalleleFactory
from app.agent.unic.generators import Generators
from app.agent.unic.discriminators import Discriminators

class UnicTrainer(BasicTrainer):
    """ docstring for UnicTrainer: 
        This class is an extention of the Basic Trainer
        His goal is to trainer encoder_unic and decoder unic over. 
        So it should be call withe app.unic.generators and app.unic.discriminators agents
    """
    def __init__(self,cv,losses_disc, losses_gens, gradients_gens, 
                    gradients_disc, strategy_discriminator, strategy_generator ,prob_disc, trial=None):
        
        globals().update(cv)
        globals().update(opt.__dict__)

        self.cv         = cv
        self.std        = 1.0
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
                        debug_writer = self.debug_writer, strategy= strategy_generator)

        self.logger.info("Creating Discriminator Agent ...")
        self.disc_agent = Discriminators(network=network, config=self.opt.config, logger=self.logger,
                        accel=DP, opt=self.opt, gradients_tracker = self.gradients_tracker_disc,
                        debug_writer = self.debug_writer, strategy=strategy_discriminator)

        self.logger.info("Loading the dataset ...")
        self.data_trn_loader,  self.data_vld_loader, self.data_tst_loader = trn_loader,vld_loader,tst_loader
        self.lat_trn_loader, self.lat_vld_loader, self.lat_tst_loader   = get_latent_dataset(
            dataset=LatentDataset,nsy=self.opt.nsy,batch_size=self.opt.batchSize)
        self.bce_loss         = torch.nn.BCELoss(reduction='mean').cuda()
        self.bce_logit_loss   = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
        self.l1_loss          = torch.nn.L1Loss(reduction='mean').cuda()

        super(UnicTrainer,self).__init__(
            settings  = self.opt, logger = self.logger, config = self.opt,
            models    = {'generators':self.gen_agent,'discriminators':self.disc_agent},
            losses    = {'generators':self.losses_disc, 'discriminators':self.losses_gens}, 
            strategy  = self.strategy['unique'])
        
        self.logger.info("Parameters of Generators ")
        count_parameters(self.gen_agent.generators)
        self.logger.info(f"Learning rate : {self.opt.config['hparams']['generators.encoder.lr']}")
        self.logger.info(f"Learning rate : {self.opt.config['hparams']['generators.decoder.lr']}")

        self.logger.info("Parameters of Discriminators")
        count_parameters(self.disc_agent.discriminators)
        self.logger.info(f"Learning rate        : {self.opt.config['hparams']['discriminators.lr']}")

        self.logger.info(f"Number of GPU        : {torch.cuda.device_count()} GPUs")
        self.logger.info(f"Root checkpoint      : {self.opt.root_checkpoint}")
        self.logger.info(f"Saving network every : {self.opt.save_checkpoint} epochs")
        
        self.logger.info(f"Root summary")
        for _, root in self.opt.config['log_dir'].items():
            if isinstance(root,dict):
                for (_,subroot) in root.items():
                    self.logger.info(f"Summary{subroot}")
            else:
                self.logger.info(f"Summary:{root}")

    def on_training_epoch(self, epoch, bar):
        _bar = tq(enumerate(zip(self.data_trn_loader,self.lat_trn_loader)),
                    position=1,leave=False, desc='train.', 
                    total=len(self.data_trn_loader))

        for idx, (batch_data,batch_latent) in _bar:
            y, x, *others = batch_data
            zyy,zyx, *other = batch_latent
            y,x     = y.to(app.DEVICE, non_blocking = True), x.to(app.DEVICE, non_blocking = True)
            zyx,zyy = zyx.to(app.DEVICE,non_blocking=True),zyy.to(app.DEVICE,non_blocking=True)
            pack = patch(y=y,x=x,zyy=zyy,zyx=zyx)
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
                    position=1, leave=False, desc='valid.', 
                    total=len(self.data_vld_loader))
        for idx, (batch_data,batch_latent) in _bar:
            y, x, *others   = batch_data
            zyy,zyx, *other = batch_latent
            y,x     = y.to(app.DEVICE, non_blocking = True), x.to(app.DEVICE, non_blocking = True)
            zyx,zyy = zyx.to(app.DEVICE,non_blocking=True),zyy.to(app.DEVICE,non_blocking=True)
            pack = y,x,zyy,zyx
            self.train_discriminators(ncritics=1, batch=pack,epoch=epoch, 
                        modality='eval',net_mode=['eval','eval'])
            self.train_generators(ncritics=1, batch=pack, epoch=epoch, 
                        modality='eval',net_mode=['eval','eval'])
        if epoch%self.opt.config['hparams']['validation_epochs'] == 0:
            self.losses_disc_tracker.write( epoch=epoch, modality = ['train','eval'])
            self.losses_gen_tracker.write(  epoch=epoch, modality = ['train','eval'])
            self.prob_disc_tracker.write(epoch=epoch, modality = ['train','eval'])

    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        """ The UnicTrainer class is extended to support different strategy
            WGAN, WGAN-GP, ALICE-explicite, ALICE-implicite
        """
        raise NotImplementedError
    
    def train_generators(self,batch,epoch, modality, net_mode, *args, **kwargs):
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
                accuracy_hb = get_accuracy(tag='hybrid',plot_function=get_gofs,
                    encoder = self.gen_agent.Fxy,
                    decoder = self.gen_agent.Gy,
                    vld_loader = self.data_tst_loader,
                    pfx ="vld_set_bb_unique",opt= self.opt,
                    outf = self.opt.outf, save = False
                )
                self.validation_writer.add_scalar('Accuracy/Hybrid',accuracy_hb,epoch)
                bar.set_postfix(accuracy_hb =accuracy_hb)

                accuracy_bb = get_accuracy(tag='broadband',plot_function=get_gofs,
                    encoder = self.gen_agent.Fxy,
                    decoder = self.gen_agent.Gy,
                    vld_loader = self.data_tst_loader,
                    pfx ="vld_set_bb_unique",opt= self.opt,
                    outf = self.opt.outf, save = False
                )
                self.validation_writer.add_scalar('Accuracy/Broadband',accuracy_bb,epoch)
                bar.set_postfix(accuracy_bb=accuracy_bb)

                accuracy_fl = get_accuracy(tag='broadband',plot_function=get_gofs,
                    encoder = self.gen_agent.Fxy,
                    decoder = self.gen_agent.Gy,
                    vld_loader = self.data_tst_loader,
                    pfx ="vld_set_bb_unique_hack",opt= self.opt,
                    outf = self.opt.outf, save = False
                )
                self.validation_writer.add_scalar('Accuracy/Filtered',accuracy_fl,epoch)
                bar.set_postfix(accuracy_fl=accuracy_fl)

                # # get weight of encoder and decoder
                # bar.set_postfix(status='saving tracked gradient ...')
                # self.disc_agent.track_weight(self.validation_writer,epoch)
                # self.gen_agent.track_weight(self.validation_writer,epoch)

                #plot broadband reconstruction signal and gof
                figure_bb, gof_bb = plot_generate_classic(tag ='broadband',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.data_tst_loader, pfx="vld_set_bb_unique",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Broadband ...')
                self.validation_writer.add_figure('STFD Broadband', figure_bb)
                self.validation_writer.add_figure('GOF Broadband', gof_bb)

                # plot filtered reconstruction signal and gof
                figure_fl, gof_fl = plot_generate_classic(tag ='broadband',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.data_tst_loader, pfx="vld_set_bb_unique_hack",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Filtered ...')
                self.validation_writer.add_figure('STFD Filtered', figure_fl)
                self.validation_writer.add_figure('GOF Filtered', gof_fl)

                # plot hybrid filtred reconstruction signal and gof
                figure_hf, gof_hf = plot_generate_classic(tag ='hybrid',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.data_tst_loader, pfx="vld_set_bb_unique",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Hybrid Filtered ...')
                self.validation_writer.add_figure('STFD Hybrid Filtered', figure_hf)
                self.validation_writer.add_figure('GOF Hybrid Filtered', gof_hf)

                # plot hybrid broadband reconstruction signal and gof
                figure_hb, gof_hb = plot_generate_classic(tag ='hybrid',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.data_tst_loader, pfx="vld_set_bb_unique_hack",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Hybrid Broadband ...')
                self.validation_writer.add_figure('STFD Hybrid Broadband', figure_hb)
                self.validation_writer.add_figure('GOF Hybrid Broadband', gof_hb)
    