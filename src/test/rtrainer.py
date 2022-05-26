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
from test.generators import Generators
from test.discriminators import Discriminators

class ALICE(BasicTrainer):
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
            'Dloss_ali_x':'',      'Dloss_marginal':'',   'Dloss_marginal_y':'',
            'Dloss_marginal_zd':'','Dloss_marginal_x':'', 'Dloss_marginal_zf':''
        }
        self.losses_gens = {
            'epochs':'',           'modality':'',
            'Gloss':'',            'Gloss_ali':'',        'Gloss_ali_x':'',
            'Gloss_ali_y':'',      'Gloss_marginal':'',   'Gloss_marginal_y':'',
            'Gloss_marginal_zd':'','Gloss_marginal_x':'', 'Gloss_marginal_zf':'',

            'Gloss_rec':'',        'Gloss_rec_y':'',      'Gloss_rec_x':'',
            'Gloss_rec_zd':'',     'Gloss_rec_zx':'',     'Gloss_rec_zxy':'',
            'Gloss_rec_x':'', 
        }

        self.gradients_gens = {
            'epochs':'',    'modality':'',
            'Fxy':'',       'Gy':'',
        }
        self.gradients_disc = {
            'epochs':'',    'modality':'',
            'Dy':'',   'Dx':'',   'Dsy':'',  'Dsx':'',
            'Dzb':'',  'Dszb':'', 'Dyz':'',  'Dzf':'',
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

        super(ALICE,self).__init__(
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
            y, x, *others = batch
            y   = y.to(app.DEVICE, non_blocking = True)
            x   = x.to(app.DEVICE, non_blocking = True)
            zyy,zyx, *other = generate_latent_variable(batch=len(y))
            pack = y,x,zyy,zyx
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
            y, x, *others = batch
            y   = y.to(app.DEVICE, non_blocking = True)
            x   = x.to(app.DEVICE, non_blocking = True)
            zyy,zyx, *other = generate_latent_variable(batch=len(y))
            pack = y,x,zyy,zyx
            self.train_discriminators(ncritics=1, batch=pack,epoch=epoch, 
                        modality='eval',net_mode=['eval','eval'])
            self.train_generators(ncritics=1, batch=pack, epoch=epoch, 
                        modality='eval',net_mode=['eval','eval'])
        if epoch%self.opt.config['hparams']['validation_epochs'] == 0:
            self.losses_disc_tracker.write( epoch=epoch, modality = ['train','eval'])
            self.losses_gen_tracker.write(  epoch=epoch, modality = ['train','eval'])

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
                    vld_loader = self.test_loader,
                    pfx ="vld_set_bb_unique",opt= self.opt,
                    outf = self.opt.outf, save = False
                )
                self.validation_writer.add_scalar('Accuracy/Hybrid',accuracy_hb,epoch)
                bar.set_postfix(accuracy_hb =accuracy_hb)

                accuracy_bb = get_accuracy(tag='broadband',plot_function=get_gofs,
                    encoder = self.gen_agent.Fxy,
                    decoder = self.gen_agent.Gy,
                    vld_loader = self.test_loader,
                    pfx ="vld_set_bb_unique",opt= self.opt,
                    outf = self.opt.outf, save = False
                )
                self.validation_writer.add_scalar('Accuracy/Broadband',accuracy_bb,epoch)
                bar.set_postfix(accuracy_bb=accuracy_bb)

                accuracy_fl = get_accuracy(tag='broadband',plot_function=get_gofs,
                    encoder = self.gen_agent.Fxy,
                    decoder = self.gen_agent.Gy,
                    vld_loader = self.test_loader,
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
                        trn_set=self.test_loader, pfx="vld_set_bb_unique",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Broadband ...')
                self.validation_writer.add_figure('STFD Broadband', figure_bb)
                self.validation_writer.add_figure('GOF Broadband', gof_bb)

                # plot filtered reconstruction signal and gof
                figure_fl, gof_fl = plot_generate_classic(tag ='broadband',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.test_loader, pfx="vld_set_bb_unique_hack",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Filtered ...')
                self.validation_writer.add_figure('STFD Filtered', figure_fl)
                self.validation_writer.add_figure('GOF Filtered', gof_fl)

                # plot hybrid filtred reconstruction signal and gof
                figure_hf, gof_hf = plot_generate_classic(tag ='hybrid',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.test_loader, pfx="vld_set_bb_unique",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Hybrid Filtered ...')
                self.validation_writer.add_figure('STFD Hybrid Filtered', figure_hf)
                self.validation_writer.add_figure('GOF Hybrid Filtered', gof_hf)

                # plot hybrid broadband reconstruction signal and gof
                figure_hb, gof_hb = plot_generate_classic(tag ='hybrid',
                        Qec= self.gen_agent.Fxy, Pdc= self.gen_agent.Gy,
                        trn_set=self.test_loader, pfx="vld_set_bb_unique_hack",
                        opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Hybrid Broadband ...')
                self.validation_writer.add_figure('STFD Hybrid Broadband', figure_hb)
                self.validation_writer.add_figure('GOF Hybrid Broadband', gof_hb)
    
    def train_discriminators(self,ncritics,batch,epoch,modality,net_mode,*args,**kwargs):
        y,x,zyy,zxy = batch
        for _ in range(ncritics):
            zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])
            
            # 1.1 We Generate conditional samples
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            zd_inp      = zcat(zxy,zyy)
            y_inp       = zcat(y,wny) 
            y_gen       = self.gen_agent.Gy(zxy,zyy)
            zyy_F,zyx_F,*other = self.gen_agent.Fxy(y_inp)
            zd_gen      = zcat(zyx_F,zyy_F)

            # 1.2 Let's match the proper joint distributions
            Dreal_yz,Dfake_yz = self.disc_agent.discriminate_yz(y,y_gen,zd_inp,zd_gen)
            Dloss_ali_y = self.bce_loss(Dreal_yz,o1l(Dreal_yz))+\
                            self.bce_loss(Dfake_yz,o0l(Dfake_yz))

            # 1.3. We comput the marginal probability distributions
            Dreal_y,Dfake_y  = self.disc_agent.discriminate_marginal_y(y,y_gen)
            Dloss_marginal_y = self.bce_loss(Dreal_y,o1l(Dreal_y))+\
                                self.bce_loss(Dfake_y,o0l(Dfake_y))
            # And also, we do the evaluation the marginal probabiliti distribution
            Dreal_zd,Dfake_zd= self.disc_agent.discriminate_marginal_zd(zd_inp,zd_gen)
            Dloss_marginal_zd= self.bce_loss(Dreal_zd,o1l(Dreal_zd))+\
                                self.bce_loss(Dfake_zd,o0l(Dfake_zd))

            # Part II.- Training the Filtered signal
            # 1.1 Let's compute the Generate samples
            wnx,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            x_inp       = zcat(x,wnx)
            _x_gen      = self.gen_agent.Gy(zxy,o0l(zyy))
            _, zxy_gen, *others = self.gen_agent.Fxy(x_inp)

            # 1.2 Now, we match the probability distribution of (x,F|(x)_zxy) ~ (G(zxy,0), zxy)
            Dreal_xz,Dfake_xz    = self.disc_agent.discriminate_xz(x,_x_gen,zxy,zxy_gen)
            Dloss_ali_x          = self.bce_loss(Dreal_xz,o1l(Dreal_xz))+\
                                    self.bce_loss(Dfake_xz,o0l(Dfake_xz)) 

            # 1.3 It is important to evaluate the marginal probability distribution. 
            Dreal_x,Dfake_x      = self.disc_agent.discriminate_marginal_x(x,_x_gen)
            Dloss_marginal_x     = self.bce_loss(Dreal_x,o1l(Dreal_x))+\
                                    self.bce_loss(Dfake_x,o0l(Dfake_x))
            # 1.4 For zxy, we should satisfy this equation : 
            Dreal_zf,Dfake_zf   = self.disc_agent.discriminate_marginal_zxy(zxy,zxy_gen)
            Dloss_marginal_zf   = self.bce_loss(Dreal_zf,o1l(Dreal_zf))+\
                                    self.bce_loss(Dfake_x,o0l(Dfake_zf))
            # Marginal losses
            Dloss_marginal      = (Dloss_marginal_y + Dloss_marginal_zd + Dloss_marginal_x + Dloss_marginal_zf)
            # ALI losses
            Dloss_ali           = (Dloss_ali_y + Dloss_ali_x)
            # Total losses
            Dloss               = Dloss_ali + Dloss_marginal
            
           
            if modality == 'train':
                Dloss.backward()
                self.disc_agent.optimizer.step()
                self.disc_agent.track_gradient(epoch)
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])

            self.losses_disc['epochs'       ] = epoch
            self.losses_disc['modality'     ] = modality
            self.losses_disc['Dloss'        ] = Dloss.tolist()
            self.losses_disc['Dloss_ali'    ] = Dloss_ali.tolist()
            self.losses_disc['Dloss_ali_y'  ] = Dloss_ali_y.tolist()
            self.losses_disc['Dloss_ali_x'  ] = Dloss_ali_x.tolist()

            self.losses_disc['Dloss_marginal'    ] = Dloss_marginal.tolist()
            self.losses_disc['Dloss_marginal_y'  ] = Dloss_marginal_y.tolist()
            self.losses_disc['Dloss_marginal_zd' ] = Dloss_marginal_zd.tolist()
            self.losses_disc['Dloss_marginal_x'  ] = Dloss_marginal_x.tolist()
            self.losses_disc['Dloss_marginal_zf' ] = Dloss_marginal_zf.tolist()
            
            self.losses_disc_tracker.update()

    def train_generators(self,ncritics,batch,epoch, modality, net_mode, *args, **kwargs):
        y,x,zyy,zxy = batch
        for _ in range(ncritics):
            zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode =net_mode[0])
            modalite(self.disc_agent.discriminators,  mode =net_mode[1])
            
            # As we said before the Goal of this function is to compute the loss of Y and the 
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            y_inp   = zcat(y,wny)
            zd_inp  = zcat(zxy,zyy)
            y_gen   = self.gen_agent.Gy(zxy,zyy)
            zyy_gen,zyx_gen,*others = self.gen_agent.Fxy(y_inp)
            zd_gen= zcat(zyx_gen,zyy_gen)

            # So, let's evaluate the loss of ALI 
            Dreal_yz,Dfake_yz = self.disc_agent.discriminate_yz(y,y_gen,zd_inp,zd_gen)  
            Gloss_ali_y =  self.bce_loss(Dreal_yz,o0l(Dreal_yz))+\
                            self.bce_loss(Dfake_yz,o1l(Dfake_yz))

            _ , Dfake_y = self.disc_agent.discriminate_marginal_y(y,y_gen)
            Gloss_marginal_y = (self.bce_loss(Dfake_y,o1l(Dfake_y)))
            # The marginal loss on zd is as follow : 
            #       min (E[log(1 - Dzd(F(x)])
            _, Dfake_zd = self.disc_agent.discriminate_marginal_zd(zd_inp,zd_gen)
            Gloss_marginal_zd= (self.bce_loss(Dfake_zd,o1l(Dfake_zd)))

            # 2. Let's generate the reconstructions, i.e G(F(y)) and F(G(z))
            # So, we pepare our input for the training, adding noise on broadband, and concatenation
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            # Then we generate reconstructions ...
            y_rec = self.gen_agent.Gy(zyx_gen,zyy_gen)
            y_gen = zcat(y_gen,wny)
            zyy_rec,zyx_rec,*other = self.gen_agent.Fxy(y_gen)
            zd_rec  = zcat(zyx_rec,zyy_rec)
        
            Gloss_rec_y     = torch.mean(torch.abs(y-y_rec))
            Gloss_rec_zd    = torch.mean(torch.abs(zd_inp-zd_rec))

            # Part II.- Training the Filtered signals
            wnx,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})   
            x_inp       = zcat(x,wnx)
            zx_gen, zxy_gen, *others = self.gen_agent.Fxy(x_inp)
            _x_gen      = self.gen_agent.Gy(zxy,o0l(zyy))
            # We are able to match joint probability distribution and compute losses of marginal
            Dreal_xz,Dfake_xz     = self.disc_agent.discriminate_xz(x,_x_gen,zxy,zxy_gen)
            Gloss_ali_x = self.bce_loss(Dreal_xz,o0l(Dreal_xz))+\
                            self.bce_loss(Dfake_xz,o1l(Dfake_xz)) 
            
            _ , Dfake_x = self.disc_agent.discriminate_marginal_x(x,_x_gen)
            Gloss_marginal_x = (self.bce_loss(Dfake_x,o1l(Dfake_x)))
            _ ,Dfake_zf = self.disc_agent.discriminate_marginal_zxy(zxy, zxy_gen)
            Gloss_marginal_zf= (self.bce_loss(Dfake_zf,o1l(Dfake_zf)))

            # 2. This second time we generate the reconstuction G(F|(x)_zxy,0) and F|(G(x))_zxy
            x_rec       = self.gen_agent.Gy(zxy_gen, o0l(zx_gen))
            x_gen       = zcat(_x_gen,wnx)
            zxx_rec, zxy_rec, *others = self.gen_agent.Fxy(x_gen)
            
            Gloss_rec_zxy = torch.mean(torch.abs(zxy - zxy_rec))
            Gloss_rec_x   = torch.mean(torch.abs(x - x_rec))
            Gloss_rec_zx  = torch.mean(torch.abs(zxx_rec))
            
            # 8. Total Loss
            Gloss_marginal  = Gloss_marginal_y+ Gloss_marginal_zd+ Gloss_marginal_x+Gloss_marginal_zf
            Gloss_rec   = Gloss_rec_y +Gloss_rec_zd + Gloss_rec_zx + Gloss_rec_x +Gloss_rec_zxy
            Gloss_ali   = Gloss_ali_y+ Gloss_ali_x
            Gloss       = Gloss_ali+ Gloss_marginal+ 0.1*Gloss_rec

            if modality == 'train':
                Gloss.backward()
                self.gen_agent.optimizer.step()
                self.gen_agent.track_gradient(epoch)
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])

            self.losses_gens['epochs'     ] = epoch
            self.losses_gens['modality'   ] = modality
            self.losses_gens['Gloss'      ] = Gloss.tolist()
            self.losses_gens['Gloss_ali'  ] = Gloss_ali.tolist()
            self.losses_gens['Gloss_ali_x'] = Gloss_ali_x.tolist()
            self.losses_gens['Gloss_ali_y'] = Gloss_ali_y.tolist()

            self.losses_gens['Gloss_marginal'    ] = Gloss_marginal.tolist()
            self.losses_gens['Gloss_marginal_y'  ] = Gloss_marginal_y.tolist()
            self.losses_gens['Gloss_marginal_zd' ] = Gloss_marginal_zd.tolist()
            self.losses_gens['Gloss_marginal_x'  ] = Gloss_marginal_x.tolist()
            self.losses_gens['Gloss_marginal_zf' ] = Gloss_marginal_zf.tolist()

            self.losses_gens['Gloss_rec'    ] = Gloss_rec.tolist()
            self.losses_gens['Gloss_rec_zd' ] = Gloss_rec_zd.tolist()
            self.losses_gens['Gloss_rec_y'  ] = Gloss_rec_y.tolist()
            self.losses_gens['Gloss_rec_zx' ] = Gloss_rec_zx.tolist()
            self.losses_gens['Gloss_rec_x'  ] = Gloss_rec_x.tolist()
        
            self.losses_gens['Gloss_rec_zxy'] = Gloss_rec_zxy.tolist()
            self.losses_gen_tracker.update()
    