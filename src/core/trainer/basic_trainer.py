from abc import abstractmethod
from common.common_nn import dir_setup
from tqdm import tqdm, trange
import torch
class BasicTrainer:
    """
    BasicTrainer for all trainer
    """
    def __init__(self, config, logger, models, losses,strategy, 
                    actions=None,*args, **kwargs):
        super(BasicTrainer, self).__init__()
        self.config     = config 
        self.logger     = logger
        self.models     = models
        self.strategy   = strategy
        self.actions    = actions
        self.losses     = losses

        self.root_checkpoint = self.config.root_checkpoint
        self.save_checkpoint = self.config.save_checkpoint
        
        self.start_epoch = 1
        if self.actions is not None: 
            self.on_resuming_checkpoint()
        else:
            self.logger.info("No resumed models")

    @abstractmethod
    def on_training_epoch(self, epoch, *args, **kwargs):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def train(self):
        bar = trange(self.start_epoch, self.config.niter +1,position=0,desc='epochs')
        for epoch in bar:
            self.on_training_epoch(epoch,bar)
            self.on_validation_epoch(epoch,bar)
            self.on_test_epoch(epoch, bar)
            self.on_saving_checkpoint(epoch, bar)

    @abstractmethod
    def on_test_epoch(self, epoch,*args, **kwargs):
        """ test of the trained network
            :param epoch: Current epoch number
            Printing figures and histogram every specified epoch
        """
        raise NotImplementedError
    
    @abstractmethod
    def on_validation_epoch(self,epoch,*args, **kwargs):
        """ validation of the trained network
            Do the same as training epoch, but every thing are in eval models
            required grad of the network, should be at False
        """
        raise NotImplementedError


    def on_saving_checkpoint(self, epoch, bar,*args, **kwargs):
        dir_setup(self.root_checkpoint)
        if epoch%self.save_checkpoint == 0:
            bar.set_postfix(satus = f'saving models at {epoch}...')
            for group_name, group_models in self.models.items():
                bar.set_postfix(satus = f'saving models for group {group_name}')
                for model, optimizer in group_models:
                    state = {
                        'epoch'                 :epoch,
                        'model_state_dict'      :model.module.state_dict(),
                        'optimizer_state_dict'  :optimizer.state_dict()
                    }
                    filename = f'{self.root_checkpoint}checkpoint-{model.module.model_name}_epoch-{epoch}.pth'
                    torch.save(state, filename)
                    self.logger.debug("saving checkpoint-epoch : {}".format(filename))

    def on_resuming_checkpoint(self, epoch, bar, *args, **kwargs):
        bar.set_postfix(satus =f'loading models from {self.root_checkpoint} from {epoch}...')
        for group_name, group_models in self.models.items():
            bar.set_postfix(satus =f'loading models of {group_name}')
            for model, optimizer in group_models:
                filename = f'{self.root_checkpoint}checkpoint-{model.module.model_name}_epoch-{epoch}.pth'
                self.logger.info("Loading checkpoint-epoch : {}".format(filename))
                checkpoint = torch.load(filename)
                model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.start_epoch = checkpoint['epoch']+1
        self.logger.debug("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))