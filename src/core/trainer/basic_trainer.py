from abc import abstractmethod
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
        bar = trange(self.start_epoch, self.config.niter +1)
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
        if epoch%self.save_checkpoint == 0:
            bar.set_postfix(satus = f'saving models at {epoch}...')
            for group_name, group_models in self.models.items():
                for model in group_models:
                    state = {
                        'epoch'                 :epoch,
                        'model_state_dict'      :model.module.state_dict(),
                        'optimizer_state_dict'  :group_models.optmizer.state_dict(),
                        'loss'                  :self.losses[group_name]
                    }
                    filename = str(self.root_checkpoint /'checkpoint-epoch{}-{}'.format(model.name, epoch))
                    torch.save(state, filename)
                    self.logger.info("saving checkpoint-epoch : {}".format(filename))

    def on_resuming_checkpoint(self):
        for model, path in zip(self.models, self.strategy) :
            self.logger.info("Loading checkpoint-epoch : {}".format(path))
            checkpoint = torch.load(path)
            model.module.load_state_dict(checkpoint['state_dict'])
            model.optimizer.load_state_dict(checkpoint['optmizer'])
        self.start_epoch = checkpoint['epoch']+1
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))