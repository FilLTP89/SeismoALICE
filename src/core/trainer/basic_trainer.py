from abc import abstractmethod
from tqdm import tqdm, trange
import torch
class BasicTrainer:
    """
    BasicTrainer for all trainer
    """
    def __init__(self, config, logger, models,optimizer, losses,strategy, 
                    actions=None,*args, **kwargs):
        super(BasicTrainer, self).__init__()
        self.config     = config 
        self.logger     = logger
        self.models     = models
        self.strategy   = strategy
        self.actions    = actions
        self.losses     = losses
        self.optimizer  = optimizer

        self.root_checkpoint = self.config.root_checkpoint
        self.save_checkpoint = self.config.save_checkpoint
        
        self.start_epoch = 1
        if self.actions is not None: 
            self.on_resuming_checkpoint()
        else:
            self.logger.info("No resumed models")

    @abstractmethod
    def on_training_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def train(self):
        for epoch in trange(self.start_epoch, self.config.niter +1):
            self.on_training_epoch(epoch)
            self.on_validation_epoch(epoch)
            self.on_saving_checkpoint(epoch)

    @abstractmethod
    def on_validation_epoch(self, epoch):
        """ Validation of the training network
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def on_saving_checkpoint(self, epoch):
        if epoch%self.save_checkpoint == 0:
            breakpoint()
            self.logger.info('saving models ...')
            for model in self.models:
                state = {
                    'epoch'                 :epoch,
                    'model_state_dict'      :model.state_dict(),
                    'optimizer_state_dict'  :self.optmizer.state_dict(),
                    'loss'                  :self.losses
                }

                filename = str(self.root_checkpoint /'checkpoint-epoch{}-{}'.format(model.name, epoch))
                torch.save(state, filename)
                self.logger.info("saving checkpoint-epoch : {}".format(filename))

    def on_resuming_checkpoint(self):
        for model, path in zip(self.models, self.strategy) :
            self.logger.info("Loading checkpoint-epoch : {}".format(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optmizer'])
        self.start_epoch = checkpoint['epoch']+1
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))