from abc import abstractmethod
from tqdm import tqdm, trange
import torch
class BasicTrainer:
    """
    BasicTrainer for all trainer
    """
    def __init__(self, config, logger, models, strategy, actions):
        super(BasicTrainer, self).__init__()
        self.config     = config 
        self.logger     = logger
        self.models     = models
        self.strategy   = strategy
        self.actions    = actions
        self.losses     = []
        self.optimizer  = []

        self.checkpoint_dir = self.config.checkpoint_dir
        self.save_epoch     = self.config.save_epoch
        
        self.start_epoch = 1
        if not None in self.actions: 
            self._resume_checkpoint()
        else:
            self.logger.info("No resumed models")

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def train(self):
        for epoch in trange(self.start_epoch, self.epochs +1):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            self._save_checkpoint(epoch)

    @abstractmethod
    def _validate_epoch(self, epoch):
        """ Validation of the training network
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch):
        if epoch%self.save_epoch == 0:
            self.logger.info('saving models ...')
            for model in self.models:
                state = {
                    'epoch'                 :epoch,
                    'model_state_dict'      :model.state_dict(),
                    'optimizer_state_dict'  :self.optmizer.state_dict(),
                    'loss'                  :self.losses
                }

                filename = str(self.checkpoint_dir /'checkpoint-epoch{}-{}'.format(model.name, epoch))
                torch.save(state, filename)
                self.logger.info("saving checkpoint-epoch : {}".format(filename))

    def _resume_checkpoint(self):
        for model, path in zip(self.models, self.strategy) :
            self.logger.info("Loading checkpoint-epoch : {}".format(path))
            checkpoint = torch.load(path)
            self.start_epoch = checkpoint['epoch']+1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optmizer'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))