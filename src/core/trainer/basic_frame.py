import BasicTrainer


class Frame(BasicTrainer):
    """ 
        This class define the frame of the tranining that is the template
        for the use of GAN, ALI, ALICE and classifier used in the future of the 
        programme
    """
    def __init__(self,config, logger, models, strategy, actions, modules, accelerator):
        super(BasicTrainer).__init__(self, config, logger, models, strategy, actions)
        pass

        