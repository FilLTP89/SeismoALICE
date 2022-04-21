

class Actors:
    """ 
        This class define the frame of the tranining that is the template
        for the use of GAN, ALI, ALICE and classifier used in the future of the 
        programme
    """
    def __init__(self, model, config, logger, accelerator,*args, **kwargs):
        self.config = config
        self.logger = logger
        self.accelerator  = accelerator
        