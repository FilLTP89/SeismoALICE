class BasicModule(object):
    """This class contain de base to create any modules 
        : Generators (Encoder, Decoder), Discriminators
    """
    def __init__(self, models_names=None, *args, **kwargs):
        super(BasicModule, self).__init__()
        self.models_names   = models_names

    
    def initialize(self, models, *args,**kwargs):        
        self.init_func(models, *args, **kwargs)
    

