class BasicModule(object):
    """This class contain de base to create any modules 
        : Generators (Encoder, Decoder), Discriminators
    """
    def __init__(self, models_names, init_func):
        super(BasicModule, self).__init__()
        self.models_names   = models_names
        self.init_func      = init_func

    
    def initialize(self, models, *args,**kwargs):        
        self.init_func(modes, *args, **kwargs)
    

