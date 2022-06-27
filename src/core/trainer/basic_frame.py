from common.common_model import track_gradient_change

class Agent:
    """ 
        This class define the frame of the tranining that is the template
        for the use of GAN, ALI, ALICE and classifier used in the future of the 
        programme
    """
    def __init__(self, models, optimizer, config, logger, accelerator,*args, **kwargs):
        self.config = config
        self.logger = logger
        self.models = models
        self.optimizer = optimizer
        self.accelerator  = accelerator
        self.current_val = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        # the class is able to give is child model in a loop
        if self.current_val>=len(self.models):
            self.current_val=0
            raise StopIteration
        _model = self.models[self.current_val]
        self.current_val +=1
        return _model, self.optimizer
    
    def track_gradient_change(self,gradient_values,nets,epoch,*args, **kwargs):
        _values = {'epochs':epoch,'modality':'train'}
        _values.update({net.module.model_name : track_gradient_change(net) for net in nets})
        gradient_values.set(_values)
        gradient_values.update()
    
    def track_weight_change(self, writer, tag, model,epoch):
        for idx in range(len(model)):
            classname = model[idx].__class__.__name__
            if (classname.find('Conv1d')!= -1 
                    or 
                classname.find('Linear')!= -1
                    or
                classname.find('ConvTranspose1d')!= -1) and writer != None:
                writer.set_step(mode='debug',step=epoch)
                writer.add_histogram(f'{tag}/{idx}', model[idx].weight, epoch)
            else:
                self.logger.debug("weights are not tracked ... ")
        
    def track_gradient(self,*args,**kwargs):
        raise NotImplementedError
    
    def track_weight(self, *args, **kwargs):
        raise NotImplementedError