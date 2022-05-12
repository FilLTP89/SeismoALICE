from common.common_model import track_gradient_change

class Agent:
    """ 
        This class define the frame of the tranining that is the template
        for the use of GAN, ALI, ALICE and classifier used in the future of the 
        programme
    """
    def __init__(self, config, logger, accelerator,*args, **kwargs):
        self.config = config
        self.logger = logger
        self.accelerator  = accelerator
    
    def track_gradient_change(self,gradient_values,nets):
        _values = {net.module.model_name : track_gradient_change(net) for net in nets}
        gradient_values.set(_values)
        gradient_values.update()
    
    def track_weight_change(self, writer, tag, model,epoch):
        for idx in range(len(model)):
            classname = model[idx].__class__.__name__
            if (classname.find('Conv1d')!= -1 
                    or 
                classname.find('ConvTranspose1d')!= -1) and writer != None:
                writer.add_histogram(f'{tag}/{idx}', model[idx].weight, epoch)
            else:
                self.logger.info("weights are not tracked ... ")
        
    def track_gradient(self,*args,**kwargs):
        raise NotImplementedError
    
    def track_weight(self, *args, **kwargs):
        raise NotImplementedError