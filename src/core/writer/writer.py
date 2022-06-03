import importlib

class Writer(object):
    def __init__(self, log_dir, logger, enabeled=True):
        self.writer = None
        self.logger = logger
        if enabeled:
            log_dir = str(log_dir)
            self.selected_module = ""
            succeeded = False
            self.writer_functions = {
                'add_scalar','add_scalars','add_image','add_figure','add_images',
                'add_histogram','add_graph'
            }
            for module in ['torch.utils.tensorboard',"tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module=module

        if not succeeded:
            self.logger.warning(" Tensorboard is not installed")
    
    def set_step(self, step,mode='train'): 
        _mode =  [  'train', 'eval','debug', 'test', 'tuning',
                    'Dloss',  'Gloss', 'Ggradient','Dgradient','Probs']
        if mode in _mode:
            self.mode = mode
        else:
            self.logger.warning("mode doesn't exist in the training")
        
        if isinstance(step,int) and step>0:
            self.step = step

    def __getattr__(self,name):
        if name in self.writer_functions:
            add_data = getattr(self.writer,name, None)
            def wrapper(tag,data,*args, **kwargs):
                if add_data is not None:
                    tag = '{}/{}'.format(self.mode, tag)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object {} "\
                        "has no attribute '{}'".format(self.selected_module,name))
            return attr