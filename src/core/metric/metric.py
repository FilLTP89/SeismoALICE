import re
import pandas as pd

class MetricTracker:
    """
        This class has to manage the losses function. The pandas library 
        is used as a structure of data, which give a easy way to manage data
        The values will be passed as value to a table. 
        By doing this, this way we get an overview of the losses functions

        Some useful services that this object should provide 
        is to give is to:
             - stack values by epoches;
             - save in a excel file and load values from an excel file
        
        So the expected form of the table sould be : 

        index   |Dloss_ali  |Dloss_ali_x    |... | Gloss_ali_x |...|epochs|
                 ###           ###                   ###             ###
                 ###           ###                   ###             ###

    """
    def __init__(self, columns, writer=None):
        # On Initialisation, 
        # columns   :the columns name should be passed, 
        #            by default the "epochs" column is created.
        # writer    : this is a writer like Tensorboard.
        self.writer = writer
        self.index = 0
        self.columns = columns
        self._data = pd.DataFrame(colums=self.columns)
    
    def update(self):
        # this function should save values in a row
        for key, value in self.columns:
            self._data[self.index, key] = value
        self.index += 1 

    def write(self,epoch):
        # make the mean of the loss and pass it to the writer
        if self.writer is not None:
            for column, _ in self.columns:
                value = self._data[self._data.epochs == epoch][column].mean()
                self.writer.add_scalar(column, value, epoch)
    
    def save_metric(self,dir_name):
        return self._data.to_excel(f'{dir_name}/losses',
                sheet_name='losses',index=True)

    def resume_metric(self,file_path):
        self._data = pd.read_excel(file_path)
        self.index = self._data.index[-1]