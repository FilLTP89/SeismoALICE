import pandas as pd
import numpy as np

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
    def __init__(self, columns, writer=None, mode='train'):
        # On Initialisation, 
        # columns   :the columns name should be passed, 
        #            by default the "epochs" column is created.
        # writer    : this is a writer like Tensorboard.
        
        self.writer = writer
        self.mode   = mode
        self.columns= columns
        self.keys   = {columns_keys for columns_keys in columns.keys()}
        self._data  = pd.DataFrame(columns=self.keys, dtype=np.float32)
    
    def set(self, _values):
        #add values to columns
        self.columns= _values
    
    def get(self,column,epoch):
        #get a specific column and average value for a specified epochs 
        return self._data.loc[self._data['epochs'] == epoch][column].mean()

    def update(self):
        # this function should save values in a row
        self._data = self._data.append({key:value for key, value in self.columns.items()}, ignore_index=True)
        

    def write(self,epoch):
        # make the mean of the loss and pass it to the writer
        if self.writer is not None:
            for column in self.keys:
                if column != 'epochs':
                    value = self._data.loc[self._data['epochs'] == epoch][column].mean()
                    self.writer.set_step(epoch,self.mode)
                    self.writer.add_scalar(column, value, epoch)
    
    def save_metric(self,dir_name, filename):
        return self._data.to_excel(f'{dir_name}/{filename}',
                sheet_name=filename, index=True)

    def resume_metric(self,file_path):
        self._data = pd.read_excel(file_path)