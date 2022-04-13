import re
import pandas as pd

class MetricTracker:
    def __init__(self, columns, writer=None):
        self.writer = writer
        self.index = 0
        columns.appends('epochs')
        self._data = pd.DataFrame(colums=columns)
    
    def update(self,keys, values):
        for key, value in zip(keys, values):
            self._data[self.index, key] = value
        self.index += 1 

    def write(self,key,epoch):
        if self.writer is not None:
            value = self._data[self._data.epochs == epoch][key].mean()
            self.writer.add_scalar(key, value, epoch)
    
    def save_metric(self,dir_name):
        return self._data.to_excel(f'{dir_name}/losses',
                sheet_name='losses',index=True)

    def resume_metric(self,file_path):
        self._data = pd.read_excel(file_path)
        self.index = self._data.index[-1]