import os

from data_selection import data_selector
from data_selection import data_selector_with_plot

import matlab.engine

# data for the accelerograms ##############################################################################################################
earth_data_root = 'D:\\Luca\\Database\\STEAD\\'
# 'csv_file' is the name (.csv) of the considered chunk of the STandford EArthquake Dataset (STEAD) (https://github.com/smousavi05/STEAD)
csv_file        = 'chunk2.csv'
csv_file        = earth_data_root + csv_file
# 'file_name' is the name (.hdf5) of the considered chunk of the STandford EArthquake Dataset (STEAD) (https://github.com/smousavi05/STEAD)
file_name       = 'chunk2.hdf5'
file_name       = earth_data_root + file_name
# end data for the accelerograms ##########################################################################################################

# si immagina di dare in input anche un set di parametri capaci di effettuare i sampling
acc_for_loading = data_selector(csv_file,file_name)

# data for the matlab routine #############################################################################################################
data_file_undamaged = 'telaio_GL_9_und'
data_file_damaged   = 'telaio_GL_9_dam'
# end data for the matlab routine #########################################################################################################

# data generator part #####################################################################################################################
# undamaged part
eng = matlab.engine.start_matlab()
eng.telaio_2_for_py(data_file_undamaged,acc_for_loading,time_for_loading,nargout=0)

# damaged part
eng.telaio_2_for_py(data_file_damage,acc_for_loading,time_for_loading,nargout=0)
# end data generator part #################################################################################################################