from utils_data_preprocessing import *

import m02_Data_Files.d06_SDF_Ready

folder_path = os.path.dirname(m02_Data_Files.d06_SDF_Ready.__file__)
file = os.path.join(folder_path, "idx_int2str_dict.npy")
print(file)
npy_read_out(file)