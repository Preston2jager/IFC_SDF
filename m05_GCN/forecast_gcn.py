import os
import torch

from gcn_runner import GCN_Runner

import m02_Data_Files.d08_Forecast_Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    data_folder = os.path.dirname(m02_Data_Files.d08_Forecast_Data.__file__)
        
    Runner = GCN_Runner()
    result = Runner.forecast() 
