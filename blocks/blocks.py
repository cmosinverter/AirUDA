import torch
import torch.nn as nn

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################


class GRU(nn.Module):
    def __init__(self, configs):
        super(GRU, self).__init__()
        self.gru = nn.GRU(configs.input_channels, configs.final_out_channels, configs.lstm_n_layers, batch_first=True)
        
    def forward(self, x_in):
        out = x_in.permute(0, 2, 1)
        out, _ = self.gru(out)
        out = out[:, -1, :]
        return out


##################################################
###############  Regressor  ######################
##################################################

class Regressor(nn.Module):
    def __init__(self, configs):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(configs.features_len * configs.final_out_channels, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
    
        return out