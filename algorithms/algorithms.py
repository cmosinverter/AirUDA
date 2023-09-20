import torch
import torch.nn as nn
import numpy as np
from blocks.blocks import Regressor


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.mean_squared_error = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.regressor = Regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            
            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model
    
    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError
    
class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, tgt_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:
            
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)
            src_reg_loss = self.mean_squared_error(src_pred, src_y)

            loss = src_reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_reg_loss': src_reg_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, self.hparams["batch_size"])
