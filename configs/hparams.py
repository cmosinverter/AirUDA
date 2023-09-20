## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class Airbox():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 64,
        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-4, 'src_reg_loss_wt': 1},
            'TARGET_ONLY': {'learning_rate': 1e-4, 'trg_reg_loss_wt': 1},
        }