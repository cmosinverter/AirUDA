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
            'num_epochs': 800,
            'batch_size': 128,
        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-4, 'src_reg_loss_wt': 1},
            'TARGET_ONLY': {'learning_rate': 1e-4, 'trg_reg_loss_wt': 1},
            "DANN": {
                "domain_loss_wt": 1.0296390274908802,
                "learning_rate": 1e-4,
                "src_reg_loss_wt": 4.038458138479581,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "cond_ent_wt": 1.329734510542011,
                "domain_loss_wt": 6.632293308809388,
                "learning_rate": 1e-4,
                "src_reg_loss_wt": 7.729881324550688,
                "vat_loss_wt": 6.912258476982827,
                "weight_decay": 0.0001
            },
            "MMDA": {
                "cond_ent_wt": 0.01,
                "coral_wt": 100,
                "learning_rate": 1e-4,
                "mmd_wt": 0.01,
                "src_reg_loss_wt": 1,
                "weight_decay": 0.0001
            },
            "AirDA": {
                "learning_rate": 1e-4,
                "mmd_wt": 0.01,
                "src_reg_loss_wt": 1,
                "weight_decay": 0.0001
            },
        }