def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class Airbox(object):
    def __init__(self):
        super(Airbox, self).__init__()
        self.sequence_len = 24
        self.scenarios = [("0", "2", "3"), ("0", "3", "4"), ("0", "4", "5"), ("0", "5", "6"), ("0", "6", "7"), ("0", "7", "8"), ("0", "8", "9")] 

        # Model configs
        self.input_channels = 6
        self.kernel_size = 4
        self.dropout = 0.1