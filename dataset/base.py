

class BaseDataset:
    def __init__(self, params, full_config):
        self.params = params
        self.full_config = full_config
    
    def construct_shape(self):
        raise NotImplemented 

    def construct_poses(self):
        raise NotImplemented

    def gen(self):
        raise NotImplemented