"""
Construct simgle shape for 2D reconstruction

Data format:
{
    "train":
    {   
        "poses": [2, N, 2]
        "ref_data": [N, resolution]
        "i_train": list of index
        "i_test": list of index
    }
    "test: ...
}
"""
from .triangle_dset import TriangleDset
datasets = {
    "triangle": TriangleDset
}
def get_dset(config):
    dset_name = config["shape_config"]["shape"]
    params = config["shape_config"]["params"]
    return datasets[dset_name](params, config)