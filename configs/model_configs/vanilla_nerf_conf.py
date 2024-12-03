CONFIG = {
    "i_embed": 0,           # 'set 0 for default positional encoding, -1 for none'
    "multires": 10,         # 'log2 of max freq for positional encoding'
    "multires_views": 4,    # 'log2 of max freq for positional encoding for direction'
    "netdepth": 8,          # 'layers in network'
    "netwidth": 256,        # 'channels per layer'
    "perturb": 1.,          # 'set to 0. for no jitter, 1. for jitter'
    "chunk": 1024*32,       # 'number of rays processed in parallel, decrease if running out of memory'
    "lrate": 5e-4,          
    "lrate_decay": 250,
    "raw_noise_std": 0.,    # 'std dev of noise added to regularize sigma_a output, 1e0 recommended'
    "netchunk": 1024*64,    # 'number of pts sent through network in parallel, decrease if running out of memory'
    "N_rand": 32*32*4,      # 'batch size (number of random rays per gradient step)'
    "N_samples": 128,       # 'number of coarse samples per ray'
    "white_bkgd": True,
    "device": "cuda"
}
