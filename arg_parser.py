import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--expname", type=str, default='preliminary',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--config", type=str, default='./configs/setting1.json', 
                        help='where to load config of cameras and center shape')
    parser.add_argument("--model_name", type=str, default='vanilla_nerf', 
                        help='model names defined in models/__init__.py')

    parser.add_argument("--training_iters", type=int, default=1001) 
    parser.add_argument("--print_every", type=int, default=200) 
    args = parser.parse_args()
    return args