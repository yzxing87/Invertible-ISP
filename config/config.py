import argparse

BATCH_SIZE = 1

DATA_PATH = "./data/"



def get_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    
    parser.add_argument("--task", type=str, help="Name of this training")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Dataset root path.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training. ")       
    parser.add_argument("--debug_mode", dest='debug_mode', action='store_true',  help="If debug mode, load less data.")    
    parser.add_argument("--gamma", dest='gamma', action='store_true', help="Use gamma compression for raw data.")     
    parser.add_argument("--camera", type=str, default="NIKON_D700", choices=["NIKON_D700", "Canon_EOS_5D"], help="Choose which camera to use. ")    
    parser.add_argument("--rgb_weight", type=float, default=1, help="Weight for rgb loss. ")                 
    
    
    return parser
