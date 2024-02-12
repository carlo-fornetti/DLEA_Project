from argparse import ArgumentParser

# define all the parser to execute with custom parameters and custom paths
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--b', type=int, default=4, help='input batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--save-weights', default='./res/weights', help='directory for saving checkpoint models (default: ./weights)')
    return parser.parse_args()