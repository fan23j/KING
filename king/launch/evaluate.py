import torch
import os
from king import KING
from king.distributed import dist_init, print0
from datetime import datetime
import argparse
import json
from king.misc import DEFAULT_DIMENSIONS as default_dimensions

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='KING', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        required=True,
        help="folder that contains the sampled videos",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs='+',
        required=True,
        choices=default_dimensions,
        default=default_dimensions,
        help=f'list of evaluation metrics, unless specified, default dimensions are: {default_dimensions}',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dist_init()
    print0(f'args: {args}')
    device = torch.device("cuda")
    my_King = KING(device, args.output_path)
    
    print0(f'start evaluation')

    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    kwargs = {}

    my_King.evaluate(
        videos_path = args.videos_path,
        name = f'results_{current_time}',
        dimension_list = args.dimensions,
        **kwargs
    )
    print0('done')


if __name__ == "__main__":
    main()
