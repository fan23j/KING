import torch
import os
from vbench import VBench
from vbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json

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
        "--metric",
        nargs='+',
        required=True,
        help="list of evaluation metrics, usage: --metric <metric_1> <metric_2>",
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
        metrics_list = args.metric,
        **kwargs
    )
    print0('done')


if __name__ == "__main__":
    main()
