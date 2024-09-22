import os
import subprocess
import argparse
from king.misc import DEFAULT_DIMENSIONS as default_dimensions
import shlex

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def register_subparsers(subparser):
    parser = subparser.add_parser('evaluate', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "videos_path",
        type=str,
        help="folder that contains the generated samples",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Number of GPUs to run evaluation on"
        )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs='+',
        choices=default_dimensions,
        default=default_dimensions,
        help=f'list of evaluation metrics, unless specified, default dimensions are: {default_dimensions}',
    )
    parser.set_defaults(func=evaluate)

def stringify_cmd(cmd_ls):
    cmd = ""
    for string in cmd_ls:
        cmd += string + " "
    return cmd

def evaluate(args):
    cmd = ['python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(args.ngpus), f'{CUR_DIR}/../launch/evaluate.py']
    args_dict = vars(args)
    for arg,value in args_dict.items():
        if arg in ("ngpus", "func") or value is None:
            continue
        if arg == "videos_path":
            cmd.append(f"--videos_path={shlex.quote(str(value))}")
        elif arg == "dimensions":
            cmd.extend(["--dimensions"] + [shlex.quote(dim) for dim in value])
        else:
            cmd.append(f'--{arg}')
            cmd.append(shlex.quote(str(value)))


    subprocess.run(stringify_cmd(cmd), shell=True)

