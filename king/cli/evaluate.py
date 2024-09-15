import os
import subprocess
import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
def register_subparsers(subparser):
    parser = subparser.add_parser('evaluate', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "videos_path",
        type=str,
        required=True,
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
        required=True,
        help="list of evaluation metrics, usage: --dimensions <metric_1> <metric_2>",
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
    for arg in args_dict:
        if arg == "ngpus" or (args_dict[arg] == None) or arg == "func":
            continue
        if arg == "videos_path":
            cmd.append(f"--videos_path=\"{str(args_dict[arg])}\"")
            continue
        cmd.append(f'--{arg}')
        cmd.append(str(args_dict[arg]))


    subprocess.run(stringify_cmd(cmd), shell=True)

