import argparse
import importlib
import subprocess

king_cmd = ['evaluate']

def main():
    parser = argparse.ArgumentParser(prog="king", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='king subcommands')

    for cmd in king_cmd:
        module = importlib.import_module(f'king.cli.{cmd}')
        module.register_subparsers(subparsers)
    parser.set_defaults(func=help)
    args = parser.parse_args()
    args.func(args)

def help(args):
    subprocess.run(['king', '-h'], check=True)
