#
#
#


import sys
import argparse

from experiments import registry


def main():
    """
    Run experiments.
    """
    parser = argparse.ArgumentParser(
        description='Run CheXpert experiments.',
        usage='func_run.py <experiment> [<args>]'
    )

    parser.add_argument('experiment', choices=registry.EXPERIMENTS.keys())
    args = parser.parse_args(sys.argv[1:2])

    registry.EXPERIMENTS[args.experiment](sys.argv[2:])


if __name__ == '__main__':
    main()
