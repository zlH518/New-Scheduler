import argparse


def _add_base_cmd(parser):
    parser.add_argument(
        "--config", type=str, default="example1/config.yaml", help="Path of config"
    )
    parser.add_argument(
        "--seed", type=int, default=3407, help="Random seed in integer, default is 3407"
    )
    return parser



def get_cmd_args():
    parser = argparse.ArgumentParser(description="")
    parser = _add_base_cmd(parser)
    return parser.parse_args()