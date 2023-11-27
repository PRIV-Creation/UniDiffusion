import sys
import argparse


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--save_diffusers_path", default="", help="path to save diffusers pipeline")
    parser.add_argument(
        "opts",
        help='"path.key=value"',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
