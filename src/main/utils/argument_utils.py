import sys


def parse_arguments(arg_parser):
    a_args = sys.argv
    a_args.pop(0)
    return arg_parser.parse_args(a_args)