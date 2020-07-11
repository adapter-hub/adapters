#!/usr/bin/env python3
from argparse import ArgumentParser

from transformers.commands.adapter_pack import AdapterPackCommand


def main():
    parser = ArgumentParser()
    command_parser = parser.add_subparsers()

    AdapterPackCommand.register_subcommand(command_parser)

    args = parser.parse_args()

    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
