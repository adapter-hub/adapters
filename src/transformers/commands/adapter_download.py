from argparse import ArgumentParser, Namespace

from transformers.commands import BaseTransformersCLICommand


def adapter_download_command_factory(args: Namespace):
    return AdapterDownloadCommand(
        args.adapter, args.config, args.type, args.model, args.version, args.cache_dir, args.force
    )


class AdapterDownloadCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("download", help="Download an adapter from AdapterHub.")
        download_parser.add_argument("adapter", type=str, help="String identifier of the adapter to download")
        download_parser.add_argument(
            "--cache-dir", type=str, default=None, help="Path to location to store the adapters"
        )
        download_parser.add_argument(
            "--force", action="store_true", help="Force the adapter to be downloaded even if already in cache"
        )
        download_parser.add_argument(
            "-c",
            "--config",
            type=str,
            help="The config of the adapter to be downloaded. Can be a path or an identifier.",
        )
        download_parser.add_argument(
            "-t",
            "--type",
            type=str,
            default="text_task",
            help="The type of adapter to be downloaded. Defaults to text_task.",
        )
        download_parser.add_argument(
            "-m", "--model", type=str, help="The name of the model for which to download an adapter."
        )
        download_parser.add_argument("-v", "--version", type=str, help="The version of the adapter to be downloaded.")
        download_parser.set_defaults(func=adapter_download_command_factory)

    def __init__(
        self, adapter: str, config: str, type: str, model_name: str, version: str, cache_dir: str, force_download: bool
    ):
        self.adapter = adapter
        self.config = config
        self.type = type
        self.model_name = model_name
        self.version = version
        self.cache_dir = cache_dir
        self.force_download = force_download

    def run(self):
        from transformers import pull_from_hub

        pull_from_hub(
            self.adapter,
            self.config,
            self.type,
            self.model_name,
            self.version,
            cache_dir=self.cache_dir,
            force_download=self.force_download,
        )
