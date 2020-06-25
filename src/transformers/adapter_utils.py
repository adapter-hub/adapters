import hashlib
import json
import logging
import os
import re
import shutil
import tarfile
from collections.abc import Mapping
from os.path import isdir, isfile, join
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import requests
from filelock import FileLock

from .adapter_config import ADAPTER_CONFIG_MAP, AdapterType, get_adapter_config_hash
from .file_utils import get_from_cache, is_remote_url, torch_cache_home


logger = logging.getLogger(__name__)

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "pytorch_adapter.bin"
HEAD_CONFIG_NAME = "head_config.json"
HEAD_WEIGHTS_NAME = "pytorch_model_head.bin"

ADAPTER_IDENTIFIER_PATTERN = r"[a-zA-Z\-_\/@]{2,}"
ADAPTER_HUB_URL = "https://raw.githubusercontent.com/calpt/nothing-to-see-here/master/dist/"
ADAPTER_HUB_INDEX_FILE = ADAPTER_HUB_URL + "index_{}/{}.json"
ADAPTER_HUB_CONFIG_FILE = ADAPTER_HUB_URL + "architectures.json"

# the download cache
ADAPTER_CACHE = join(torch_cache_home, "adapters")


def inherit_doc(cls):
    for name, func in vars(cls).items():
        if isinstance(func, Callable) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, "__doc__", None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls


def urljoin(*args):
    return "/".join([s.strip("/") for s in args])


def remote_file_exists(url):
    r = requests.head(url)
    return r.status_code == 200


def download_cached(url, checksum=None, checksum_algo="sha1", cache_dir=None, force_extract=False, **kwargs):
    if not cache_dir:
        cache_dir = ADAPTER_CACHE
    if isinstance(url, Path):
        url = str(url)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url):
        output_path = get_from_cache(url, cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError("Unable to parse '{}' as a URL".format(url))

    if not output_path:
        return None

    # if checksum is given, verify it
    if checksum and checksum_algo:
        h = hashlib.new(checksum_algo)
        with open(output_path, "rb") as f:
            h.update(f.read())
        calculated_checksum = h.hexdigest()
        if calculated_checksum != checksum.lower():
            raise EnvironmentError("Failed to verify checksum of '{}'".format(output_path))

    if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
        return output_path

    # Path where we extract compressed archives
    # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
    output_dir, output_file = os.path.split(output_path)
    output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
    output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

    if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
        return output_path_extracted

    # Prevent parallel extractions
    lock_path = output_path + ".lock"
    with FileLock(lock_path):
        shutil.rmtree(output_path_extracted, ignore_errors=True)
        os.makedirs(output_path_extracted)
        if is_zipfile(output_path):
            with ZipFile(output_path, "r") as zip_file:
                zip_file.extractall(output_path_extracted)
                zip_file.close()
        elif tarfile.is_tarfile(output_path):
            tar_file = tarfile.open(output_path)
            tar_file.extractall(output_path_extracted)
            tar_file.close()
        else:
            raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

    return output_path_extracted


def resolve_adapter_config(config: Union[dict, str]) -> dict:
    """Resolves a given adapter configuration specifier to a full configuration dictionary.

    Args:
        config (Union[dict, str]): The configuration to resolve. Can be either:
            - a dictionary: returned without further action
            - an identifier string available in the local ADAPTER_CONFIG_MAP
            - the path to a file containing a full adapter configuration
            - an identifier string available in Adapter-Hub

    Returns:
        dict: The resolved adapter configuration dictionary.
    """
    # already a dict, so we don't have to do anything
    if isinstance(config, Mapping):
        return config
    # first, look in local map
    if config in ADAPTER_CONFIG_MAP:
        return ADAPTER_CONFIG_MAP[config]
    # load from file system if it's a local file
    if isfile(config):
        with open(config, "r") as f:
            loaded_config = json.load(f)
            # search for nested config if the loaded dict has the form of a config saved with an adapter module
            if "config" in loaded_config:
                return loaded_config["config"]
            else:
                return loaded_config
    # now, try to find in hub index
    index_file = download_cached(ADAPTER_HUB_CONFIG_FILE)
    if not index_file:
        raise EnvironmentError("Unable to load adapter hub index file. The file might be temporarily unavailable.")
    with open(index_file, "r") as f:
        config_index = json.load(f)
    if config in config_index:
        return config_index[config]
    else:
        raise ValueError("Could not identify '{}' as a valid adapter configuration.".format(config))


def _split_identifier(identifier):
    task, subtask, org_name = None, None, None
    identifier = identifier.split("@")
    if len(identifier) > 1:
        org_name = identifier[1]
    identifier = identifier[0].split("/")
    if len(identifier) > 1:
        subtask = identifier[1]
    task = identifier[0]
    return task, subtask, org_name


def _dict_extract(d, primary_key, secondary_key=None):
    for k, v in d.items():
        if k == primary_key:
            if secondary_key:
                if secondary_key in v.keys():
                    yield v[secondary_key]
            else:
                for k, v in v.items():
                    yield v
        else:
            for k, v in v.items():
                if k == primary_key:
                    yield v


def find_in_index(
    identifier: str, adapter_config: dict, adapter_type: AdapterType, model_name: str, strict: bool = True
) -> Optional[str]:
    if not model_name:
        raise ValueError("Unable to resolve adapter without the name of a model. Please specify model_name.")
    index_file = download_cached(ADAPTER_HUB_INDEX_FILE.format(adapter_type, model_name))
    if not index_file:
        raise EnvironmentError("Unable to load adapter hub index file. The file might be temporarily unavailable.")
    with open(index_file, "r") as f:
        adapter_index = json.load(f)
    # split into <task>/<subtask>@<org>
    task, subtask, org = _split_identifier(identifier)
    # find all entries for this task and subtask
    entries = list(_dict_extract(adapter_index, task, subtask))
    if not entries:
        # we found no matching entry
        return None
    elif len(entries) == 1:
        index_entry = entries[0]
    else:
        # there are multiple possible options for this identifier
        raise ValueError("Found multiple possible adapters matching '{}'.".format(identifier))
    # go on with searching a matching adapter_config hash in the task entry
    assert adapter_config, "Specify an adapter configuration to search for."
    config_hash = get_adapter_config_hash(adapter_config)
    if config_hash in index_entry:
        # now match the org if given
        hub_entry = _get_matching_version(index_entry[config_hash], org)
        if hub_entry:
            logger.info("Found matching adapter at: {}".format(hub_entry))
        return hub_entry
    # there's only one possible config and we allow matches with different configs
    elif not strict and len(index_entry) == 1:
        logger.warn("No matching adapter config found for this specifier, falling back to default.")
        config_entry = list(index_entry.values())[0]
        return _get_matching_version(config_entry, org)
    else:
        raise ValueError("No adapter '{}' found for the current model or configuration.".format(identifier))


def _get_matching_version(config_entry, org):
    if org:
        return config_entry["versions"].get(org, None)
    elif len(config_entry["versions"]) == 1:
        return list(config_entry["versions"].values())[0]
    elif "default" in config_entry:
        return config_entry["versions"]["default"]
    else:
        raise ValueError("Multiple adapters with this name are available for this config.")


def http_get_json(url):
    # check if it's a relative url
    if not urlparse(url).netloc:
        url = urljoin(ADAPTER_HUB_URL, url)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise EnvironmentError("Failed to get file {}".format(url))


def pull_from_hub(
    specifier: str,
    adapter_config: Union[dict, str],
    adapter_type: AdapterType,
    model_name: str,
    version: str = None,
    strict: bool = True,
    **kwargs
) -> str:
    """Downloads a pre-trained adapter module from Adapter-Hub

    Args:
        specifier (str): A string specifying the adapter to be loaded.
        adapter_config (Union[dict, str]): The configuration of the adapter to be loaded.
        adapter_type (AdapterType): The adapter type.
        model_name (str): The identifier of the pre-trained model for which to load an adapter.
        version (str, optional): The version of the adapter to be loaded. Defaults to None.
        strict (bool, optional): If set to True, only allow adapters exactly matching the given config to be loaded. Defaults to True.

    Returns:
        str: The local path to which the adapter has been downloaded.
    """
    if not adapter_config or not adapter_type or not model_name:
        raise ValueError("adapter_config, adapter_type and model_name arguments must be given.")
    # resolve config if it's an identifier
    adapter_config = resolve_adapter_config(adapter_config)
    # search the correct entry in the index
    hub_entry_url = find_in_index(specifier, adapter_config, adapter_type, model_name, strict=strict)
    if not hub_entry_url:
        raise EnvironmentError("No adapter with name '{}' was found in the adapter index.".format(specifier))
    hub_entry = http_get_json(hub_entry_url)

    # set version
    if not version:
        version = hub_entry["default_version"]
    elif version not in hub_entry["files"]:
        logger.warn("Version '{}' of adapter '{}' not found. Falling back to default.".format(version, specifier))
        version = hub_entry["default_version"]
    file_entry = hub_entry["files"][version]

    # start downloading
    logger.info("Resolved adapter files at {}.".format(file_entry["url"]))
    # TODO add support for other checksums
    download_path = download_cached(file_entry["url"], checksum=file_entry["sha1"], **kwargs)
    if not download_path:
        raise EnvironmentError("Unable to load file from {}. The file might be unavailable.".format(file_entry["url"]))
    return download_path


def resolve_adapter_path(
    adapter_name_or_path,
    adapter_config: Union[dict, str] = None,
    adapter_type: AdapterType = AdapterType.text_task,
    model_name: str = None,
    version: str = None,
    **kwargs
) -> str:
    """Resolves the path to a pre-trained adapter module.
    Note: If attempting to resolve an adapter from the Hub, adapter_config, adapter_type and model_name must be present.

    Args:
        adapter_name_or_path (str): Can be either:
            - the path to a folder in the file system containing the adapter configuration and weights
            - an url pointing to a zip folder containing the adapter configuration and weights
            - a specifier matching a pre-trained adapter uploaded to Adapter-Hub
        adapter_config (Union[dict, str], optional): The configuration of the adapter to be loaded.
        adapter_type (AdapterType, optional): The adapter type.
        model_name (str, optional): The identifier of the pre-trained model for which to load an adapter.
        version (str, optional): The version of the adapter to be loaded. Defaults to None.

    Returns:
        str: The local path from where the adapter module can be loaded.
    """
    # url of a folder containing pretrained adapters -> try to load from this url
    if is_remote_url(adapter_name_or_path):
        resolved_folder = download_cached(adapter_name_or_path, **kwargs)
        if not resolved_folder:
            raise EnvironmentError(
                "Unable to load file from {}. The file might be unavailable.".format(resolved_folder)
            )
        return resolved_folder
    # path to a local folder saved using save()
    elif isdir(adapter_name_or_path):
        if isfile(join(adapter_name_or_path, WEIGHTS_NAME)) and isfile(join(adapter_name_or_path, CONFIG_NAME)):
            return adapter_name_or_path
        else:
            raise EnvironmentError(
                "No file {} or no file {} found in directory {}".format(
                    WEIGHTS_NAME, CONFIG_NAME, adapter_name_or_path
                )
            )
    # matches possible form of identifier in hub
    elif re.fullmatch(ADAPTER_IDENTIFIER_PATTERN, adapter_name_or_path):
        if not adapter_type:  # make sure we have set an adapter_type
            adapter_type = AdapterType.text_task
        return pull_from_hub(adapter_name_or_path, adapter_config, adapter_type, model_name, version=version, **kwargs)
    else:
        raise ValueError("Unable to identify {} as a valid module location.".format(adapter_name_or_path))
