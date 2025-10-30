import argparse
import torch
import os
import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


def merge_cli_opt(config, key, value):
    key_hierarchy = key.split(".")
    item_container = config
    for hierarchy in key_hierarchy[:-1]:
        item_container = item_container[hierarchy]
    original_value = item_container[key_hierarchy[-1]]

    if value == 'False':
        value = False
    elif value == 'True':
        value = True
    value = type(original_value)(value) # convert to the same type as the original value
    item_container[key_hierarchy[-1]] = value


def merge_cli_opts(config, cli_opts):
    assert len(cli_opts) % 2 == 0, f"{len(cli_opts)} should be even"
    for key, value in zip(cli_opts[::2], cli_opts[1::2]):
        merge_cli_opt(config, key, value)


def dump_args(args, filename="config.yaml"):
    assert not os.path.exists(filename), f"Do not dump to existing file: {filename}"
    with open(filename, "w") as f:
        yaml.safe_dump(namespace_to_dict(args), f, sort_keys=False)


# https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
def merge_dict(a, b, path=None, allow_replace=False):
    """Merges b into a"""

    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)], allow_replace=allow_replace)
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                if allow_replace:
                    a[key] = b[key]
                else:
                    raise ValueError(f"Conflict at {'.'.join(path + [str(key)])}")
        else:
            a[key] = b[key]
    return a


def load_config(config_path):
    """Load yaml into dictionary.


    Only merges dictionary. Lists will be replaced.

    Args:
        config_path (str): yaml path

    Returns:
        config (dict): config in dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=UniqueKeyLoader)

    if "base_config" not in config:
        return config

    base_path = os.path.join(os.path.dirname(config_path), config["base_config"])
    base_config = load_config(base_path)
    merged_config = merge_dict(base_config, config, allow_replace=True)
    return merged_config


def dict_to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def namespace_to_dict(ns):
    if isinstance(ns, argparse.Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns


def convert_keys_to_lowercase(d):
    lowercase_d = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = convert_keys_to_lowercase(value)
        lowercase_d[key.lower()] = value
    return lowercase_d


def load_args(config_path, cli_opts):
    """Load yaml into args

    Only merges dictionary. Lists will be replaced.

    Args:
        config_path (str): yaml path

    Returns:
        argparse.Namespace: args
    """
    config = load_config(config_path)
    merge_cli_opts(config, cli_opts)
    args = dict_to_namespace(config)
    return args


def to_device(instance, device="cpu"):
    if isinstance(instance, torch.Tensor):
        return instance.to(device=device)
    
    if isinstance(instance, list):
        return [to_device(item, device=device) for item in instance]
    
    if isinstance(instance, dict):
        return {key: to_device(item, device=device) for key, item in instance.items()}

    if isinstance(instance, tuple):
        return tuple((to_device(item, device=device) for item in instance))
    
    return instance
