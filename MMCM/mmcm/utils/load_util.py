import json
import yaml
import importlib


def load_dct_from_file(path, obj_name=None):
    if path.endswith(".json"):
        dct = load_json(path)
    elif path.endswith(".yaml"):
        dct = load_yaml(path)
    elif path.endswith(".py"):
        dct = load_edct_py(path, obj_name)
    else:
        raise ValueError("unsupported config file")
    return dct


def load_json(path):
    with open(path, "r") as f:
        dct = json.load(f)
    return dct


def load_yaml(path):
    dct = yaml.load(path)
    return dct


def load_pyhon_obj(path, obj_name):
    module_name = "module_name"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, obj_name)
