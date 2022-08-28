import json
from typing import Dict

import numpy as np
import zipfile
import io

from . import path_tools


def save_config(config_data: Dict):
    """
    save config files
    """

    fileName = "config.json"
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)
    print("config file saved successfully!")


def npz_append_save(file_name: str, key: str, data):
    """
    save to filename.npz
    key: str, usually the time index
    data: np.ndarray, data to save
    """
    try:
        bio = io.BytesIO()
        np.save(bio, data)
        with zipfile.ZipFile(file_name + ".npz", 'a') as zipf:
            zipf.writestr(key, data=bio.getbuffer().tobytes())
    except FileNotFoundError:
        save_dict = {key: data}
        np.savez_compressed(file_name + ".npz", **save_dict)


def load_config():
    """
    find and load config data in current directory

    -------
    return : config Dict
    """

    file_items = path_tools.get_all_file()
    config_files = path_tools.select_file(file_items, 'config', '.json', ret='name')
    if len(config_files) != 1:
        raise RuntimeError("Configuration file not unique!")

    with open(config_files[0]) as f:
        config = json.load(f)

    return config


def load_data():
    """
    find and load all .npy files in current directory

    ---------
    return : Dict[filename : data]
    """

    file_items = path_tools.get_all_file()
    data_files = path_tools.select_file(file_items, '.npy', ret='name')

    return {data_name: np.load(data_name) for data_name in data_files}
