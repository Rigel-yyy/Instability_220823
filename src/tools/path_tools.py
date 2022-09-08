import os
import time
from contextlib import contextmanager
from pathlib import Path

from .day_tools import getTimeStamp


@contextmanager
def working_directory(path):
    """
    A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    Usage:
    > # Do something in original directory
    > with working_directory('/my/new/path'):
    >     # Do something in new directory
    > # Back to old directory
    """

    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def get_save_path(res_path, day_stamp=None, name=None, max_retry=100):
    """
    by default simulation results are saved in path: results/220621/003
    max_retry on creating order dir
    """

    if day_stamp is None:
        day_stamp = getTimeStamp(minute=False)

    path_to_day_result = Path(res_path).joinpath(day_stamp)
    path_to_day_result.mkdir(exist_ok=True)

    if name is None:
        for _ in range(max_retry):
            sim_order_list = [extract_order(item) for item in get_all_dir(str(path_to_day_result))
                                                  if extract_order(item) is not None]
            if not sim_order_list:
                max_order = 0
            else:
                max_order = max(sim_order_list)
            path_to_sim_result = path_to_day_result.joinpath(str(max_order + 1).zfill(3))
            try:
                path_to_sim_result.mkdir()
                return path_to_sim_result
            except FileExistsError:
                time.sleep(0.5)  # retry after 0.5s

        raise RuntimeError(f"FileExistsError encountered in all {max_retry} retries!")

    else:
        path_to_sim_result = path_to_day_result.joinpath(name)
        path_to_sim_result.mkdir()
        return path_to_sim_result


def extract_order(pathobj: Path):
    """
    extract simulation order from path name
    """

    dirstr = str(pathobj.name)
    if not dirstr.isdigit():
        return
    else:
        return int(dirstr[-3:])


def get_all_dir(des_path: str = None, ret: str = 'obj'):
    """
    get all directory in desPath

    ----------
    des_path : str, optional
        Destination path. The default is None.
    ret : str, 'name' or 'obj'
        Return type. The default is 'obj'.
    """

    if des_path is None:
        des = Path.cwd()
    else:
        des = Path(des_path)
        if not des.exists():
            raise FileExistsError
    if ret == 'obj':
        return [item for item in des.iterdir() if item.is_dir()]
    if ret == 'name':
        return [item.name for item in des.iterdir() if item.is_dir()]


def get_all_file(des_path: str = None, ret: str = 'obj'):
    """
    get all file objects in desPath

    ----------
    des_path : str, optional
        Destination path. The default is None.
    ret : str, 'name' or 'obj'
        Return type. The default is 'obj'.
    """

    if des_path is None:
        des = Path.cwd()
    else:
        des = Path(des_path)
        if not des.exists():
            raise FileExistsError
    if ret == 'obj':
        return [item for item in des.iterdir() if item.is_file()]
    if ret == 'name':
        return [item.name for item in des.iterdir() if item.is_file()]


def select_file(item_list, *args, ret: str = 'obj'):
    """
    select item from item_list according to key word strings

    Parameters
    ----------
    item_list : List[Path | str]
    ret : str, 'name' or 'obj'
        Return type. The default is 'obj'.
    *args : str
        White list keywords for selection.
    """

    if not isinstance(item_list, list):
        item_list = [item_list]

    obj_list = []
    for item in item_list:
        if isinstance(item, str):
            obj_list.append(Path(item))
        if isinstance(item, Path):
            obj_list.append(item)

    file_list = [obj for obj in obj_list if obj.is_file()]

    result = []
    for file in file_list:
        check = True
        for key_word in args:
            if key_word not in file.name:
                check = False
                break
        if check:
            if ret == 'obj':
                result.append(file)
            if ret == 'name':
                result.append(file.name)
    return result
