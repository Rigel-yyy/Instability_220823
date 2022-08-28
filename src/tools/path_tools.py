import os
from contextlib import contextmanager
from pathlib import Path, PurePath

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


def get_save_path(resPath, dayStamp=None, name=None):
    """
    by default simulation results are saved in path: results/220621/003
    """

    if dayStamp is None:
        dayStamp = getTimeStamp(minute=False)
    tempPathName = PurePath(resPath).joinpath(dayStamp)
    Path(tempPathName).mkdir(exist_ok=True)

    if name is None:
        dirPathList = get_all_dir(str(tempPathName))
        orderList = [extract_order(item) for item in dirPathList
                     if extract_order(item) is not None]
        if not orderList:
            maxOrder = 0
        else:
            maxOrder = max(orderList)
        name = str(maxOrder + 1).zfill(3)

    tempPathName = tempPathName.joinpath(name)
    pathObj = Path(tempPathName)
    pathObj.mkdir()
    return pathObj


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
