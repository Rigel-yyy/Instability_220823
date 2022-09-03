import csv
import ast
from pathlib import Path
from collections import namedtuple


class CsvParamParser:
    bjx_src_path = Path('./bjx_main_src')

    def __init__(self, name_stem: str) -> None:
        self.csv_pathobj = self.bjx_src_path.joinpath(name_stem + '.csv')
    
    def read_param(self, array_id: int):
        """
        从csv文件里拿参数，array_id对应了行数，第0行是header name
        :param array_id: slurm array id starting from 1
        :return: namedtuple Param object
        """

        with open(self.csv_pathobj) as f:
            f_csv = csv.reader(f)
            headings = next(f_csv)
            Param = namedtuple("Param", headings)
            for _ in range(array_id):
                tmp = next(f_csv)

        arg_list = (ast.literal_eval(item) for item in tmp)
        return Param(*arg_list)
    