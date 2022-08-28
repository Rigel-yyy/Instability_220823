# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:05:15 2022

@author: Rigel_yyy
"""


class BaseModel:
    """
    simulation configuration info
    """
    __slots__ = ["N_ROW", "N_COLUMN", "GRID",
                 "T_FINAL", "T_STEP", "N_FRAME",
                 "PHI_CELL", "PHI_ECM",
                 "ZETA_CELL", "ZETA_ECM", "XI",
                 "NRG", "SURF", "GROWTH_RATE"]

    @staticmethod
    def set_config(config):
        for name in BaseModel.__slots__:
            setattr(BaseModel, name, config[name])

    @staticmethod
    def get_config():
        return {name: getattr(BaseModel, name) for name in BaseModel.__slots__}

    def __init__(self, config=None):
        if config:
            self.set_config(config)
