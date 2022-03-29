# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File        : json_reader.py
@Version     : 1.0
@Description : json file reader
"""
import json


class JsonReader(object):
    @staticmethod
    def read_json(json_file):
        """ Loading mix info data according to json info file

        Args:
            json_file: str
                absolute path to json info file
        Returns:
            item_num: int
                the length of mix info
            item_key: list
                the key list of the item_dict
            item_dict: dict[dict]
                the content of mix info:
                key: key of the mix info
                value: dict_keys:['s1', 's2', 'mix', 'snr', 'sr', 'ref']
        """
        with open(json_file, 'r') as f:
            item_dict = json.load(f)
        # for key, value in item_dict.items():
        #     print(key, value)
        item_num = len(item_dict.keys())
        item_key = [key for key in item_dict.keys()]
        return item_num, item_key, item_dict
