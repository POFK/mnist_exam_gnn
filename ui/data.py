#!/usr/bin/env python3

class DataSet(object):
    def __init__(self, *args):
        self._keys = args
        self._data = {}
        for i in args:
            self._data[i] = []

    def append(self, dict):
        for i in dict.keys():
            if i not in self._data:
                self._data[i] = []
            self._data[i].append(dict.get(i))
        return self

    def __getattr__(self, attr):
        return self._data[attr]

    @property
    def data(self):
        return self._data
