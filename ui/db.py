#!/usr/bin/env python3

import json
import redis

class DB(object):
    def __init__(self, host='redis', port=6379):
        self.conn = redis.Redis(host=host, port=port, decode_responses=True)
        assert self.conn.ping()
        self._data = None

    @property
    def data(self):
        if isinstance(self._data, str):
            self._data = json.loads(self._data)
        elif isinstance(self._data, list):
            for i in range(len(self._data)):
                if isinstance(self._data[i], str):
                    self._data[i] = json.loads(self._data[i])
        return self._data

    def write(self, key, data):
        if isinstance(data, dict):
            data = json.dumps(data)
        if isinstance(data, list):
            data = json.dumps(data)
        if isinstance(data, tuple):
            data = json.dumps(data)
        self.conn.rpush(key, data)
        return self

    def len(self, key):
        return self.conn.llen(key)

    def read(self, key, index):
        self._data = self.conn.lindex(key, index)
        return self

    def read_all(self, key):
        self._data = []
        for i in range(self.len(key)):
            self._data.append(self.conn.lindex(key, i))
        return self

if __name__ == '__main__':
    db = DB()
    for i in range(10):
        data = {'x': i, 'y':i, 'epoch':i}
        db.write('tlist', data)
    data = db.read('tlist', 0).data
    print(type(data), data)
    print(db.len('tlist'))
    print(db.read_all('tlist').data)
