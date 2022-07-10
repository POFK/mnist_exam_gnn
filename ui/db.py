#!/usr/bin/env python3

import json
import redis
import time

class DB(object):
    def __init__(self, host='redis', port=6379):
        self.conn = redis.Redis(host=host, port=port, decode_responses=True)
        assert self.conn.ping()
        self._data = None
        self._status = {"name": "test_ui", "running": "False"}
        self.conn.hset("status", mapping=self._status)
        self.child = []

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
            data = self.conn.lindex(key, i)
            if not len(data):
                break
            self._data.append(data)
        return self

    def get_child(self, key):
        self.child = list(self.conn.scan_iter(key+"/*"))
        return self

    def wait(self, sec):
        time.sleep(sec)
        return self

    def blpop(self, key):
        while len(self.child) == 0:
            print("Waiting for task running!")
            self.get_child(key).wait(5)
        attr_comb = {}
        for k in self.child:
            attr = self.conn.blpop(k)[1]
            attr = json.loads(attr)
            for i,j in attr.items():
#               print(k,i, j)
                attr_comb[i] = attr_comb.get(i, 0) + j
        for k in attr_comb.keys():
            attr_comb[k] /= len(self.child)
        self.pv(key, attr_comb)
        return attr_comb

    def pv(self, key, value):
        self.write("pv_"+key, value)
        return self

    @property
    def status(self):
        d = {'True': True, 'False': False}
        self._status = self.conn.hgetall("status")
        self._status["running"] = d.get(self._status["running"])
        if self.len(self._status["name"]+"/0"):
            self._status["running"] = True
        return self._status


if __name__ == '__main__':
    db = DB()
    name = "test_ui"
    # mock data
    import math
    import random
    import time
    from os.path import join as opj
    data = {}
    status = {"name": name, "running": "True"}
    db.conn.hset("status", mapping=status)
    for i in range(20):
        time.sleep(0.2)
        x = i
        y = math.sin(x/3.14/2+random.random())
        data['epoch'] = (x)
        data['tr_loss'] = (y)
        data['loss'] = (y)
        data['te_loss'] = (y+0.1)
        data['acc'] = (y-0.1)
        db.write(opj(name,"0"), data)
        db.write(opj(name,"1"), data)
        print(db.status)
    time.sleep(2)
    db.conn.hset("status", "running", "False")
    print(db.status)


#   print(list(db.conn.scan_iter(name+"/*")))
#   print(db.blpop("test_ui"))
#   print("read all points:")
#   for p in db.read_all("pv_"+name).data:
#       print(p)
#   print(db.status)


#   for i in range(10):
#       data = {'loss': i, 'acc':i, 'epoch':i}
#       db.write(name, data)
#   data = db.read(name, 0).data
#   print(type(data), data)
#   print(db.read_all(name).data)
