import json
import time
from flask import Flask, render_template
from flask_sock import Sock

from db import DB
from data import DataSet

app = Flask(__name__)
sock = Sock(app)

@app.route('/')
def index():
    return render_template('index.html')

db = DB()
ds = DataSet()
name = 'test_ui'

@sock.route('/echo')
def echo(sock):
    ds = DataSet()
    points = db.read_all("pv_"+db.status['name']).data
    cnt = 0
    for point in points:
        ds.append(point)
        cnt += 1
    if cnt:
        output = json.dumps(ds.data)
        sock.send(output)
    while True:
        print("sock called!", db.status["running"])
        if not db.status["running"]:
            time.sleep(10)
            continue
        point = db.blpop(name)
        print(point)
        ds.append(point)
        output = json.dumps(ds.data)
        sock.send(output)
        time.sleep(0.1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
