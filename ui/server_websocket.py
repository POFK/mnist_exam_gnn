import json
import time
from flask import Flask, render_template, request
from flask_sock import Sock

from db import DB
from data import DataSet

app = Flask(__name__)
sock = Sock(app)

db = DB()
ds = DataSet()

@app.route('/')
def index():
    model_name = db.status['name']
    return render_template('index.html', model_name=model_name)

@app.route('/model_name', methods=['POST'])
def name():
    model_name = request.form['value']
    print(f"get model name: {model_name}")
    db.conn.hset("status","name",model_name)
    db.conn.hset("status","running","False")
    return {}

@sock.route('/echo')
def echo(sock):
    ds = DataSet()
    points = db.read_all("pv_"+db.status['name']).data
    print("exist data points:", points)
    cnt = 0
    for point in points:
        ds.append(point)
        cnt += 1
    if cnt:
        output = json.dumps(ds.data)
        sock.send(output)
    while True:
        status = db.status
        print("sock called!", status)
        if not status["running"]:
            time.sleep(10)
            continue
        point = db.blpop()
        print(status, point)
        if point is None:
            continue
        ds.append(point)
        output = json.dumps(ds.data)
        sock.send(output)
        time.sleep(0.001)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
