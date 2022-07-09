import json
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
    while True:
        point = json.loads(db.conn.blpop(name)[1])
        ds.append(point)
        output = json.dumps(ds.data)
        db.conn.hset(name+"_pv", 'value', output)
        sock.send(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
