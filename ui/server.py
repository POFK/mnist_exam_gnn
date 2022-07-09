#!/usr/bin/env python3

import asyncio
import random

from async_dash import Dash
from dash import html, Output, Input, dcc
from dash_extensions import WebSocket
from quart import websocket, json

from db import DB

db = DB()
name = 'test_ui'
DATA = {}
DATA['epoch'] = []
DATA['loss'] = []

app = Dash(__name__)

app.layout = html.Div([WebSocket(id="ws"), dcc.Graph(id="graph")])

app.clientside_callback(
    """
function(msg) {
    if (msg) {
        const data = JSON.parse(msg.data);
        return {data: [{x: data.epoch, y: data.loss, type: "scatter"}]};
    } else {
        return {};
    }
}""",
    Output("graph", "figure"),
    [Input("ws", "message")],
)


@app.server.websocket("/ws")
async def ws():
    while True:
        point = json.loads(db.conn.blpop(name)[1])
        DATA['epoch'].append(point['epoch'])
        DATA['loss'].append(point['loss'])
        output = json.dumps(DATA)
        await websocket.send(output)
#       await asyncio.sleep(1)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80, debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter
