import bjoern
import wsgilog

from tfi.serve.flask import make_app, make_deferred_app

import socket
import logging

LISTEN_BACKLOG = 1024

def _run_app(app, host, port, on_bind=None):
    logged_app = wsgilog.WsgiLog(app, tostream=True, loglevel=logging.WARN)

    with socket.socket(socket.AF_INET) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        bound_port = s.getsockname()[1]
        if on_bind:
            on_bind("http://%s:%s" % (host, bound_port))
        s.listen(LISTEN_BACKLOG)
        bjoern.server_run(s, logged_app)

def run_deferred(*, host, port, on_bind, **kw):
    app = make_deferred_app(**kw)
    return _run_app(app, host=host, port=port, on_bind=on_bind)

def run(model, *, host, port, on_bind, **kw):
    app = make_app(model, **kw)
    return _run_app(app, host=host, port=port, on_bind=on_bind)
