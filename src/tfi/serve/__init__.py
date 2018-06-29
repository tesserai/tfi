import bjoern
import wsgilog

from tfi.serve.flask import make_app, make_deferred_app

def _run_app(app, host, port):
    logged_app = wsgilog.WsgiLog(app, tostream=True
    bjoern.run(logged_app, host, port)

def run_deferred(*, host, port, **kw):
    app = make_deferred_app(**kw)
    return _run_app(app, host=host, port=port)

def run(model, *, host, port, **kw):
    app = make_app(model, **kw)
    return _run_app(app, host=host, port=port)
