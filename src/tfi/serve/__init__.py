from tfi.serve.flask import run_deferred as flask_run_deferred
from tfi.serve.flask import run as flask_run

from tfi.serve.gunicorn import run_deferred as gunicorn_run_deferred
from tfi.serve.gunicorn import run as gunicorn_run

def run_deferred(**kw)
    try:
        return gunicorn_run_deferred(**kw)
    except ModuleNotFoundError:
        pass

    return flask_run_deferred(**kw)

def run(model, **kw)
    try:
        return gunicorn_run(model, **kw)
    except ModuleNotFoundError:
        pass

    return flask_run(model, **kw)
