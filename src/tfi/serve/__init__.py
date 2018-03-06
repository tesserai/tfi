from tfi.serve.flask import make_app, make_deferred_app

def _run_app(app, host, port, prefork_ok):
    if prefork_ok:
        try:
            from tfi.serve.gunicorn import run_app as gunicorn_run_app
            return gunicorn_run_app(app, host=host, port=port)
        except ModuleNotFoundError:
            pass

    import werkzeug.serving
    return werkzeug.serving.run_simple(hostname=host, port=port, application=app)

def run_deferred(*, host, port, prefork_ok, **kw):
    app = make_deferred_app(**kw)
    return _run_app(app, host=host, port=port, prefork_ok=prefork_ok)

def run(model, *, host, port, prefork_ok, **kw):
    app = make_app(model, **kw)
    return _run_app(app, host=host, port=port, prefork_ok=prefork_ok)
