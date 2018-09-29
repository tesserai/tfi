import bjoern as _bjoern
import logging as _logging
import socket as _socket
import sys as _sys

from functools import partial as _partial

from tfi.serve.app import make_app as _make_app
from tfi.serve.fission import FissionEnvironment as _FissionEnvironment
from tfi.serve.logging import log_requests_logrus as _log_requests_logrus
from tfi.serve.tracing import maybe_trace_app as _maybe_trace_app

LISTEN_BACKLOG = 1024

def _run_wsgi(app, host, port, on_bind=None):
  with _socket.socket(_socket.AF_INET) as s:
    s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    bound_port = s.getsockname()[1]
    if on_bind:
      on_bind("http://%s:%s" % (host, bound_port))
    s.listen(LISTEN_BACKLOG)
    _bjoern.server_run(s, app)

def _log_to_stdout(app):
  for handler in app.logger.handlers:
    app.logger.removeHandler(handler)

  h = _logging.StreamHandler(_sys.stdout)
  h.setLevel(_logging.INFO)
  h.setFormatter(_logging.Formatter('level=%(levelname)s %(message)s'))
  app.logger.addHandler(h)
  app.logger.propagate = False

  app.logger.setLevel(_logging.INFO)

def run_deferred(*, host, port, on_bind, jaeger_tags, jaeger_host, load_model_from_path_fn, **kw):
  def _specialize(specialization_request):
    tags = dict(jaeger_tags)
    tags.update(specialization_request.jaeger_tags())

    tracer = _partial(_maybe_trace_app,
      jaeger_service_name=specialization_request.jaeger_service_name(),
      jaeger_tags=tags,
      jaeger_host=jaeger_host,
      operation_name_fn=specialization_request.operation_name_fn(),
    )

    app = _make_app(
      load_model_from_path_fn(specialization_request.filepath()),
      tracer,
      model_file_fn=lambda: specialization_request.filepath(),
      **kw,
    )
    _log_to_stdout(app)
    _log_requests_logrus(app)

    return app

  fission_env = _FissionEnvironment(_specialize)
  _log_to_stdout(fission_env.app())
  _log_requests_logrus(fission_env.app())
  
  return _run_wsgi(fission_env, host=host, port=port, on_bind=on_bind)

def run(model, *, host, port, on_bind, jaeger_tags, jaeger_host, jaeger_service_name=None, **kw):
  tracer = _partial(_maybe_trace_app,
    jaeger_service_name=jaeger_service_name or 'tfi',
    jaeger_tags=jaeger_tags,
    jaeger_host=jaeger_host,
    operation_name_fn=lambda request, parsed_url: "HTTP %s %s" % (request.method, parsed_url.path),
  )

  app = _make_app(model, tracer, **kw)
  _log_to_stdout(app)
  _log_requests_logrus(app)

  return _run_wsgi(app, host=host, port=port, on_bind=on_bind)
