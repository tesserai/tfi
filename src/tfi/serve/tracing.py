from flask import request

from urllib.parse import urlparse

import opentracing
from opentracing.ext import tags as opentracing_ext_tags

def _as_public_url(parsed_url, headers):
  if 'X-Forwarded-Proto' in headers:
    parsed_url = parsed_url._replace(scheme=headers['X-Forwarded-Proto'])

  if 'X-Forwarded-Host' in headers:
    parsed_url = parsed_url._replace(netloc=headers['X-Forwarded-Host'])

  return parsed_url


def _maybe_inject_trace_id(span, response):
  if 'Request-Id' in response.headers:
    return
  trace_id = '{:x}'.format(span.trace_id)
  response.headers['Request-Id'] = trace_id
  span.set_tag(opentracing_ext_tags.HTTP_STATUS_CODE, str(response.status_code))
  return

def _trace_request(span, request):
  span.log_kv({
    "http.request.header." + k.lower(): v
    for k, v in request.headers.items()
  })

  if request.form:
    span.log_kv({
      "http.request.form.%s" % k: v
      for k, v in request.form.items(multi=True)
    })
  
  if request.data:
    span.log_kv({
      "http.request.data": request.data.decode("utf-8"),
    })

def _trace_response(span, response):
  _maybe_inject_trace_id(span, response)

  span.log_kv({
    "http.response.header." + k.lower(): v
    for k, v in response.headers.items()
  })
  span.log_kv({
    "http.response.data": response.data.decode("utf-8"),
  })

def maybe_trace_app(app,
    operation_name_fn=None,
    jaeger_host=None,
    jaeger_tags=None,
    jaeger_service_name=None):
  import opentracing

  if jaeger_host is None:
    return lambda _: lambda fn: fn

  import jaeger_client

  # Create configuration object with enabled logging and sampling of all requests.
  config = jaeger_client.Config(config={'sampler': {'type': 'const', 'param': 1},
              'logging': True,
              'local_agent':
              # Also, provide a hostname of Jaeger instance to send traces to.
              {'reporting_host': jaeger_host}},

          # Service name can be arbitrary string describing this particular web service.
          service_name=jaeger_service_name or 'tfi')

  if operation_name_fn is None:
    operation_name_fn = lambda request, parsed_url: "HTTP %s %s" % (request.method, parsed_url.path)

  current_spans = {}
  traced_attributes = []
  tracer = config.initialize_tracer()
  if tracer is None:
    return lambda _: lambda fn: fn

  from flask import _request_ctx_stack as stack

  def get_span(request=None):
    '''
    Returns the span tracing `request`, or the current request if
    `request==None`.

    If there is no such span, get_span returns None.

    @param request the request to get the span from
    '''
    if request is None and stack.top:
      request = stack.top.request
    # the pop call can fail if the request is interrupted by a `before_request` method so we need a default
    return current_spans.get(request, None)

  @app.before_request
  def _start_trace():
    request = stack.top.request
    parsed_url = _as_public_url(urlparse(request.url), request.headers)

    operation_name = operation_name_fn(request, parsed_url)
    headers = {}
    for k,v in request.headers:
      headers[k.lower()] = v

    span = None
    try:
      span_ctx = tracer.extract(opentracing.Format.HTTP_HEADERS, headers)
      span = tracer.start_span(operation_name=operation_name, child_of=span_ctx)
    except (opentracing.InvalidCarrierException, opentracing.SpanContextCorruptedException) as e:
      span = tracer.start_span(operation_name=operation_name, tags={"Extract failed": str(e)})
    if span is None:
      span = tracer.start_span(operation_name)

    span.set_tag(opentracing_ext_tags.HTTP_URL, parsed_url.geturl())
    span.set_tag(opentracing_ext_tags.HTTP_METHOD, request.method)

    if jaeger_tags:
      for k, v in jaeger_tags.items():
        span.set_tag(k, v)

    current_spans[request] = span
    for attr in traced_attributes:
      if hasattr(request, attr):
        payload = str(getattr(request, attr))
        if payload:
          span.set_tag(attr, payload)

  @app.after_request
  def _after_request(response):
    span = get_span()
    if span:
      _maybe_inject_trace_id(span, response)
    return response

  @app.teardown_request
  def _teardown_request(exc):
    span = get_span()
    if span:
      span.finish()

  def _decorate_route(tags=None):
    def _wrap(fn):
      def _do():
        parent_span = get_span(request)
        if parent_span:
          if tags:
            for k, v in tags.items():
              parent_span.set_tag(k, v)

          _trace_request(parent_span, request)
        response = fn()

        if parent_span:
          _trace_response(parent_span, response)
        return response
      return _do
    return _wrap

  return _decorate_route
