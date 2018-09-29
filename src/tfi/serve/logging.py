import datetime
import json
import logging
import sys
import time
import urllib.parse

from flask import request

def log_requests_logrus(app):
  start = None
  @app.before_request
  def start_timer():
    nonlocal start
    start = time.time()

  @app.after_request
  def log_request(response):
    if request.path == '/favicon.ico':
      return response
    elif request.path.startswith('/static'):
      return response
    
    now = time.time()

    log_params = [
      ('method', request.method),
      ('path', request.path),
      ('status', response.status_code),
      ('duration', round(now - start, 4) if start else None),
      ('time', datetime.datetime.fromtimestamp(now).isoformat('T') + 'Z'),
      ('ip', request.headers.get('X-Forwarded-For', request.remote_addr)),
      ('host', request.host.split(':', 1)[0]),
      ('params', dict(request.args)),
    ]
    request_id = request.headers.get('X-Request-ID')
    if request_id:
        log_params.append(('request_id', request_id))

    line = " ".join(
      "{}={}".format(name, value)
      for name, value in log_params
    )
    if response.status_code < 400:
      app.logger.info(line)
    else:
      app.logger.warn(line)
    return response
