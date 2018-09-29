import json

from flask import Flask, request

class FissionSpecializationRequest(object):
  def __init__(self, request):
    self._payload = request.json
    self._metadata = self._payload.get('FunctionMetadata', {})
    print("metadata", self._metadata)
    self._annotations = self._metadata.get('annotations', {})
    self._labels = self._metadata.get('labels', {})

  def filepath(self):
    return self._payload['filepath']

  def name(self):
    return self._metadata.get('name', None)

  def jaeger_service_name(self):
    jaeger_service_name = self._annotations.get('jaeger-service-name', None)

    # TODO(adamb) Delete once we migrate existing models to have the jaeger-service-name annotation
    if not jaeger_service_name:
      jaeger_service_name = self._labels.get('ts2-account', None)

    if not jaeger_service_name:
      jaeger_service_name = self.name()

    return jaeger_service_name

  def jaeger_tags(self):
    tags = json.loads(self._annotations.get('jaeger-tags', '{}'))

    # TODO(adamb) Delete once we migrate existing models to have the jaeger-tags annotation
    if 'jaeger-tags' not in self._annotations:
      if 'ts2-account' in self._labels:
        tags['owner'] = self._labels['ts2-account']
      if self.name():
        tags['ancestors.model'] = self.name()

    return tags

  def operation_name_fn(self):
    name = self.name()
    return lambda request, parsed_url: "HTTP %s %s %s" % (name, request.method, parsed_url.path)

class FissionEnvironment(object):
  def __init__(self, specialize):
    self._specialized_app = None
    self._specialize = specialize

    app = Flask(__name__)
    self._app = app

    @app.route('/v2/specialize', methods=['POST'])
    def fission_v2_specialize():
      self._specialized_app = self._specialize(FissionSpecializationRequest(request))
      return ""
    
  def app(self):
    return self._app

  def __call__(self, environ, start_response):
    if self._specialized_app is None:
      return self._app(environ, start_response)

    if 'HTTP_X_FISSION_PARAMS_PATH_INFO' in environ:
      environ['PATH_INFO'] = '/' + environ['HTTP_X_FISSION_PARAMS_PATH_INFO']
    elif 'HTTP_X_FISSION_PARAMS_METHOD' in environ:
      environ['PATH_INFO'] = '/api/' + environ['HTTP_X_FISSION_PARAMS_METHOD']

    return self._specialized_app(environ, start_response)
