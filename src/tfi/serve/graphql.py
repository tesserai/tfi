from flask_graphql import GraphQLView as _GraphQLView
from graphql_server import GraphQLParams as _GraphQLParams

from tfi.serve.graphiql import render_graphiql as _render_graphiql

from graphql import (
    GraphQLScalarType,
    GraphQLSchema,
    GraphQLObjectType,
    GraphQLNonNull,
    GraphQLArgument,
    GraphQLField,
    GraphQLString,
    GraphQLFloat,
    GraphQLInt,
    GraphQLBoolean,
    GraphQLList,
    GraphQLUnionType,
)

from graphql.language.ast import StringValue


import inspect
import base64 as _base64

from tensorflow.core.framework import types_pb2

def _parse_base64_literal(ast):
    # type: (Union[StringValue]) -> Optional[str]
    if isinstance(ast, StringValue):
        return _base64.b64decode(ast.value)

    return None

GraphQLBase64Type = GraphQLScalarType(
    name="Base64",
    description="The `Base64` scalar type represents Base64-encoded bytes.",
    serialize=lambda s: _base64.b64encode(s).encode('utf-8'),
    parse_value=lambda s: _base64.b64decode(s),
    parse_literal=_parse_base64_literal,
)

_graphql_types_for_dtype = {
  types_pb2.DT_HALF: GraphQLFloat,
  types_pb2.DT_BFLOAT16: GraphQLFloat,
  types_pb2.DT_FLOAT: GraphQLFloat,
  types_pb2.DT_DOUBLE: GraphQLFloat,
  types_pb2.DT_INT32: GraphQLInt,
  types_pb2.DT_INT16: GraphQLInt,
  types_pb2.DT_INT8: GraphQLInt,
  types_pb2.DT_UINT8: GraphQLInt,
  types_pb2.DT_STRING: GraphQLUnionType(
    name='BytesList',
    types=[GraphQLString,GraphQLBase64Type],
    resolve_type=lambda v: GraphQLBase64Type,
  ),
  types_pb2.DT_INT64: GraphQLInt,
  types_pb2.DT_UINT32: GraphQLInt,
  types_pb2.DT_UINT64: GraphQLInt,
  types_pb2.DT_BOOL: GraphQLBoolean,
}

def _graphql_type_for(tensor_info):
  graphql_type = _graphql_types_for_dtype[tensor_info.dtype]

  for _ in range(len(tensor_info.tensor_shape.dim)):
    graphql_type = GraphQLList(graphql_type)

  return GraphQLNonNull(graphql_type)

def _make_query(method_name, method):
  # HACK(adamb) Should be doing something better than this.
  inputs = {
    k: v
    for k, v in method.__annotations__.items()
    if k != 'return'
  }
  outputs = method.__annotations__['return']

  outputs_object_type = GraphQLObjectType(
    name="outputs_%s" % method_name,
    fields={
      k: GraphQLField(
        description=None,
        type=_graphql_type_for(v),
      )
      for k, v in outputs.items()
    }
  )

  return GraphQLField(
      type=outputs_object_type,
      args={
        k: GraphQLArgument(
          description=None,
          type=_graphql_type_for(v),
        )
        for k, v in inputs.items()
      },
      resolver=lambda _1, _2, **kw: method(**kw),
    )

def _make_schema(model):
  return GraphQLSchema(
    query=GraphQLObjectType(
      name='Root',
      fields={
        method_name: _make_query(method_name, method)
        for method_name, method in inspect.getmembers(model, predicate=inspect.ismethod)
        if not method_name.startswith('_')
      },
    ), 
  )

def add_endpoints(model, app):
  schema = _make_schema(model)

  app.add_url_rule(
    '/graphql',
    view_func=_GraphQLView.as_view('graphql', schema=schema),
  )

  # app.add_url_rule(
  #   '/graphql/batch',
  #   view_func=_GraphQLView.as_view('graphql', schema=schema, batch=True),
  # )

  @app.route('/graphiql', methods=['GET'])
  def graphiql():
      return _render_graphiql(
        graphiql_version='0.11.11', 
        graphiql_html_title=None, 
        result='null', 
        params=_GraphQLParams(None, None, None), 
        requests_endpoint='/graphql')