from tfi.doc.example_code_generators.json import Json as _Json
from tfi.doc.example_code_generators.tensorflow_grpc import TensorFlowGrpc as _TensorFlowGrpc

_EXAMPLE_CODE_GENERATORS = {}
def example_code_generator(name):
    return _EXAMPLE_CODE_GENERATORS[name]

def _register_code_generator(ex):
    _EXAMPLE_CODE_GENERATORS[ex.name] = ex


_register_code_generator(_Json())
# _register_code_generator(_Python())
_register_code_generator(_TensorFlowGrpc())

