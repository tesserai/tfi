from tfi.doc.example_code_generators.util import (
    FetchContext as _FetchContext,
    except_log as _except_log,
)

from tfi.doc.example_code import ExampleCode, ExampleCodeSet
from yapf.yapflib.yapf_api import FormatCode
from yapf.yapflib.style import CreateChromiumStyle
from tfi.format.html.python import html_repr as _python_html_repr

class TensorFlowGrpc(object):
    name = 'tensorflow-grpc-python'
    label = 'gRPC (Python)'
    example_for_class = 'language-python'
    repr_class = 'language-python'
    getting_started_class = 'language-bash'

    def getting_started(self, proto, host):
        return """pip install tensorflow_serving_apis"""
        
    def example_for(self, proto, host, method):
        # need examples as TensorProtos
        inputs = method.example().inputs()

        def shape_of(k):
            pass

        # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/core/example/example.proto
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated


        # .half_val # // DT_HALF, DT_BFLOAT16.
        # .float_val # // DT_FLOAT.
        # .double_val # // DT_DOUBLE.
        # .int_val # // DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
        # .string_val # // DT_STRING
        # # from urllib.request import urlopen
        # # .string_val = urlopen(url).read()
        # .scomplex_val # // DT_COMPLEX64. scomplex_val(2*i) and scomplex_val(2*i+1) are real and imaginary parts of i-th single precision complex.
        # .int64_val # // DT_INT64
        # .bool_val # // DT_BOOL
        # .dcomplex_val # // DT_COMPLEX128. dcomplex_val(2*i) and dcomplex_val(2*i+1) are real and imaginary parts of i-th double precision complex.
        # .uint32_val # // DT_UINT32
        # .uint64_val # // DT_UINT64
  

        # outputs {
        #   key: "classes"
        #   value {
        #     dtype: DT_STRING
        #     tensor_shape {
        #       dim {
        #         size: 1
        #       }
        #       dim {
        #         size: 1
        #       }
        #     }
        #     string_val: "unused background"
        #   }
        # }
        # outputs {
        #   key: "scores"
        #   value {
        #     dtype: DT_FLOAT
        #     tensor_shape {
        #       dim {
        #         size: 1
        #       }
        #       dim {
        #         size: 1
        #       }
        #     }
        #     float_val: 0.7362784147262573
        #   }
        # }
        # model_spec {
        #   name: "oedema"
        #   version {
        #     value: 1
        #   }
        #   signature_name: "predict_images"
        # }


        request_inputs = "".join([
            """request.inputs['%s'].CopyFrom(\
tf.contrib.util.make_tensor_proto(%r, shape=%r))\n""" % (k, v, shape_of(k))
            for k, v in inputs.items()
        ])

        pa = _FetchContext(proto, host)
        if 'images' in inputs:
            f = pa.consider(inputs['images'][0])
            if f:
                request_inputs = """
def tensor(tp, dtype, shape=[]):
  tp.dtype = dtype
  for size in shape: tp.tensor_shape.dim.add(size=size)
  return tp

from urllib.request import urlopen

def append_fetched_string_val(tp, url, shape=[]):
  with urlopen(url) as f:
    tensor(tp, dtype=7, shape=shape).string_val.append(f.read())

inputs_url = "{url}"
append_fetched_string_val(request.inputs['images'], inputs_url, shape=[1])
""".format(url=f['url']) # put in fetch url

        style_config = CreateChromiumStyle()
        style_config.update({'COLUMN_LIMIT': 60})

        def _styled(s):
            s, _ = FormatCode(s, style_config=style_config)
            return s.strip()

        return ExampleCodeSet(
            examples=[
                ExampleCode(
                    name='tensor-proto',
                    label='',
                    lines="""from tensorflow_serving.apis import (predict_pb2, prediction_service_pb2_grpc)
import grpc

request = predict_pb2.PredictRequest()
request.model_spec.signature_name = '{method_name}'
{request_inputs}

ch = grpc.insecure_channel("{host}")
stub = prediction_service_pb2_grpc.PredictionServiceStub(ch)
result = stub.Predict(request, 10.0)
print(result)\n""".format(
                        proto=proto,
                        host=host,
                        method_name=method.name(),
                        request_inputs=_styled(request_inputs),
                    ).split("\n"),
                ),
            ],
        )

    @_except_log
    def repr(self, v, max_width=None, max_seq_length=None):
        return _python_html_repr(v, max_width, max_seq_length)
