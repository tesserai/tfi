from tfi.doc.example_code_generators.util import (
    except_log as _except_log,
)
from yapf.yapflib.yapf_api import FormatCode
from yapf.yapflib.style import CreateChromiumStyle
from tfi.format.html.python import html_repr as _python_html_repr

from tfi.doc.example_code import ExampleCode, ExampleCodeSet

class Python(object):
    name = 'python'
    label = 'Python (local)'
    example_for_class = 'language-python'
    repr_class = 'language-python'
    getting_started_class = 'language-bash'

    def getting_started(self, proto, host):
        return """pip install tfi
    tfi %s://%s""" % (proto, host)

    @_except_log
    def repr(self, v, max_width=None, max_seq_length=None):
        return _python_html_repr(v, max_width, max_seq_length)

    def example_for(self, proto, host, method):
        s = "m.%s(%s)" % (method.name(), ", ".join(
            [
                "%s=%r" % (k, v)
                for k, v in method.example().inputs().items()
            ]
        ))

        style_config = CreateChromiumStyle()
        style_config.update({'COLUMN_LIMIT': 50})
        s, _ = FormatCode(s, style_config=style_config)
        s = s.strip()
        return ExampleCodeSet(
            examples=[
                ExampleCode(
                    name='python',
                    label='Python',
                    lines=[s],
                ),
            ],
        )
