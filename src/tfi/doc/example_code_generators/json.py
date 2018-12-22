from tfi.doc.example_code import ExampleCode, ExampleCodeSet

import json
import tfi.json
import shlex
from tfi.base import _recursive_transform

from tfi.doc.example_code_generators.util import (
    FetchContext as _FetchContext,
    except_log as _except_log,
)

def _pretty_json(o):
    return json.dumps(
        json.loads(tfi.json.dumps(o)),
        indent=2,
        separators=(',', ': '),
        sort_keys=True,
    )

class Json(object):
    name = 'json'
    label = 'JSON'
    example_for_class = 'language-curl'
    repr_class = 'language-json2'
    getting_started_class = 'language-bash'

    def getting_started(self, proto, host):
        return """curl %s://%s/ok""" % (proto, host)

    @_except_log
    def repr(self, v, max_width=None, max_seq_length=None):
        return tfi.json.dumps(v, coerce=True)

    def example_for(self, proto, host, method):
        method_name = method.name()
        example_args = method.example().inputs()

        if not example_args:
            return "curl %s" % " \\\n   ".join([
                "%s://%s/api/%s" % (proto, host, method_name),
                "-d '{}'",
            ])

        pa = _FetchContext(proto, host)
        prefetch_commands = []
        prefetch_replacements = [('$', '\\$')]
        ref_replacements = []
        def _gensym(length=16):
            import random
            return ''.join(random.sample('abcdefghijklmnopqrstuvwxyz', length))

        def _replace_fetchable_refs(o):
            f = pa.consider(o)
            if not f or not hasattr(o, '__json__'):
                return o

            j = o.__json__()
            sym = _gensym()
            prefetch_replacement = None
            ref_replacement = None
            if j.get('$encode', None) == 'base64':
                prefetch_replacement = '"$(base64 < ./%s)"' % f['basename']
            else:
                ref_replacement = '{"$ref": "%s"}' % f['url']
                if f['mimetype']:
                    prefetch_replacement = '{"\\$mimetype": "%s", "\\$base64": "$(base64 < ./%s)"}' % (f['mimetype'], f['basename'])
                else:
                    prefetch_replacement = '{"\\$base64": "$(base64 < ./%s)"}' % (f['basename'])

            if prefetch_replacement:
                prefetch_commands.append('curl -sO %s' % f['url'])
                prefetch_replacements.append(('"%s"' % sym, prefetch_replacement))
            if ref_replacement:
                ref_replacements.append(('"%s"' % sym, ref_replacement))
            return sym

        def _swap(o, replacements):
            for search, replace in replacements:
                o = o.replace(search, replace)
            return o

        def _escape(o):
            if '$(' in o:
                return "\n".join([
                    "@- <<JSON",
                    o,
                    "JSON",
                    "",
                ])
            if '\n' in o:
                return "\n".join([
                    "@- <<'JSON'",
                    o,
                    "JSON",
                    "",
                ])

            return shlex.quote(o)

        example_json = _pretty_json(_recursive_transform(example_args, _replace_fetchable_refs))
        examples = []
        if ref_replacements:
            examples.append(
                ExampleCode(
                    name='ref-data',
                    label='Using data from the web',
                    lines=[
                        "\n".join([
                            "curl %s" % (
                                " \\\n   ".join([
                                    "%s://%s/api/%s" % (proto, host, method_name),
                                    "-H 'Content-Type: application/json'",
                                    "-d " + _escape(_swap(example_json, ref_replacements)),
                                ]),
                            ),
                        ]),
                    ],
                )
            )

        if prefetch_commands:
            examples.append(
                ExampleCode(
                    name='inline-data',
                    label='Using inline data',
                    lines=[
                        *prefetch_commands,
                        "\n".join([
                            "curl %s" % (
                                " \\\n   ".join([
                                    "%s://%s/api/%s" % (proto, host, method_name),
                                    "-H 'Content-Type: application/json'",
                                    "-d " + _escape(_swap(example_json, prefetch_replacements)),
                                ]),
                            ),
                        ]),
                    ],
                )
            )
        
        return ExampleCodeSet(examples)
