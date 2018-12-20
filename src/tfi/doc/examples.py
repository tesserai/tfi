import json
import tfi.json
import shlex
import os
from tfi.base import _recursive_transform
from yapf.yapflib.yapf_api import FormatCode
from yapf.yapflib.style import CreateGoogleStyle
from tfi.format.html.python import html_repr as _python_html_repr

def _except_log(fn):
    def _do(*a, **kw):
        print("_do", fn, a, kw)
        try:
            return fn(*a, **kw)
        except Exception as ex:
            print(ex)
            import traceback
            traceback.print_exc()
    return _do

class PreludeAuthor(object):
    def __init__(self, proto, host):
        self._proto = proto
        self._host = host
    
    def consider(self, o):
        if not hasattr(o, '__fetchable__'):
            return None

        f = dict(o.__fetchable__())
        f['url'] = '%s://%s/%s' % (self._proto, self._host, f['urlpath'])
        return f

def _pretty_json(o):
    return json.dumps(
        json.loads(tfi.json.dumps(o)),
        indent=2,
        separators=(',', ': '),
        sort_keys=True,
    )

class SingleExample(object):
    def __init__(self, name, label, lines):
        self.name = name
        self.label = label
        self.lines = lines

class MultipleExamples(object):
    def __init__(self, examples):
        self.examples = examples

class JsonExemplar(object):
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

        pa = PreludeAuthor(proto, host)
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
                prefetch_replacement = '"$(cat ./%s | base64)"' % f['basename']
            else:
                ref_replacement = '{"$ref": "%s"}' % f['url']
                if f['mimetype']:
                    prefetch_replacement = '{"\\$mimetype": "%s", "\\$base64": "$(cat ./%s | base64)"}' % (f['mimetype'], f['basename'])
                else:
                    prefetch_replacement = '{"\\$base64": "$(cat ./%s | base64)"}' % (f['basename'])

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
                SingleExample(
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
                SingleExample(
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
        
        return MultipleExamples(examples)
_LANGUAGE_IMPLS = {}
def _add_exemplar(ex):
    _LANGUAGE_IMPLS[ex.name] = ex


_add_exemplar(JsonExemplar())

def resolve_language(name):
    return _LANGUAGE_IMPLS[name]
