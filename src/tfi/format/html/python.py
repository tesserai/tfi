import re
import base64

from tfi.format.pretty import pretty
from tfi.tensor.codec import encode

class escaped(object):
    def __init__(self, s):
        self.s = s
    def __repr__(self):
        return self.s

def img_png_html(d):
    return """<img src="data:image/png;base64,%s">""" % base64.b64encode(d).decode()

def text_plain(d):
    print("text_plain", d)
    return d

def _recursive_transform(o, fn):
    if isinstance(o, dict):
        return {
            k: _recursive_transform(v, fn)
            for k, v in o.items()
        }
    elif isinstance(o, list):
        return [
            _recursive_transform(e, fn)
            for e in o
        ]
    else:
        return fn(o)

def inspect(o, max_width=None, max_seq_length=None):
    # TODO(adamb) Need to do the right thing here...
    replacements = {}
    def replaceall(s):
        for k, v in replacements.items():
            s = s.replace(k, v)
        return s

    prefix_pattern = "alsdkfjaslkdfj%s"
    prefixix = 0
    def genreplacement(v):
        nonlocal prefixix
        k = prefix_pattern % prefixix
        prefixix += 1
        replacements[k] = v
        return escaped(k)

    def genpng(v):
        return genreplacement(img_png_html(v))


    accept_mimetypes = {"image/png": genpng, "text/plain": text_plain}

    val = _recursive_transform(o, lambda x: encode(accept_mimetypes, x))
    if val is not None:
        o = val

    # return '%r' % val
    if max_width is None:
        max_width = 79
    if max_seq_length is None:
        max_seq_length = 1000
    r = pretty(o, max_width=max_width, max_seq_length=max_seq_length)
    xform = replaceall
    return r, xform
    # return '%r' % o

inspect_html = inspect
