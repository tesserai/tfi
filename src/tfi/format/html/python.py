import re
import base64

from io import BytesIO
from PIL import Image

from tfi.base import _recursive_transform
from tfi.format.pretty import pretty
from tfi.tensor.codec import encode

class escaped(object):
    def __init__(self, s):
        self.s = s
    def __repr__(self):
        return self.s

def img_png_html(d):
    if False:
        return """<img src="data:image/png;base64,%s">""" % base64.b64encode(d).decode()

    width, height = Image.open(BytesIO(d)).size
    # TODO(adamb) Actually host the image itself somewhere, don't keep using data URIs
    datauri = "data:image/png;base64,%s" % base64.b64encode(d).decode()
    small = datauri
    large = datauri

    return """<figure style="display: inline-block; margin: 0" itemprop="associatedMedia" itemscope itemtype="http://schema.org/ImageObject">
  <a href="%s" itemprop="contentUrl" data-size="%sx%s">
    <img src="%s" itemprop="thumbnail" alt="Image description" />
  </a>
  <figcaption itemprop="caption description" style="display: none;"></figcaption>
</figure>""" % (large, width, height, small)

    
def text_plain(d):
    print("text_plain", d)
    return d

def html_repr(obj, max_width=None, max_seq_length=None):
    # TODO(adamb) Need to do the right thing here...
    replacements = {}

    prefix_pattern = "alsdkfjaslkdfj%s"
    prefixix = 0
    def genreplacement(v):
        nonlocal prefixix
        k = prefix_pattern % prefixix
        prefixix += 1
        replacements[k] = v
        return escaped(k)

    image_count = 0
    def genpng(v):
        nonlocal image_count
        image_count += 1
        return genreplacement(img_png_html(v))

    accept_mimetypes = {"image/png": genpng, "text/plain": text_plain}

    def xform_or_none(v):
        t = encode(accept_mimetypes, v)
        return v if t is None else t
    xformed = _recursive_transform(obj, xform_or_none)

    if max_width is None:
        max_width = 79
    if max_seq_length is None:
        max_seq_length = 1000
    r = pretty(xformed, max_width=max_width, max_seq_length=max_seq_length)

    # If there are images in this expression, make them zoomable.
    if image_count:
        r = """<div itemscope itemtype="http://schema.org/ImageGallery">%s</div>""" % r

    for k, v in replacements.items():
        r = r.replace(k, v)

    return r
