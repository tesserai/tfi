from jinja2 import Template as JinjaTemplate

from tfi.json import as_jsonable as _as_jsonable

from tfi.format.html.bibtex import citation_bibliography_html
from tfi.format.html.python import inspect_html
from tfi.format.html.rst import parse_rst as _parse_rst

from tfi.parse.python import parse_example_args as _parse_example_args

from yapf.yapflib.yapf_api import FormatCode
from yapf.yapflib.style import CreateGoogleStyle

from collections import OrderedDict

import json
import shlex
import os.path as _os_path

import tfi.tensor.codec
from tfi.base import _recursive_transform

_page_template_path = __file__[:-2] + "html"
if _page_template_path.endswith("__init__.html"):
    _page_template_path = _page_template_path.replace("__init__.html", __name__.split('.')[-1] + '.html')

def _read_template_file(subpath):
    path = _os_path.join(_os_path.dirname(__file__), subpath)
    with open(path, encoding="utf-8") as f:
        return f.read()

def render(
        source,
        authors,
        title,
        overview,
        methods,
        hyperparameters,
        implementation_notes,
        references,
        include_snapshot,
        proto,
        host,
        extra_scripts=""):

    with open(_page_template_path, encoding='utf-8') as f:
        t = JinjaTemplate(f.read())

    def python_example_for(method, more_style_config={}):
        s = "m.%s(%s)" % (method['name'], ", ".join(
            [
                "%s=%r" % (k, v)
                for k, v in method['example args'].items()
            ]
        ))

        style_config = CreateGoogleStyle()
        style_config.update(more_style_config)
        s, _ = FormatCode(s, style_config=style_config)
        s = s.strip()
        return s

    def json_repr(v, max_width=None, max_seq_length=None):
        jsonable = _as_jsonable(v)
        return json.dumps(jsonable)

    def python_repr(v, max_width=None, max_seq_length=None):
        s, xform = inspect_html(v, max_width=max_width, max_seq_length=max_seq_length)
        return xform(s)

    def curl_example_for(method_name, example_args):
        def json_default(o):
            if not hasattr(o, '__json__'):
                raise TypeError("Unserializable object {} of type {}".format(o, type(o)))
            return o.__json__()

        def quoted_json_dumps(v):
            try:
                s = json.dumps(v, default=json_default)
                return shlex.quote(s)
            except ValueError as ex:
                print(ex)
                return None

        return "curl %s" % " \\\n   ".join([
            "%s://%s/api/%s" % (proto, host, method_name),
            *[
                "-F %s=%s" % (k, quoted_json_dumps(v))
                for k, v in example_args.items()
            ],
        ])

    def curl_example_for_method(method):
        return curl_example_for(method['name'], method['example args'])

    def html_repr(v):
        s, xform = inspect_html(v, max_width=None, max_seq_length=None)
        return xform(s)

    citation_id = 0
    citation_label_by_refname = {}
    def citation_label_by_refname_fn(citation_refname):
        nonlocal citation_id
        if citation_refname not in citation_label_by_refname:
            citation_id += 1
            citation_label_by_refname[citation_refname] = str(citation_id)
        return citation_label_by_refname[citation_refname]

    def reference_fn(citation_refname):
        return references[citation_refname]

    def demo_block_html(method_name, method_doc):
        code = curl_example_for(
                method_name,
                _parse_example_args(method_doc['example args'], {}))

        return """<div class="method-example">
<div class="method-example-code">
    <pre><code class="language-curl line-numbers">%s</code></pre>
</div></div>""" % code

    # TODO(adamb) What about arxiv ids already within []_ ??
    parsed = _parse_rst(overview or "", "<string>", 2, 'overview-',
            citation_label_by_refname_fn, reference_fn, demo_block_html)
    if not parsed['title']:
        parsed['title'] = title

    def refine_method(method):
        method = dict(method)
        parsed_overview = _parse_rst(method['overview'], "<string>", 3,
                'method-%s-' % method['name'],
                citation_label_by_refname_fn, reference_fn, demo_block_html)
        method['overview'] = parsed_overview['body']
        return method

    methods = [
        refine_method(method)
        for method in methods
    ]

    def visible_text_for(html):
        import re
        from bs4 import BeautifulSoup
        from bs4.element import Comment, NavigableString
        
        def visible(element):
            if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
                return False
            elif isinstance(element, Comment):
                return False

            # Hidden if any parent has inline style "display: none"
            parent = element.parent
            while parent:
                if 'style' in parent.attrs and re.match('display:\\s*none', parent['style']):
                    return False
                parent = parent.parent
            return True
        
        soup = BeautifulSoup(html, "html.parser")
        data = soup.findAll(text=True)
        return re.sub('\\s+', ' ', " ".join(t.strip() for t in filter(visible, data)))

    description = visible_text_for(parsed['subtitle'])

    body_sections = []
    if parsed['body']:
        forged_body_title = 'Overview'
        forged_body_id = 'overview'
        body_sections.append({
            'title': forged_body_title,
            'id': forged_body_id,
            'body': """<section id="%s"><h2>%s</h2>%s</section>""" % (
                forged_body_id,
                forged_body_title,
                parsed['body']
            ),
        })

    appendix_section_ids = ['overview-dataset']
    appendix_sections = []
    for section in parsed['sections']:
        if section['id'] in appendix_section_ids:
            appendix_sections.append(section)
        else:
            body_sections.append(section)

    languages = [
        {
            'language': 'curl',
            'label': 'curl (remote)',
            'example_for': curl_example_for_method,
            'example_for_class': 'language-curl',
            'repr': json_repr,
            'repr_class': 'language-json2',
            'getting_started_class': 'language-bash',
            'getting_started': """curl %s://%s/ok""" % (proto, host)
        },
    ]

    if include_snapshot:
        languages.append(
            {
                'language': 'python',
                'label': 'Python (local)',
                'example_for': lambda method: python_example_for(method, {'COLUMN_LIMIT': 50}),
                'example_for_class': 'language-python',
                'repr': python_repr,
                'repr_class': 'language-python',
                'getting_started_class': 'language-bash',
                'getting_started': """pip install tfi
    tfi %s://%s""" % (proto, host)
            },
        )

    return t.render(
            read_template_file=_read_template_file,
            languages=languages,
            html_repr=html_repr,
            include_snapshot=include_snapshot,
            extra_scripts=extra_scripts,
            title=parsed['title'],
            subhead=parsed['subtitle'],
            description=description,
            bibliography_cite=citation_bibliography_html,
            overview=parsed['body'],
            body_sections=body_sections,
            appendix_sections=appendix_sections,
            authors=authors,
            hyperparameters=hyperparameters,
            implementation_notes=implementation_notes,
            references=OrderedDict([
                (refname, references[refname])
                for refname in citation_label_by_refname
            ]),
            proto=proto,
            host=host,
            methods=methods)
