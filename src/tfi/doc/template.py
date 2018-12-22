import base64
from collections import OrderedDict
from jinja2 import Template as JinjaTemplate
from tfi.format.html.bibtex import citation_bibliography_html
from tfi.format.html.python import html_repr
from tfi.format.html.rst import parse_rst as _parse_rst
from tfi.parse.html import visible_text_for as _visible_text_for
from tfi.doc.example_code_generators import example_code_generator as _resolve_language

_page_template_path = __file__[:-2] + "html"
if _page_template_path.endswith("__init__.html"):
    _page_template_path = _page_template_path.replace("__init__.html", __name__.split('.')[-1] + '.html')

class HtmlRenderer(object):
    def __init__(self, documentation, include_snapshot, extra_scripts=""):
        d = documentation
        self._citation_id = 0
        self._citation_label_by_refname = {}
        self._references = d.references()
        self._include_snapshot = include_snapshot

        # TODO(adamb) What about arxiv ids already within []_ ??
        parsed = self._rst_to_html(d.overview() or "", 2, 'overview-')

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

        # Sort and render method overviews here so that references are properly ordered
        methods = sorted(d.methods(), key=lambda method: method.name())
        method_overviews = {
            method.name(): self._rst_to_html(
                method.overview(),
                initial_header_level=3,
                id_prefix='method-%s-' % method.name(),
            )['body']
            for method in methods
        }

        language_names = ['json', 'tensorflow-grpc-python']
        # if include_snapshot:
        #     language_names.append('python')

        self._template_fields = {
            'facets_overview_proto_base64': base64.b64encode(d.facets_overview_proto()).decode('utf-8') if d.facets_overview_proto() else None,
            'title': parsed['title'] or d.name(),
            'rst_to_html': self._rst_to_html,
            'html_repr': html_repr,
            'include_snapshot': include_snapshot,
            'extra_scripts': extra_scripts,
            'subhead': parsed['subtitle'],
            'description': _visible_text_for(parsed['subtitle']),
            'bibliography_cite': citation_bibliography_html,
            'overview': parsed['body'],
            'body_sections': body_sections,
            'appendix_sections': appendix_sections,
            'authors': d.authors(),
            'hyperparameters': [
                (name, " ".join(["value", "was", repr(value)]), doc)
                for name, _, value, doc in d.hyperparameters()
            ],
            'implementation_notes': d.implementation_notes(),
            'references': OrderedDict([
                (refname, self._references[refname])
                for refname in self._citation_label_by_refname
            ]),
            'method_overviews': method_overviews,
            'languages': [
                _resolve_language(language_name)
                for language_name in language_names
            ],
            'methods': methods,
        }

    def _citation_label_by_refname_fn(self, citation_refname):
        if citation_refname not in self._citation_label_by_refname:
            self._citation_id += 1
            self._citation_label_by_refname[citation_refname] = str(self._citation_id)
        return self._citation_label_by_refname[citation_refname]

    def _rst_to_html(self, rst, id_prefix, initial_header_level):
        if not rst:
            return {
                'body': '',
                'sections': {},
                'title': '',
                'subtitle': '',
            }

        return _parse_rst(
            rst,
            source_path="<string>",
            initial_header_level=initial_header_level,
            id_prefix=id_prefix,
            citation_label_by_refname=self._citation_label_by_refname_fn,
            bibtex_entries_by_refname=lambda citation_refname: self._references[citation_refname],
        )

    def render(self, proto, host):
        with open(_page_template_path, encoding='utf-8') as f:
            t = JinjaTemplate(f.read())

        return t.render(
            proto=proto,
            host=host,
            **self._template_fields,
        )
