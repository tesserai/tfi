import re

from string import Template

# Inspired by https://github.com/distillpub/template/blob/master/components/citation.js
def _bibtex_link_string(ent):
    if "url" not in ent:
        return ""

    url = ent['url']
    arxiv_match = re.search(r'arxiv\.org/abs/([0-9\.v]*)', url)
    if arxiv_match is not None:
        url = "https://arxiv.org/pdf/%s.pdf" % arxiv_match[1]

    label = "link"
    if url.endswith(".pdf"):
        label = "PDF";
    elif url.endswith(".html"):
        label = "HTML";

    return ' &ensp;<a href="%s">[%s]</a>' % (url, label)

def _bibtex_venue_string(ent):
    cite = ent.get('journal', None) or ent.get('booktitle', None) or ""
    if "volume" in ent:
        issue = ent.get('issue', None) or ent.get('number', None)
        issue = "("+issue+")" if issue is not None else ""
        cite += ", Vol " + ent['volume'] + issue;

    if "pages" in ent:
        cite += ", pp. " + ent['pages']

    if cite != "":
        cite += ". "

    if "publisher" in ent:
        cite += ent['publisher']
        if cite[-1] != ".":
            cite += "."

    return cite

def _bibtex_author_string(ent, template, sep, final_sep):
    names = ent['author'].split(" and ");

    def name_str(name):
        name = name.strip()

        if "," in name:
            splitname = name.split(",")
            last = splitname[0].strip()
            firsts = splitname[1]
        else:
            last = name.split(" ")[-1].strip()
            firsts = " ".join(name.split(" ")[:-1])

        initials = ".".join([s.strip()[0] for s in firsts.strip().split(" ")]) + "."
        return Template(template).substitute(F=firsts, L=last, I=initials)

    name_strs = [name_str(name) for name in names]
    if len(name_strs) > 1:
        return sep.join(name_strs[:-1]) + (final_sep or sep) + name_strs[-1]
    else:
        return name_strs[0]

def _bibtex_doi_string(ent, newline=False):
    if "doi" not in ent:
        return ""

    return '%s <a href="https://doi.org/%s" style="text-decoration:inherit;">DOI: %s</a>' % (
        "<br>" if newline else "",
        ent['doi'],
        ent['doi'],
    )

def citation_hover_html(ent):
    cite = ""
    cite += "<b>" + ent['title'] + "</b>"
    cite += _bibtex_link_string(ent)
    cite += "<br>"

    a_str = _bibtex_author_string(ent, "${I} ${L}", ", ", ".")
    v_str = _bibtex_venue_string(ent).strip() + " " + ent['year'] + ". " + _bibtex_doi_string(ent, True)

    if len(a_str+v_str) < min(40, len(ent['title'])):
      cite += a_str + " " + v_str
    else:
      cite += a_str + "<br>" + v_str

    return cite

def citation_bibliography_html(ent):
    if ent is None:
        return "?"

    cite = "<b>" + ent['title'] + "</b> "
    cite += _bibtex_link_string(ent) + "<br>"
    cite += _bibtex_author_string(ent, "${L}, ${I}", ", ", " and ")
    if 'year' in ent or 'date' in ent:
        cite += ", " + (ent['year'] or ent['date']) + ". "
    else:
        cite += ". "

    cite += _bibtex_venue_string(ent)
    cite += _bibtex_doi_string(ent)
    return cite
