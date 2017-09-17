# Modified by Adam Bouhenguel, 2017. https://github.com/Sperlea/cittex/blob/6722879be790bbf5cabd04532cb50bef9ff01fec/arxiv2bibtex.py
# Modified by Theodor Sperlea, 2016. Taken from https://github.com/nathangrigg/arxiv2bib
#
# Copyright (c) 2012, Nathan Grigg
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of this package nor the
#   names of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as
# is" and any express or implied warranties, including, but not limited
# to, the implied warranties of merchantability and fitness for a
# particular purpose are disclaimed. In no event shall Nathan Grigg be
# liable for any direct, indirect, incidental, special, exemplary, or
# consequential damages (including, but not limited to, procurement of
# substitute goods or services; loss of use, data, or profits; or business
# interruption) however caused and on any theory of liability, whether in
# contract, strict liability, or tort (including negligence or otherwise)
# arising in any way out of the use of this software, even if advised of
# the possibility of such damage.
#
# (also known as the New BSD License)
#
# Indiscriminate automated downloads from arXiv.org are not permitted.
# For more information, see http://arxiv.org/help/robots
#
# This script usually makes only one call to arxiv.org per run.
# No caching of any kind is performed.

from xml.etree import ElementTree
import re
import os
from urllib.parse import urlencode
from urllib.request import urlopen


# Namespaces
ATOM = '{http://www.w3.org/2005/Atom}'
ARXIV = '{http://arxiv.org/schemas/atom}'

# regular expressions to check if arxiv id is valid
NEW_STYLE = re.compile(r'^\d{4}\.\d{4,}(v\d+)?$')
OLD_STYLE = re.compile(r"""(?x)
^(
   math-ph
  |hep-ph
  |nucl-ex
  |nucl-th
  |gr-qc
  |astro-ph
  |hep-lat
  |quant-ph
  |hep-ex
  |hep-th
  |stat
    (\.(AP|CO|ML|ME|TH))?
  |q-bio
    (\.(BM|CB|GN|MN|NC|OT|PE|QM|SC|TO))?
  |cond-mat
    (\.(dis-nn|mes-hall|mtrl-sci|other|soft|stat-mech|str-el|supr-con))?
  |cs
    (\.(AR|AI|CL|CC|CE|CG|GT|CV|CY|CR|DS|DB|DL|DM|DC|GL|GR|HC|IR|IT|LG|LO|
      MS|MA|MM|NI|NE|NA|OS|OH|PF|PL|RO|SE|SD|SC))?
  |nlin
    (\.(AO|CG|CD|SI|PS))?
  |physics
    (\.(acc-ph|ao-ph|atom-ph|atm-clus|bio-ph|chem-ph|class-ph|comp-ph|
      data-an|flu-dyn|gen-ph|geo-ph|hist-ph|ins-det|med-ph|optics|ed-ph|
      soc-ph|plasm-ph|pop-ph|space-ph))?
  |math
      (\.(AG|AT|AP|CT|CA|CO|AC|CV|DG|DS|FA|GM|GN|GT|GR|HO|IT|KT|LO|MP|MG
      |NT|NA|OA|OC|PR|QA|RT|RA|SP|ST|SG))?
)/\d{7}(v\d+)?$""")


def is_valid(arxiv_id):
    """Checks if id resembles a valid arxiv identifier."""
    return bool(NEW_STYLE.match(arxiv_id)) or bool(OLD_STYLE.match(arxiv_id))


class FatalError(Exception):
    """Error that prevents us from continuing"""


class NotFoundError(Exception):
    """Reference not found by the arxiv API"""


class Reference(object):
    """Represents a single reference.
    Instantiate using Reference(entry_xml). Note entry_xml should be
    an ElementTree.Element object.
    """
    def __init__(self, entry_xml):
        self.xml = entry_xml
        self.url = self._field_text('id')
        self.id = self._id()
        self.authors = self._authors()
        self.title = self._field_text('title')
        if len(self.id) == 0 or len(self.authors) == 0 or len(self.title) == 0:
            raise NotFoundError("No such publication", self.id)
        self.category = self._category()
        self.year, self.month = self._published()
        self.updated = self._field_text('updated')
        self.bare_id = self.id[:self.id.rfind('v')]
        self.doi = self._field_text('doi', namespace=ARXIV)

    def _authors(self):
        """Extracts author names from xml."""
        xml_list = self.xml.findall(ATOM + 'author/' + ATOM + 'name')
        return [field.text for field in xml_list]

    def _field_text(self, id, namespace=ATOM):
        """Extracts text from arbitrary xml field"""
        try:
            return self.xml.find(namespace + id).text.strip()
        except:
            return ""

    def _category(self):
        """Get category"""
        try:
            return self.xml.find(ARXIV + 'primary_category').attrib['term']
        except:
            return ""

    def _id(self):
        """Get arxiv id"""
        try:
            id_url = self._field_text('id')
            return id_url[id_url.find('/abs/') + 5:]
        except:
            return ""

    def _published(self):
        """Get published date"""
        published = self._field_text('published')
        if len(published) < 7:
            return "", ""
        y, m = published[:4], published[5:7]
        try:
            m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                 "Aug", "Sep", "Nov", "Dec"][int(m) - 1]
        except:
            pass
        return y, m

    def as_dict(self):
        return OrderedDict([
            ("author", self.authors),
            ("title", self.title.split()),
            ("published", self._published()),
            ("updated", self.updated),
            ("doi", self.doi),
            ("journal", "arXiv"),
            ("year", self.year),
            ("month", self.month),
            ("url", self.url),
            ("arxivID", self.id),
        ])

    def bibtex(self):
        """BibTex string of the reference."""
        lines = ["@article{" + self.id]
        for k, v in [("\tauthor", " and ".join(self.authors)),
                    ("\ttitle", " ".join(self.title.split())),
                    ("\tdoi", self.doi),
                    ("\tjournal", "arXiv"),
                    ("\tyear", self.year),
                    ("\tmonth", self.month),
                    ("\turl", self.url),
                    ("\tarxivID", self.id),
                    ]:
            if len(v):
                lines.append("%s = {%s}" % (k, v))

        return ("," + os.linesep).join(lines) + "," + os.linesep + "}"


class ReferenceErrorInfo(object):
    """Contains information about a reference error"""
    def __init__(self, message, id):
        self.message = message
        self.id = id
        self.bare_id = id[:id.rfind('v')]
        # mark it as really old, so it gets superseded if possible
        self.updated = '0'

    def bibtex(self):
        """BibTeX comment explaining error"""
        return "@comment{%(id)s: %(message)s}" % \
                {'id': self.id, 'message': self.message}

    def __str__(self):
        return "Error: %(message)s (%(id)s)" % \
                {'id': self.id, 'message': self.message}


def arxiv2bib(id_list):
    """Returns a list of references, corresponding to elts of id_list"""
    d = arxiv2bib_dict(id_list)
    l = []
    for id in id_list:
        try:
            l.append(d[id])
        except:
            l.append(ReferenceErrorInfo("Not found", id))

    return l


def arxiv_request(ids):
    """Sends a request to the arxiv API."""
    q = urlencode([
         ("id_list", ",".join(ids)),
         ("max_results", len(ids))
         ])
    xml = urlopen("http://export.arxiv.org/api/query?" + q)
    # xml.read() returns bytes, but ElementTree.fromstring decodes
    # to unicode when needed (python2) or string (python3)
    return ElementTree.fromstring(xml.read())


def arxiv2bib_dict(id_list):
    """Fetches citations for ids in id_list into a dictionary indexed by id"""
    ids = []
    d = {}

    # validate ids
    for id in id_list:
        if is_valid(id):
            ids.append(id)
        else:
            d[id] = ReferenceErrorInfo("Invalid arXiv identifier", id)

    if len(ids) == 0:
        return d

    # make the api call
    while True:
        xml = arxiv_request(ids)

        # check for error
        entries = xml.findall(ATOM + "entry")
        try:
            first_title = entries[0].find(ATOM + "title")
        except:
            raise FatalError("Unable to connect to arXiv.org API.")

        if first_title is None or first_title.text.strip() != "Error":
            break

        try:
            id = entries[0].find(ATOM + "summary").text.split()[-1]
            del(ids[ids.index(id)])
        except:
            raise FatalError("Unable to parse an error returned by arXiv.org.")

    # Parse each reference and store it in dictionary
    for entry in entries:
        try:
            ref = Reference(entry)
        except NotFoundError as error:
            message, id = error.args
            ref = ReferenceErrorInfo(message, id)
        if ref.id:
            d[ref.id] = ref
        if ref.bare_id:
            if not (ref.bare_id in d) or d[ref.bare_id].updated < ref.updated:
                d[ref.bare_id] = ref

    return d
