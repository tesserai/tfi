import urllib.request
from urllib.error import HTTPError

_BASE_URL = 'http://dx.doi.org/'

class _DoiReference(object):
    def __init__(self, bibtex):
        self._bibtex = bibtex

    def bibtex(self):
        return self._bibtex

def _doi_request(doi):
    print("_doi_request", doi)
    url = _BASE_URL + doi
    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/x-bibtex')

    try:
        with urllib.request.urlopen(req) as f:
            return _DoiReference(f.read().decode())
    except HTTPError:
        raise Exception("Unable to resolve: %s" % doi)

def doi2bib(doi_list):
    return [
        _doi_request(doi)
        for doi in doi_list
    ]
