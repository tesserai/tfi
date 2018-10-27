from collections import OrderedDict

import inspect
import re
import importlib

import tinydb
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

class BibtexRepo(object):
    def __init__(self, path, resolvers):
        storage = CachingMiddleware(JSONStorage)
        self._db = tinydb.TinyDB(path, storage=storage)
        self._resolvers = resolvers
        self._flush = storage.flush

    def close(self):
        self._db.close()

    def resolve(self, **kw):
        RefQuery = tinydb.Query()
        results = []
        needed = []
        ix = 0
        for key, ids in kw.items():
            for id in ids:
                found = self._db.search(getattr(RefQuery, key) == id)
            if len(found) > 0:
                results.append(found[0]['bibtex'])
            else:
                results.append(None)
                needed.append((key, id, ix))
            ix += 1

        if len(needed) > 0:
            resolved = [ref for ref in self._resolvers[key]([id]) for key, id, ix in needed]
            for reference, (key, id, ix) in zip(resolved, needed):
                bibtex = reference.bibtex()
                self._db.insert(OrderedDict([(key, id), ('bibtex', bibtex)]))
                results[ix] = bibtex
            self._flush()

        return results
