def except_log(fn):
    def _do(*a, **kw):
        print("_do", fn, a, kw)
        try:
            return fn(*a, **kw)
        except Exception as ex:
            print(ex)
            import traceback
            traceback.print_exc()
    return _do

class FetchContext(object):
    def __init__(self, proto, host):
        self._proto = proto
        self._host = host
    
    def consider(self, o):
        if not hasattr(o, '__fetchable__'):
            return None

        f = dict(o.__fetchable__())
        f['url'] = '%s://%s/%s' % (self._proto, self._host, f['urlpath'])
        return f
