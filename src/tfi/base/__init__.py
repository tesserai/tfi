class _GetAttrAccumulator:
    @staticmethod
    def apply(gaa, target):
        if not isinstance(gaa, _GetAttrAccumulator):
            if isinstance(gaa, dict):
                return {
                    _GetAttrAccumulator.apply(k, target): _GetAttrAccumulator.apply(v, target)
                    for k, v in gaa.items()
                }
            if isinstance(gaa, list):
                return [_GetAttrAccumulator.apply(v, target) for v in gaa]
            return gaa

        result = target
        for fn in gaa._gotten:
            result = fn(target, result)
            result = _GetAttrAccumulator.apply(result, target)
        return result

    def __init__(self, gotten=None, text=None):
        if gotten is None:
            gotten = []
        self._gotten = gotten
        self._text = "" if text is None else text

    def __getitem__(self, item):
        gotten = [
            *self._gotten,
            lambda t, o: o[item],
        ]
        return _GetAttrAccumulator(gotten, "%s[%s]" % (self._text, item))

    def __getattr__(self, name):
        gotten = [
            *self._gotten,
            lambda t, o: getattr(o, name),
        ]
        return _GetAttrAccumulator(gotten, "%s.%s" % (self._text, name))

    def __call__(self, **kw):
        gotten = [
            *self._gotten,
            lambda t, o: o(**{
                k: _GetAttrAccumulator.apply(v, t)
                for k, v in kw.items()
            }),
        ]
        return _GetAttrAccumulator(gotten, "%s(...)" % self._text)

    def __str__(self):
        return "_GetAttrAccumulator<%s>" % self._text

def _recursive_transform(o, fn):
    # First, a shallow tranform.
    o = fn(o)

    # Now a recursive one, if needed.
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
        return o
