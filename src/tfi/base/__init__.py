class _GetAttrAccumulator:
    def __init__(self, gotten=None):
        if gotten is None:
            gotten = []
        self._gotten = gotten

    def __getattr__(self, name):
        return _GetAttrAccumulator([*self._gotten, name])

    def __call__(self, target):
        result = target
        for name in self._gotten:
            result = getattr(result, name)
        return result

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
