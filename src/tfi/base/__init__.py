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
