import numpy as np
from collections import OrderedDict

class TensorFrame(object):
    @staticmethod
    def _zip(data, dimlabels, value_fn, dictclass):
        ready = dictclass() # Post-zip
        willzip = OrderedDict()
        lengths = {}
        for shape, name, tensor in data:
            if shape is None or shape == (...,):
                ready[name] = value_fn(tensor)
                continue
            if len(shape) == 0: # Scalars can't be zipped, they're ready as-is.
                ready[name] = value_fn(tensor.item()) if hasattr(tensor, 'item') else tensor
                continue
            dimname = shape[0]
            if dimname is None:
                ready[name] = value_fn(tensor)
                continue
            if dimname not in willzip:
                willzip[dimname] = []
                lengths[dimname] = tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)

            willzip[dimname].append((shape[1:], name, tensor))

        hassolokey = len(willzip) == 1
        for dimname, nested in willzip.items():
            readykey = dimlabels[dimname] if dimname in dimlabels else dimname
            ready[readykey] = [
                TensorFrame._zip(
                    [
                        (shape, name, tensor[ix, ...] if isinstance(tensor, np.ndarray) else tensor[ix])
                        for shape, name, tensor in nested
                    ],
                    dimlabels,
                    value_fn,
                    dictclass,
                )
                for ix in range(lengths[dimname])
            ]

        if not ready:
            return []

        if len(ready) != 1:
            hassolokey = False

        # If there's only one thing, don't nest it.
        if hassolokey:
            return list(ready.values())[0]

        return ready

    def __init__(self, *args, **kwargs):
        self._data = args
        self._data_dict = {
            name: tensor
            for shape, name, tensor in self._data
        }
        self._shape_dict = {
            name: shape
            for shape, name, tensor in self._data
        }
        self._shape_labels = kwargs

    def __contains__(self, k):
        return k in self._data_dict

    def __getitem__(self, ix):
        return self._data_dict[ix]

    def dict(self):
        return self._data_dict

    def shapes(self):
        return self._shape_dict

    def shape_labels(self):
        return self._shape_labels

    def tuples(self):
        return self._data

    def items(self):
        return self._data_dict.items()

    def keys(self):
        return self._data_dict.keys()

    def zipped(self, *, filter=None, dictclass=None):
        data = self._data
        if filter:
            data = [
                (shape, name, tensor)
                for shape, name, tensor in self._data
                if name in filter or next((True for k, v in self._shape_labels.items() if v in filter and k in shape), False)
            ]

        if not dictclass:
            dictclass = dict

        return TensorFrame._zip(data, self._shape_labels, value_fn=lambda x: x, dictclass=dictclass)

    # def __repr__(self):
    #     return self.zipped().__repr__()