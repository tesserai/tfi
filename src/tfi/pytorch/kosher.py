import types
import pickle
import torch.serialization
import struct

from collections import OrderedDict
from io import StringIO

import cloudpickle

class KosherUnpickler(pickle.Unpickler):
    def __init__(self, file, *, fix_imports=True, encoding="ASCII", errors="strict"):
        super(KosherUnpickler, self).__init__(file, fix_imports=fix_imports, encoding=encoding, errors=errors)
        self.module_fields = OrderedDict()

    def load(self):
        def _make_mod(name, vars):
            mod = imp.new_module(name)
            mod.__dict__.update(vars)
            sys.modules[name] = mod
            return mod
        self.module_fields = OrderedDict([
            (k, _make_mod(k, v))
            for k, v in super(KosherUnpickler, self).load().items()
        ])

        r = super(KosherUnpickler, self).load()
        return r

    def find_class(self, module, name):
        if module in self.module_fields:
            mod = self.module_fields[module]
            return mod.__dict__[name]
        return super(KosherUnpickler, self).find_class(module, name)

class PickleModule(object):
    def __init__(self, inline_module_pred):
        self._inline_module_pred = inline_module_pred

    def Unpickler(self, file):
        return KosherUnpickler(file)

    def load(self, file):
        return self.Unpickler(file).load()

    def loads(self, s):
        file = StringIO(s)
        try:
            return self.Unpickler(file).load()
        finally:
            file.close()

    def Pickler(self, file, protocol=None):
        p = KosherPickler(file, protocol=protocol, inline_module_pred=self._inline_module_pred)
        if hasattr(self, 'persistent_load'):
            p.persistent_load = self.persistent_load
        return p

    def dump(self, obj, file, protocol=None):
        self.Pickler(file, protocol=protocol).dump(obj)

    def dumps(self, obj, protocol=None):
        file = StringIO()
        try:
            self.dump(obj, file, protocol)
            return file.getvalue()
        finally:
            file.close()

import imp
def _find_module(mod_name):
    """
    Iterate over each part instead of calling imp.find_module directly.
    This function is able to find submodules (e.g. sickit.tree)
    """
    path = None
    for part in mod_name.split('.'):
        if path is not None:
            path = [path]
        file, path, description = imp.find_module(part, path)
        if file is not None:
            file.close()
    return path, description

_BUILTIN_TYPE_NAMES = {}
for k, v in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k

def _builtin_type(name):
    return getattr(types, name)

class _InternalPickler(cloudpickle.CloudPickler):
    dispatch = cloudpickle.CloudPickler.dispatch.copy()

    def __init__(self, file, protocol=None, inline_module_pred=None):
        super(_InternalPickler, self).__init__(file, protocol=protocol)
        self._inline_module_pred = inline_module_pred

    def _is_dynamic_module(self, module):
        if self._inline_module_pred is not None:
            if self._inline_module_pred(module.__name__):
                return True

        # If module is successfully found then it is not a dynamically created module
        if hasattr(module, '__file__'):
            return False

        try:
            _find_module(module.__name__)
            return False
        except ImportError:
            return True

    def save_module(self, obj):
        self.modules.add(obj)
        if self._is_dynamic_module(obj):
            self.save_reduce(cloudpickle.dynamic_subimport, (obj.__name__, vars(obj)), obj=obj)
        else:
            self.save_reduce(cloudpickle.subimport, (obj.__name__,), obj=obj)
    dispatch[types.ModuleType] = save_module

    def _is_dynamic_class(self, obj):
        return obj.__module__ == "__main__" or \
                (self._inline_module_pred is not None and \
                self._inline_module_pred(obj.__module__))

    def save_global(self, obj, name=None, pack=struct.pack):
        """
        Save a "global".

        The name of this method is somewhat misleading: all types get
        dispatched here.
        """
        if isinstance(obj, type) and self._is_dynamic_class(obj):
            return self.save_dynamic_class(obj)

        try:
            return pickle._Pickler.save_global(self, obj, name=name)
        except Exception:
            if obj.__module__ == "__builtin__" or obj.__module__ == "builtins":
                if obj in _BUILTIN_TYPE_NAMES:
                    return self.save_reduce(
                        _builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)

            typ = type(obj)
            if typ is not obj and isinstance(obj, (type, types.ClassType)):
                return self.save_dynamic_class(obj)

            raise

    dispatch[type] = save_global
    dispatch[types.ClassType] = save_global


import inspect
import io

class KosherPickler(object):
    def __init__(self, file, protocol=None, inline_module_pred=None):
        self._file = file
        self._protocol = protocol
        self._inline_module_pred = inline_module_pred

    def _dump_obj(self, mf, obj):
        pos = self._file.tell()

        p = _InternalPickler(self._file, protocol=self._protocol, inline_module_pred=self._inline_module_pred)
        if hasattr(self, 'persistent_id'):
            p.persistent_id = self.persistent_id

        # Speculatively dump module fields
        p.dump(mf)

        # Dump obj.
        p.dump(obj)

        return self._file.tell() - pos

    def _speculative_dump_obj(self, obj):
        size = self._dump_obj({}, obj)

        f = self._file

        # Now seek to beginning and see if we wrote references to any modules.
        f.seek(-size, io.SEEK_CUR)
        u = KosherUnpickler(f)
        module_fields = {}
        _find_class = u.find_class
        def find_class(module, name):
            if self._inline_module_pred is not None:
                if self._inline_module_pred(module):
                    module_fields[module] = __import__(module, fromlist=[name])
            return _find_class(module, name)

        u.find_class = find_class
        if hasattr(self, 'persistent_load'):
            u.persistent_load = self.persistent_load
        u.load()

        if len(module_fields) == 0 or self._inline_module_pred is None:
            return None

        # Seek to beginning
        f.seek(-size, io.SEEK_CUR)
        return {
            k: vars(v)
            for k, v in module_fields.items()
            if self._inline_module_pred(k)
        }

    def dump(self, obj):
        # Speculatively assume there will be no modules to write.
        missing_module_fields = self._speculative_dump_obj(obj)
        if missing_module_fields is None:
            return

        self._dump_obj(missing_module_fields, obj)
