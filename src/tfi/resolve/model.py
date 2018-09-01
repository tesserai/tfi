import ast
import hashlib
import inspect
import os.path
import weakref

from collections import OrderedDict
from functools import partial


def _parse_python_str_constants(source):
    constants = {}
    parsed = ast.parse(source)
    nv = ast.NodeVisitor()
    nv.visit_Module = lambda n: nv.generic_visit(n)
    def visit_Assign(n):
        if len(n.targets) != 1:
            return

        target = n.targets[0]
        if not isinstance(target, ast.Name):
            return

        if not isinstance(n.value, ast.Str):
            return

        constants[target.id] = n.value.s

    nv.visit_Assign = visit_Assign
    nv.visit(parsed)
    return constants

class _module_proxy(object):
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)

def _reify(resolution):
    module_fn = resolution.get('module_fn', None)
    loaded_fn = resolution.get('loaded_fn', None)
    reset_module_fn = resolution.get('reset_module_fn', None)

    module_proxy = _module_proxy(module_fn())

    models = weakref.WeakSet()

    can_refresh = resolution['can_refresh']
    def watch_module_classes(module):
        if not can_refresh:
            return

        for _, member in inspect.getmembers(module, inspect.isclass):
            if hasattr(member, '__tfi_refresh_watchers__'):
                member.__tfi_refresh_watchers__.append(models.add)

    def refresh_model(progress_fn=None):
        if not progress_fn:
            progress_fn = lambda model, ix, total: None

        reset_module_fn()
        module = module_fn()
        watch_module_classes(module)
        module_proxy._target = module_fn()

        total = len(models)
        for i, model in enumerate(models):
            reloaded = getattr(module, model.__class__.__name__)
            progress_fn(model, i, total)
            previous_class = model.__class__
            try:
                model.__class__ = reloaded
                if hasattr(model, '__tfi_refresh__'):
                    model.__tfi_refresh__()
            except:
                model.__class__ = previous_class
                raise
        progress_fn(None, total, total)

    if can_refresh:
        resolution['refresh_fn'] = refresh_model
        resolution['module_fn'] = lambda: module_proxy

    init_rest = resolution.get('init_rest', [])
    init = {}
    if init_rest:
        init = eval("dict(%s" % init_rest[0])

    watch_module_classes(module_fn())
    loaded = loaded_fn()

    if callable(loaded):
        sig = inspect.signature(loaded)
        needed_params = OrderedDict(sig.parameters.items())
        # Only allow unspecified leading_value to be given.
        for k in init.keys():
            del needed_params[k]
        members = []
        if isinstance(loaded, type):
            members = inspect.getmembers(loaded, inspect.isfunction)
        model_fn = partial(loaded, **init)
    else:
        members = inspect.getmembers(loaded, inspect.ismethod)
        needed_params = OrderedDict()
        model_fn = lambda: loaded

    members = [(k, v) for (k, v) in members if not k.startswith('_')]

    resolution['model_fn_needed_params'] = needed_params
    resolution['model_members'] = members
    resolution['model_fn'] = model_fn

    return resolution

import mimetypes
import zipfile

_tensorflow_marker_file = 'saved_model.pb'
def _detect_model_file_kind(file):
    if os.path.isdir(file) and os.path.exists(os.path.join(file, _tensorflow_marker_file)):
        # It's a SavedModel!
        return "tensorflow"

    if zipfile.is_zipfile(file):
        with zipfile.ZipFile(file) as zipf:
            try:
                zipf.getinfo(_tensorflow_marker_file)
                return "tensorflow"
            except KeyError:
                print("didn't find info")
                pass
    
    if os.path.isfile(file) and file.endswith('.msp'):
        return 'msp'

    # Assume it's a PyTorch model!
    return "pytorch"

def _model_module_for_kind(kind):
    if kind == "pytorch":
        import tfi.driver.pytorch
        return tfi.driver.pytorch
    if kind == "tensorflow":
        import tfi.driver.tf
        return tfi.driver.tf
    if kind == "msp":
        import tfi.driver.msp
        return tfi.driver.msp
    raise Exception("Can't detect model module %s" % model)

def _load_model_from_path_fn(source):
    kind = _detect_model_file_kind(source)
    mod = _model_module_for_kind(kind)
    return mod.load(source)

def resolve_exported(leading_value):
    return _reify({
        'source': os.path.abspath(leading_value),
        'loaded_fn': lambda: _load_model_from_path_fn(leading_value),
        'module_fn': lambda: None,
        'can_refresh': False,
    })

def resolve_auto(leading_value):
    if leading_value is None:
        return {
            'source': "",
            'loaded': None,
        }
    if leading_value.startswith('@'):
        return resolve_exported(leading_value[1:])
    if leading_value.startswith('/') or leading_value.startswith('./'):
        return resolve_exported(leading_value)
    if leading_value.startswith('http://') or leading_value.startswith('https://'):
        return resolve_url(leading_value)
    if '.py:' in leading_value:
        return resolve_python_source(leading_value)

    return resolve_module(leading_value)

def resolve_url(leading_value):
    # Load exported model via http(s)
    pre_initargs, *init_rest = leading_value.split("(", 1)

    if '.py:' in pre_initargs or pre_initargs.endswith('.py'):
        import imp
        import urllib.request
        source, classname = pre_initargs.split('.py:', 1)
        domain, *uri_path = source.split('://', 1)[1].split('/')
        module_name = ".".join([*reversed(domain.split('.')), *uri_path])
        source = source + ".py"

        with urllib.request.urlopen(source) as f:
            module = imp.load_source(module_name, source, f)
            return _reify({
                'source': source,
                'classname': classname,
                'module_fn': lambda: module,
                'loaded_fn': lambda: getattr(module, classname),
                'via_python': True,
                'init_rest': init_rest,
                'can_refresh': False,
            })

    source = pre_initargs

    # Assume URL is to documentation.
    from bs4 import BeautifulSoup
    import tempfile
    import urllib.request
    import requests
    from tqdm import tqdm

    found = False
    with urllib.request.urlopen(source) as f:
        doc = BeautifulSoup(f, "html.parser")
        matching = doc.html.head.findAll('meta', {"name":'tesserai:snapshot'})
        for match in matching:
            if match.has_attr('content'):
                source = urllib.parse.urljoin(source, match['content'])
                found = True
                break

    if not found:
        raise Exception("Couldn't find tesserai:snapshot in meta in %s" % doc)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        print("will download from", source, "to", f.name)

        with requests.get(source, stream=True) as r:
            progress = tqdm(
                    total=int(r.headers["Content-Length"]),
                    desc="Downloading",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=True)
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

            source = f.name
            return _reify({
                'source': source,
                'classname': classname,
                'loaded_fn': lambda: _load_model_from_path_fn(f.name),
                'module_fn': lambda: module,
                'via_python': via_python,
                'can_refresh': False,
            })

def resolve_python_source(leading_value):
    import imp
    pre_initargs, *init_rest = leading_value.split("(", 1)
    source, classname = pre_initargs.split('.py:', 1)
    module_name = source.replace('/', '.')
    source = os.path.abspath(source + ".py")

    def load_module():
        with open(source) as f:
            constants = _parse_python_str_constants(f.read())
            if '__tfi_conda_environment_yaml__' in constants:
                print("Found __tfi_conda_environment_yaml__.")

        return imp.load_source(module_name, source)

    with open(source, 'rb') as f:
        m = hashlib.sha1()
        m.update(f.read())
        sha1hex = m.hexdigest()

    mod = None
    def module():
        nonlocal mod
        if mod is None:
            mod = load_module()
        return mod

    def reset_module():
        nonlocal mod
        mod = None

    return _reify({
        'source': source,
        'source_sha1hex': sha1hex,
        'classname': classname,
        'module_fn': module,
        'loaded_fn': lambda: getattr(module(), classname),
        'reset_module_fn': reset_module,
        'via_python': True,
        'init_rest': init_rest,
        'can_refresh': True,
    })

def resolve_module(leading_value):
    # Expecting leading_value to be something like "module.class(kwargs)"
    pre_initargs, *init_rest = leading_value.split("(", 1)
    *module_fragments, classname = pre_initargs.split(".")
    module_name = ".".join(module_fragments)
    def load_module():
        return importlib.import_module(module_name)

    mod = None
    def module():
        nonlocal mod
        if mod is None:
            mod = load_module()
        return mod

    def reset_module():
        nonlocal mod
        mod = None

    return _reify({
        'source': None,
        'classname': classname,
        'loaded_fn': lambda: getattr(module(), classname),
        'module_fn': module,
        'reset_module_fn': reset_module,
        'via_python': True,
        'init_rest': init_rest,
        'can_refresh': True,
    })
