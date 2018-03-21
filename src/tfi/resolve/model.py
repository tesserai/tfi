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

def _reify(resolution):
    init_rest = resolution.get('init_rest', [])
    loaded_fn = resolution.get('loaded_fn', None)

    init = {}
    if init_rest:
        init = eval("dict(%s" % init_rest[0])

    loaded = loaded_fn()
    is_loaded_a_class = False
    
    if callable(loaded):
        sig = inspect.signature(loaded)
        needed_params = OrderedDict(sig.parameters.items())
        # Only allow unspecified leading_value to be given.
        for k in init.keys():
            del needed_params[k]
        members = []
        if isinstance(loaded, type):
            members = inspect.getmembers(loaded, inspect.isfunction)
            is_loaded_a_class = True
        model_fn = partial(loaded, **init)
    else:
        members = inspect.getmembers(loaded, inspect.ismethod)
        needed_params = OrderedDict()
        model_fn = lambda: loaded

    members = [(k, v) for (k, v) in members if not k.startswith('_')]

    if resolution['can_refresh'] and is_loaded_a_class:
        models = weakref.WeakSet()
        orig_model_fn = model_fn

        def _wrapped_model_fn(*a, **kw):
            model = orig_model_fn(*a, **kw)
            # TODO(adamb) Should actually use weak references to avoid a memory leak!

            models.add(model)
            return model
        model_fn = _wrapped_model_fn

        def refresh_model():
            reloaded = loaded_fn()
            for model in models:
                previous_class = model.__class__
                try:
                    model.__class__ = reloaded
                    print("Replacing __class__", model, id(reloaded), reloaded)
                    if hasattr(model, '__tfi_refresh__'):
                        model.__tfi_refresh__()
                except:
                    model.__class__ = previous_class
                    raise

        resolution['refresh_fn'] = refresh_model

    resolution['model_fn_needed_params'] = needed_params
    resolution['model_members'] = members
    resolution['model_fn'] = model_fn
    
    return resolution

def _detect_model_file_kind(file):
    if os.path.isdir(file):
        # It's a SavedModel!
        return "tensorflow"

    # Assume it's a PyTorch model!
    return "pytorch"

def _model_module_for_kind(kind):
    if kind == "pytorch":
        import tfi.pytorch
        return tfi.pytorch
    if kind == "tensorflow":
        import tfi.tf
        return tfi.tf
    raise Exception("Can't detect model module %s" % model)

def _model_class_from_path_fn(source):
    kind = _detect_model_file_kind(source)
    mod = _model_module_for_kind(kind)
    return mod.as_class(source)

def resolve_exported(leading_value):
    return _reify({
        'source': os.path.abspath(leading_value),
        'loaded_fn': lambda: _model_class_from_path_fn(leading_value),
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
                'loaded_fn': lambda: _model_class_from_path_fn(f.name),
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

    def module():
        with open(source) as f:
            constants = _parse_python_str_constants(f.read())
            if '__tfi_conda_environment_yaml__' in constants:
                print("Found __tfi_conda_environment_yaml__.")

        return imp.load_source(module_name, source)

    with open(source, 'rb') as f:
        m = hashlib.sha1()
        m.update(f.read())
        sha1hex = m.hexdigest()

    return _reify({
        'source': source,
        'source_sha1hex': sha1hex,
        'classname': classname,
        'module_fn': module,
        'loaded_fn': lambda: getattr(module(), classname),
        'via_python': True,
        'init_rest': init_rest,
        'can_refresh': True,
    })

def resolve_module(leading_value):
    # Expecting leading_value to be something like "module.class(kwargs)"
    pre_initargs, *init_rest = leading_value.split("(", 1)
    *module_fragments, classname = pre_initargs.split(".")
    module_name = ".".join(module_fragments)
    def module():
        return importlib.import_module(module_name)
    return _reify({
        'source': source,
        'classname': classname,
        'loaded_fn': lambda: getattr(module(), classname),
        'module_fn': module,
        'via_python': True,
        'init_rest': init_rest,
        'can_refresh': True,
    })
