import json
import os.path
import types

from tfi.driver.msp.root import make_default_env_and_scope, eval_expr_with_env

class Base(object):
  pass

def as_class(saved_model_path):
  # Trap all calls to open, so we can watch those files an reload!
  with open(saved_model_path) as f:
    model = json.load(f)
    env, scope = make_default_env_and_scope()
    if "variables" in model:
      scope.load(model["variables"])
    model_val = eval_expr_with_env(model["expression"], env)

    classdict = {
      slot: member
      for slot, member in model_val.items()
      if not slot.startswith("_")
    }

    classname, _ = os.path.splitext(os.path.basename(saved_model_path))
    return type(classname, (Base,), classdict)

def load(saved_model_path):
  return as_class(saved_model_path)()

def dump(export_path, model):
  pass

export = dump
