import os
if 'CXX' in os.environ:
  os.environ['CC'] = os.environ['CXX']

import json
from datetime import timedelta
import numpy as np
import types
import inspect

import sys
def eprint(*a):
  print(*a, file=sys.stderr)

class Proto:
  # def __new__(self, proto, *args, **kw):
  #   return super(Proto, self).__new__(self, *args, **kw)

  def __init__(self, proto, *args, **kw):
    self.__proto__ = proto
    super(Proto, self).__init__(*args, **kw)

  def __getstate__(self):
    return dict(self.__dict__)

  def __setstate__(self, d):
    self.__dict__.update(d)

  def __getattr__(self, name):
    try:
      attr = getattr(self.__proto__, name)
      # key trick: rebind methods from the prototype to the current object
      if (inspect.ismethod(attr) and attr.__self__ is self.__proto__):
        attr = types.MethodType(attr.__func__, self)
      return attr
    except AttributeError:
      s = super(Proto, self)
      if hasattr(s, '__getattr__'):
        return s.__getattr__(name)
      raise AttributeError("No attribute %s in %r" % (name, list(self.__dict__.keys())))

class ScopeEnv(Proto):
  """
  An implementation of Env that pulls from a scope.
  """
  def __init__(self, proto, scope):
    super(ScopeEnv, self).__init__(proto)
    self.__scope__ = scope

  def __getattr__(self, name):
    scope = self.__scope__
    if scope.hasvar(name):
      return scope.value(name)

    s = super(ScopeEnv, self)
    if hasattr(s, '__getattr__'):
      return s.__getattr__(name)

def dump_proto(p):
  return dir(p)

def nan_safe_round(n): return n if np.isnan(n) else round(n)


this_module = sys.modules[__name__]
default_env = Proto(this_module)

import dask.dataframe as dd

import pandas as pd
def _pivot_table(df=None, index=None, values=None, aggfunc=None):
  r = pd.pivot_table(df, index=index, values=values, aggfunc=aggfunc)
  # eprint("pd.pivot_table(...,", "index=", index, "values=", values, "aggfunc=", aggfunc, ") => ", r)
  return r
default_env.pivot_table = _pivot_table

import itertools
def defmethod(name, lmbda):
  # eprint("defining", name, "as", lmbda)
  setattr(default_env, name, lmbda)

def debug(m, o):
  eprint("debug", m, type(o), o)
  return o

from random import random

def without_keys(d, ks):
  d = d.copy()
  for k in ks:
    del d[k]
  return d

dd_from_pandas = dd.from_pandas

import dask.array as darray

defmethod('isnan', np.isnan)
defmethod('isnat', np.isnat)
defmethod('str', str)
defmethod("getattr", getattr)
defmethod('/', lambda a, b: a / b)
defmethod('*', lambda a, b: a * b)
defmethod('&', lambda a, b: a & b)
defmethod('**', lambda a, b: a ** b)
defmethod('-', lambda a, b: a - b)
defmethod('>', lambda a, b: a > b)
defmethod('>=', lambda a, b: a >= b)
defmethod('<', lambda a, b: a < b)
defmethod('<=', lambda a, b: a <= b)
defmethod('range', range)
defmethod('+', lambda a, b: a + b)
defmethod('[]', lambda o, k: o[k])
defmethod('get', lambda o, k, default: o.get(k, default))
defmethod('dot', darray.dot)
defmethod('abs', darray.fabs)
defmethod('log', darray.log)
defmethod('ceil', darray.ceil)
defmethod('sign', darray.sign)
defmethod('mean', lambda df: df.mean())
defmethod('type', type)
defmethod('pdconcat', pd.concat)
defmethod('to_timedelta', pd.to_timedelta)
defmethod('concat', lambda *l: [e for e in itertools.chain(*l)])
defmethod('tuple', lambda *l: l)
defmethod('list', lambda *l: list(l))
defmethod('object-merge', lambda *ds: dict(itertools.chain(*[d.items() for d in ds])))
defmethod('map', lambda fn, l: [fn(e) for e in l])
defmethod('json', lambda o: json.dumps(o, sort_keys=True))
defmethod('zip', lambda *l: zip(*l))
defmethod('dict', lambda l: dict(l))
defmethod('eq', lambda a, b: a == b)
defmethod('head', lambda df, n: df.head(n))
defmethod('clip', lambda df, *a: df.clip(*a))

Series = pd.Series
defmethod('reset_index', lambda df: df.reset_index())
defmethod('.drop_duplicates', lambda df, *rest: df.drop_duplicates(*rest))
defmethod('.apply', lambda df, *rest: df.apply(*rest))
defmethod('.date', lambda df, *rest: df.date(*rest))
defmethod('.join', lambda df, *rest: df.join(*rest))
defmethod('.min', lambda df, *rest: df.min(*rest))
defmethod('.date', lambda df, *rest: df.date(*rest))
defmethod('.groupby', lambda df, *rest: df.groupby(*rest))
defmethod('.max', lambda df, *rest: df.max(*rest))
defmethod('.compute', lambda df, *rest: df.compute(*rest))
defmethod('.get_group', lambda df, *rest: df.get_group(*rest))

def _assign_index(o, k, v):
  o[k] = v
  return v
defmethod('[]=', _assign_index)

import bisect
def insort(l, b):
  bisect.insort(l, b)
  return l

class defaultdict2():
  def __init__(self, fn):
    self._fn = fn
    self._dict = {}

  def __getitem__(self, key):
    if key in self._dict:
      return self._dict[key]
    v = self._fn(key)
    self._dict[key] = v
    return v

def _locitem(df, a):
  try:
    return df.loc[a]
  except KeyError as ex:
    eprint("_locitem .. KeyError", ex, a)
    return pd.DataFrame(columns=df.columns)
defmethod('loc[]', _locitem)

def envdir(env):
  return [k if k != '__proto__' else envdir(v) for k, v in env.__dict__.items() if k == '__proto__' or not k.startswith('__')]

def eval_begin_expr(subexprs, env, eval_expr_fn):
  result = None
  for subexpr in subexprs:
    result = eval_expr_fn(subexpr, env)
  return result

def eval_let_expr(expr, env, eval_expr_fn):
  param_exprs = []
  bindings_expr = expr[1]
  args = []
  if isinstance(bindings_expr, list):
    for k, v in bindings_expr:
      param_exprs.append(k)
      args.append(v)
  else:
    for k, v in bindings_expr.items():
      param_exprs.append(k)
      args.append(v)
  let_lambda = [["lambda", param_exprs, *expr[2:]], *args]
  return eval_expr_fn(let_lambda, env)

def eval_dict_expr(expr, env, eval_expr_fn):
  return dict([(k, eval_expr_fn(v, env)) for k, v in expr.items()])

def eval_cond_expr(pred_exprs, env, eval_expr_fn):
  for pred_expr, *rest_exprs in pred_exprs:
    if eval_expr_fn(pred_expr, env):
      return eval_begin_expr(rest_exprs, env, eval_expr_fn)
  return None

# TODO(adamb) We aren't looking to actually evaluate anything. Instead we're just looking to find variables
#     *without* a binding.
class FreeVar:
  def __init__(self, name):
    self.name = name
    self.constraints = []

  def __repr__(self):
    return "<FreeVar name=%r constraints=%r>" % (self.name, self.constraints)

  def accept_constraint(self, constraint):
    # eprint("FreeVar#accept_constraint", self, self.name, constraint)
    self.constraints.append(constraint)

  def merged_constraints(self):
    if len(self.constraints) == 0:
      return None
    if len(self.constraints) == 1:
      return self.constraints[0]
    raise Exception("can't merge multiple constraints yet.")

undefined = object()

class Scope(object):
  def __init__(self, parent=None, parent_env=None, prefix=None, constraints=None, values=None, children=None):
    if constraints is None:
      constraints = {}
    if prefix is None:
      prefix = ()
    if values is None:
      values = {}
    if children is None:
      children = {}

    self._env = ScopeEnv(parent_env, self)
    self._parent = parent
    self._values = values
    self._prefix = prefix
    self._constraints = constraints
    self._children = children
    self._children[prefix] = self

  def env(self):
    return self._env

  def parent(self):
    return self._parent

  def root(self):
    s = self
    while True:
      if s._parent is None:
        return s
      s = s._parent

  def name(self):
    return self._prefix[-1] if len(self._prefix) > 0 else None

  def _key(self, name, demand_new=False):
    key = (*self._prefix, name)
    if demand_new:
      self._demand_new(name, key)
    return key

  def _demand_new(self, name, key):
    try:
      if key in self._children:
        raise Exception("Already have child scope %s" % name)
      if key in self._constraints:
        raise Exception("Already have var %s" % name)
    except TypeError:
      raise Exception("Failed to _demand_new for var %s, key %s" % (name, key))

  def scope(self, name, env):
    # TODO(adamb) Should really be checking if we already have a value.
    key = self._key(name, False)

    if key in self._children:
      return self._children[key]

    # TODO(adamb) Should blow up if there's already a variable with this name!!!!

    return Scope(parent=self,
        parent_env=env,
        prefix=key,
        constraints=self._constraints,
        values=self._values,
        children=self._children)

  def hasvar(self, name):
      return self._key(name, False) in self._values

  def value(self, name):
      key = self._key(name, False)
      return self._values[key]

  def var(self, name, constraints):
    key = self._key(name, False)

    if key in self._constraints and self._constraints[key] != constraints:
      raise Exception("Encountered constaints differ. Had %r, encountered %r" % (
          self._constraints[key], constraints))

    self._constraints[key] = constraints

    # TODO(adamb) Should really be checking if value fits constraints.
    if key in self._values:
      return self._values[key]

    if "default" in constraints:
      value = constraints["default"]
    elif "min" in constraints:
      value = constraints["min"]
    elif "max" in constraints:
      value = constraints["max"]
    elif "type" in constraints:
      value = constraints["type"]()

    self._values[key] = value
    return value

  def swap_values(self, values):
    previous = []
    for k, value in values:
      key = tuple(k)
      if key in self._values:
        previous.append((k, self._values[key]))
      else:
        previous.append((k, undefined))

      if value is undefined:
        del self._values[key]
      else:
        self._values[key] = value

    return previous

  # XXX(adamb) Might be better to copy and return cloned tree?
  def load(self, values):
    for k, value in values:
      key = tuple(k)
      if value is undefined:
        self._values[key]
      else:
        self._values[key] = value

  def save(self):
    return list(self._values.items())

  def constraint_entry_for(self, name):
    key = self._key(name, False)
    return key, self._constraints[key]

  # XXX(adamb) Would be better to return a view that wasn't modifiable...
  def constraints(self):
    return list(self._constraints.items())

def gen_eval_expr(apply_pos_fn, apply_kwd_fn, make_lambda_fn, load_scope_fn, eval_var_expr_fn):
  def this_eval_expr(expr, env):
    try:
      if isinstance(expr, list):
        fn_expr = expr[0]
        if fn_expr == "quote":
          return expr[1]
        if fn_expr == "lquote":
          return expr[1:]
        if fn_expr == "eval":
          return this_eval_expr(this_eval_expr(expr[1], env), env)
        if fn_expr == "cond":
          return eval_cond_expr(expr[1:], env, this_eval_expr)
        if fn_expr == "define":
          value = this_eval_expr(expr[2], env)
          setattr(env, expr[1], value)
          return None
        if fn_expr == "let":
          return eval_let_expr(expr, env, this_eval_expr)
        if fn_expr == "lambda":
          return make_lambda_fn(expr[1], expr[2:], env, this_eval_expr)
        if fn_expr == "begin":
          return eval_begin_expr(expr[1:], env, this_eval_expr)
        if fn_expr == "apply":
          fn = this_eval_expr(expr[1], env)
          args = this_eval_expr(expr[2], env)
          if isinstance(args, dict):
            return apply_kwd_fn(fn, args)
          return apply_pos_fn(fn, args)
        if fn_expr == "scope":
          return eval_var_expr_fn(expr[1], expr[2], expr[3:], env, make_lambda_fn, this_eval_expr)
        if fn_expr == 'scope-commit':
          return load_scope_fn(expr[1], env, this_eval_expr)
        if fn_expr == "scope-root":
          scope = Scope(parent_env=env)
          scope.load(this_eval_expr(expr[1], env))
          return eval_begin_expr(expr[2:], scope.env(), this_eval_expr)
        fn = this_eval_expr(fn_expr, env)
        arg_exprs = expr[1:]
        args = [this_eval_expr(arg_expr, env) for arg_expr in arg_exprs]
        return apply_pos_fn(fn, args)
      if isinstance(expr, dict):
        return eval_dict_expr(expr, env, this_eval_expr)
      if isinstance(expr, str):
        if env is not None:
          try:
            return getattr(env, expr)
          except AttributeError as ex:
            if hasattr(env, '__resolve_free_variable__'):
              return env.__resolve_free_variable__(expr)
            raise ex
        raise Exception("Nothing bound to %s in %s" % (expr, envdir(env)))
      return expr
    except Exception as e:
      eprint("error evaluating", expr)
      raise e
  return this_eval_expr

def fn_constraints(fn):
  if hasattr(fn, '__constraints__'):
    return fn.__constraints__

  try:
    argspec = inspect.getfullargspec(fn)
  except TypeError:
    return None

  if hasattr(fn, '__annotations__'):
    annotations = fn.__annotations__
  else:
    annotations = dict(argspec[-1])

  pos_defaults = dict(zip(argspec.args[::-1], (argspec.defaults or ())[::-1]))
  kwd_defaults = argspec.kwonlydefaults or {}
  constraints = {}
  for name, constraint in annotations.items():
    if constraint is None:
      continue
    constraint = dict(constraint)
    if name in pos_defaults:
      constraint["default"] = pos_defaults[name]
    if name in kwd_defaults:
      constraint["default"] = kwd_defaults[name]
    constraints[name] = constraint
  fn.__constraints__ = constraints

  return constraints

def fv_expose_fn_constraints(fn, arg_entry_iter):
  fa = fn_constraints(fn)

  if fa is None or len(fa) == 0:
    return

  for k, v in arg_entry_iter:
    if not isinstance(v, FreeVar):
      continue
    if k not in fa:
      continue
    v.accept_constraint(fa[k])

  # If fn is a type and has __call__, then return
  # __call__ function, so the next "application" works.
  if isinstance(fn, type):
    if hasattr(fn, '__call__'):
      return fn.__call__

def fv_expose_fn_constraints_kwd(fn, args):
  return fv_expose_fn_constraints(fn, args.items())

def fv_expose_fn_constraints_pos(fn, args):
  # HACK(adamb) We must be in an apply and couldn't discover anything about the args.
  if args is None:
    # eprint("fv_expose_fn_constraints_pos no args", fn, args)
    return None

  try:
    argspec = inspect.getfullargspec(fn)
  except TypeError as te:
    # eprint("fv_expose_fn_constraints_pos te", fn, args, te)
    return None
  argnames = argspec[0]
  # If fn is a type, ignore the "self" arg.
  if isinstance(fn, type):
    argnames = argnames[1:]
  return fv_expose_fn_constraints(fn, zip(argnames, args))

import hashlib

_free_vars_expr = None
def make_lambda(params_expr, body_exprs, env, eval_expr_fn):
  def apply_lambda(*args, **kwargs):
    new_env = Proto(env)
    if len(kwargs) == 0:
      def bind_all(p, a):
        for k, v in zip(p, a):
          if isinstance(k, str):
            setattr(new_env, k, v)
          elif isinstance(k, list):
            bind_all(k, v)
          else:
            raise Exception("Can't bind %r to %r" % (k, v))
      bind_all(params_expr, args)
    else:
      for k in params_expr:
        v = kwargs[k]
        setattr(new_env, k, v)
    return eval_begin_expr(body_exprs, new_env, eval_expr_fn)

  # Calculate param constraints and free variables
  # eprint("calculating constraints for", apply_lambda, "...", params_expr, body_exprs, envdir(env), eval_expr_fn)
  def trapping_env(baseenv):
    fv_new_env = Proto(baseenv)
    fv_params = []
    fv_nonparams = []
    for param in params_expr:
      if isinstance(param, str):
        fv = FreeVar(param)
        fv_params.append(fv)
        setattr(fv_new_env, param, fv)
      elif isinstance(param, list):
        # HACK(adamb) Only make string param constraints visible for now.
        for p in param:
          setattr(fv_new_env, p, FreeVar(p))
    def resolve_free_variable(expr):
      # eprint("resolve_free_variable", expr)
      if hasattr(baseenv, '__resolve_free_variable__'):
        fv = baseenv.__resolve_free_variable__(expr)
      else:
        fv = FreeVar(expr)
      fv_nonparams.append(fv)
      setattr(fv_new_env, expr, fv)
      return fv
    fv_new_env.__resolve_free_variable__ = resolve_free_variable
    return fv_new_env, fv_params, fv_nonparams

  tenv, fv_params, fv_nonparams = trapping_env(env)
  eval_begin_expr(body_exprs, tenv, _free_vars_expr)
  fa = inspect.getfullargspec(apply_lambda)[-1]
  for fv in fv_params:
    fa[fv.name] = fv.merged_constraints()
  apply_lambda.__annotations__ = fa

  # Calculate values needed from env, but not present.
  # apply_lambda.__free_variable_constraints__ = dict([(fv.name, fv.merged_constraints()) for fv in fv_nonparams])

  # Now calculate values captured from env.
  tenv, fv_params, fv_nonparams = trapping_env(default_env)
  eval_begin_expr(body_exprs, tenv, _free_vars_expr)
  # eprint("calculated values from environment", apply_lambda.__free_variable_constraints__, fv_params, fv_nonparams)

  def captured_scope_vars():
    captured = [(fv, getattr(env, fv.name)) for fv in fv_nonparams if hasattr(env, fv.name)]
    # eprint("captured_scope_vars initial", captured)
    captured = dict(captured)

    # Walk upwards from the scope closest to the lambda's env. Track only the
    # values captured by the function.
    cscope = env.__scope__ if hasattr(env, '__scope__') else None
    # eprint("initial cscope", cscope)
    while cscope is not None:
      resolved = [fv for fv, val in captured.items() if cscope.hasvar(fv.name)]
      # eprint("cscope", cscope, resolved)
      if len(resolved) > 0:
        for fv in resolved:
          # eprint("removing", type(fv), repr(fv), fv)
          del captured[fv]
        yield True, cscope, resolved
      cscope = cscope.parent()
    for fv, val in captured.items():
      # eprint("fv, val", fv, val)
      if hasattr(val, '__captured_scope_vars__'):
        for loc, sc, arr in val.__captured_scope_vars__():
          yield False, sc, arr
      yield True, None, [fv]

  def __scope_vars__():
    svs = []
    for loc, sc, fvs in captured_scope_vars():
      if not sc:
        continue
      for fv in fvs:
        sv = sc.constraint_entry_for(fv.name)
        if sv not in svs:
          svs.append(sv)
    return svs
  apply_lambda.__captured_scope_vars__ = captured_scope_vars
  apply_lambda.__scope_vars__ = __scope_vars__
  apply_lambda.scope = lambda: env.__scope__
  apply_lambda.env = env

  return apply_lambda

def eval_var_expr(scope_name_expr, var_names_expr, body_exprs, env, make_lambda_fn, eval_expr_fn):
  # TODO(adamb) Create new scope. Push onto scope "stack".
  scope_name = eval_expr_fn(scope_name_expr, env)

  if not hasattr(env, '__scope__'):
      raise Exception("No __scope__ entry in env", env, env.__proto__, dir(env), dir(env.__proto__))

  scope = env.__scope__
  if scope_name is not None:
    scope = scope.scope(scope_name, env)
    newenv = scope.env()
  else:
    newenv = ScopeEnv(env, scope)

  # HACK(adamb) Which env should we be pasing into make_lambda_fn?
  var_lambda = make_lambda_fn(var_names_expr, body_exprs, env, eval_expr_fn)
  var_constraints = fn_constraints(var_lambda)

  # eprint("var_constraints",  var_constraints)
  # TODO(adamb) If our scope has a value for us, we should use it. Otherwise we
  #     should use the default value based on the constraints we have.
  # TODO(adamb) Provide a way to overwrite and save these values wholesale.
  # TODO(adamb) Use scope to define variables. Values of variables
  #     should be either their defaults or nan (if we can discover their constraints) or None.
  for var_name in var_names_expr:
      var_value = var_constraints[var_name]
    #   eprint("var_name", var_name, var_value)
      scope.var(var_name, var_value)
  return eval_begin_expr(body_exprs, newenv, eval_expr_fn)

_free_vars_expr = gen_eval_expr(
  apply_pos_fn=fv_expose_fn_constraints_pos,
  apply_kwd_fn=fv_expose_fn_constraints_kwd,
  eval_var_expr_fn=lambda scope_name_expr, var_names_expr, body_exprs, env, make_lambda_fn, eval_expr_fn: eval_begin_expr(body_exprs, env, eval_expr_fn),
  make_lambda_fn=make_lambda,
  load_scope_fn=lambda arg, env, eval_expr_fn: None)

def make_default_env_and_scope():
  scope = Scope(parent_env=default_env)
  return scope.env(), scope

def fallback_apply_pos_fn(fn, args):
  if len(args) == 1:
    try:
      return fn[args[0]]
    except KeyError:
      raise Exception("Can't use %r as key for (%r) %r" % (args[0], type(fn), fn))
    except IndexError:
      raise Exception("Can't use %r as index for (%r) %r" % (args[0], type(fn), fn))
  raise Exception("Can't apply %r to %r" % (fn, args))

def load_scope(arg, env, this_eval_expr):
  to_commit = this_eval_expr(arg, env)
  # eprint("scope-commit", to_commit)
  env.__scope__.load(to_commit)
  return None

eval_expr_with_env = gen_eval_expr(
  # apply_pos_fn=lambda fn, args: fn(*args) if callable(fn) else fallback_apply_pos_fn(fn, args),
  apply_pos_fn=lambda fn, args: fn(*args),
  apply_kwd_fn=lambda fn, args: fn(**args),
  eval_var_expr_fn=eval_var_expr,
  make_lambda_fn=make_lambda,
  load_scope_fn=load_scope)
eval_expr = lambda expr: eval_expr_with_env(expr, default_env)

def parse(str):
  return eval_expr(json.loads(str))

import GPyOpt
from GPyOpt.experiment_design import initial_design
from numpy.random import seed
import time

def BayesianMinimizer(model, use_seed, batch_size, iterations):
    if use_seed is None:
      use_seed = int(time.time() * 1e6) & 0xffffffff
    seed(use_seed)

    continuous_bounds = []
    continuous_xforms = []
    continuous_defaults = []
    discrete_bounds = []
    discrete_xforms = []
    discrete_defaults = []

    for var_name, var_spec in model.__scope_vars__():
      # eprint("var_name, var_spec", var_name, var_spec)
      bound = {
        "name": "|".join(var_name),
        "type": "continuous" if var_spec['type'] == float else "discrete",
        "domain": (var_spec['min'], var_spec['max']),
      }

      default = None
      if "default" in var_spec:
        default = var_spec["default"]
      elif "min" in var_spec:
        default = var_spec["min"]
      elif "max" in var_spec:
        default = var_spec["max"]
      elif "type" in var_spec:
        default = var_spec["type"]()

      if bound["type"] == "continuous":
        continuous_bounds.append(bound)
        continuous_xforms.append(var_spec['type'])
        continuous_defaults.append(default)
      else:
        discrete_bounds.append(bound)
        discrete_xforms.append(var_spec['type'])
        discrete_defaults.append(default)

    # HACK(adamb) Discrete bounds need to be at the end for some reason...
    bounds = [*continuous_bounds, *discrete_bounds]
    xforms = [*continuous_xforms, *discrete_xforms]
    defaults = [*continuous_defaults, *discrete_defaults]

    space = GPyOpt.Design_space(space=bounds)
    initial_batch_size = batch_size if batch_size > 1 else 2

    evaluations = None
    X_init = None
    Y_init = None
    if evaluations is not None:
      X_init, Y_init = evaluations
    else:
      X_init = np.vstack([initial_design('random', space, initial_batch_size), defaults])

    def args_for(args):
      # We need to properly xform the resulting value to ensure it's representable.
      args = [xform(xx) for xx, xform in zip(args, xforms)]
      return [(tuple(bound["name"].split("|")), arg) for bound, arg in zip(bounds, args)]

    def evaluate(x):
      costs = np.empty(shape=[0, 1])
      f_evals = np.empty(shape=[0, 1])
      for i in range(x.shape[0]):
        st_time = time.time()
        scope = model.scope()
        swapped = scope.swap_values(args_for(x[i]))
        r = model()
        scope.swap_values(swapped)

        cost = time.time() - st_time
        eprint("args", x[i])
        eprint("r", r)
        # eprint("cost", cost)

        f_evals = np.vstack([f_evals, r])
        costs = np.vstack([costs, cost])

      return f_evals, costs

    gpo_objective = GPyOpt.core.task.objective.Objective()
    gpo_objective.evaluate = evaluate
    gpo_model = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)

    acquisition = GPyOpt.acquisitions.AcquisitionEI(
      gpo_model,
      space,
      optimizer=GPyOpt.optimization.AcquisitionOptimizer(space))

    if batch_size == 1:
      evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    else:
      acquisition = GPyOpt.acquisitions.AcquisitionLP(
          gpo_model,
          space,
          optimizer=GPyOpt.optimization.AcquisitionOptimizer(space),
          acquisition=acquisition)
      evaluator = GPyOpt.core.evaluators.LocalPenalization(
          acquisition,
          batch_size=batch_size)
    problem = GPyOpt.methods.ModularBayesianOptimization(
        model=gpo_model,
        space=space,
        objective=gpo_objective,
        acquisition=acquisition,
        evaluator=evaluator,
        normalize_Y=True,
        X_init=X_init,
        Y_init=Y_init)

    problem.run_optimization(iterations)
    latest_args = [list(a) for a in problem.X]
    latest_rsts = [a[0] for a in problem.Y]
    for arg, rst in zip(latest_args, latest_rsts):
      yield args_for(arg), rst


def read_csv(url, **kw):
  import requests as _requests
  r = _requests.get(url, stream=True)
  r.raw.decode_content = True
  return pd.read_csv(r.raw, **kw)

def del_cols(df, cols):
  for col in cols:
    del df[col]
  return df

def map_col(df, col, fn):
  df[col] = df[col].apply(fn)
  return df

def apply_row_df(df, fn):
  return df.apply(fn, axis=1, reduce=True)

def sumrange(series, start, end):
  return series[(series.index >= start) & (series.index <= end)].sum()

def _memoize(f):
    results = {}
    def helper(*n):
        if n not in results:
            results[n] = f(*n)
        return results[n]
    return helper

@_memoize
def parse_date(text):
  return pd.to_datetime(text)

@_memoize
def parse_relative_date(text, today):
  if text.startswith("+"):
    t = today + pd.to_timedelta(text[1:])
  elif text.endswith("d"):
    t = today + pd.to_timedelta(text)
  else:
    t = pd.to_datetime(text)

  return t

def synthesize_cols_dask(df, fns):
  df = df.copy()
  toadd = [
    (k, df.apply(fn, axis=1))
    for k, fn in fns.items()
  ]
  for k, series in toadd:
    df[k] = series
  return df

def synthesize_cols(df, fns):
  toadd = [
    (k, df.apply(fn, axis=1, reduce=True))
    for k, fn in fns.items()
  ]
  df = df.copy()
  for k, series in toadd:
    df[k] = series
  return df

from collections import OrderedDict
def combinestupidobjects(df, tgt, srccols):
  df = df.copy()
  r = df.apply(lambda row: json.dumps(OrderedDict([(k, row[k]) for k in srccols])), axis=1, reduce=True)
  df[tgt] = r
  for k in srccols:
    del df[k]
  return df

def set_cols(df, d):
  for k, v in d.items():
    df[k] = v
  return df

def set_col_apply_row(df, col, fn):
  r = df.apply(fn, axis=1, raw=True)
  df[col] = r
  return df

def dicts_to_df(keys, dicts):
  return pd.DataFrame([[d[k] for k in keys] for d in dicts], columns=keys)

def set_index(df, index):
  return df.set_index(index)

def filter_df(df, filter_obj):
  cols = list(df.columns.values)

  only = None
  for k, v in filter_obj.items():
      cols.remove(k)
      if only is None:
          only = df[k] == v
      else:
          only = only & (df[k] == v)
  if only is not None:
      df = df[only]

  return df.reset_index()[cols]

class MyopicExpert:
    def __init__(self, predictor):
        self.predictor = predictor

    def name(self):
        return "myopic_" + self.predictor.name()

    def __call__(self, history, arrival_ds, min_sale_ds, max_sale_ds):
        yhat = self.predictor(
            history=history,
            date_range=(arrival_ds, min_sale_ds))
        eprint("MyopicExpert", yhat)
        return yhat

class OptimisticExpert:
    def __init__(self, predictor):
        self.predictor = predictor

    def name(self):
        return "optimistic_" + self.predictor.name()

    def __call__(self, history, arrival_ds, min_sale_ds, max_sale_ds):
        yhat = self.predictor(
            history=history,
            date_range=(arrival_ds, max_sale_ds))
        eprint("OptimisticExpert", yhat)
        return yhat

class MeanExpert:
    def __init__(self, short_predictor, long_predictor):
        self.short_predictor = short_predictor
        self.long_predictor = long_predictor

    def name(self):
        return "mean_%s_%s" % (self.short_predictor.name(), self.long_predictor.name())

    def __call__(self, history, arrival_ds, min_sale_ds, max_sale_ds):
        return np.mean([
            nan_safe_round(
                self.short_predictor(
                    history=history,
                    date_range=(arrival_ds, min_sale_ds))),
            nan_safe_round(
                self.long_predictor(
                    history=history,
                    date_range=(arrival_ds, max_sale_ds)))
        ])

class ReasonableExpert:
    def __init__(self, short_predictor, long_predictor):
        self.short_predictor = short_predictor
        self.long_predictor = long_predictor

    def name(self):
        return "reasonable_%s_%s" % (self.short_predictor.name(), self.long_predictor.name())

    def __call__(self, history, arrival_ds, min_sale_ds, max_sale_ds):
        return np.max([
            nan_safe_round(
                self.short_predictor(
                    history=history,
                    date_range=(arrival_ds, min_sale_ds))),
            nan_safe_round(
                self.long_predictor(
                    history=history,
                    date_range=(arrival_ds, max_sale_ds)))
        ])

class CautiousExpert:
    def __init__(self, short_predictor, long_predictor):
        self.short_predictor = short_predictor
        self.long_predictor = long_predictor

    def name(self):
        return "cautious_%s_%s" % (self.short_predictor.name(), self.long_predictor.name())

    def __call__(self, history, arrival_ds, min_sale_ds, max_sale_ds):
        return np.min([
            nan_safe_round(
                self.short_predictor(
                    history=history,
                    date_range=(arrival_ds, min_sale_ds))),
            nan_safe_round(
                self.long_predictor(
                    history=history,
                    date_range=(arrival_ds, max_sale_ds)))
        ])

class NextMaxPredictor:
    def name(self):
        return "next_max"

    def __call__(self, history, date_range):
        raw = history.rolling(history.shape[0], min_periods=1).max().drop_duplicates().iloc[-3:]['y'].values
        if len(raw) == 0:
            return 0
        if len(raw) == 1 or raw[-2] == 0:
            return raw[-1]
        eprint("raw", raw)
        return raw[-1] * raw[-1] / raw[-2]

class Reduce:
    def __init__(self, reduce_func=None, name=None):
        self._name = name
        if isinstance(reduce_func, str):
            reduce_func_str = reduce_func
            if self._name is None:
                self._name = reduce_func_str
            reduce_func = lambda x: getattr(x, reduce_func_str)()
        self.reduce_func = reduce_func

    def name(self):
        return self._name

    def __call__(self, history, date_range):
        ds_start, ds_end = date_range
        ds_len = (ds_end - ds_start).days + 1
        r = ds_len * self.reduce_func(history['y'])
        eprint("Reduce", r, "from", self._name, "to", history['y'])
        return r

class HistoryRolling:
    def __init__(self, predictor):
        self.predictor = predictor

    def name(self):
        return "rolling_%s" % self.predictor.name()

    def __call__(self, history, date_range):
        ds_start, ds_end = date_range
        ds_len = (ds_end - ds_start).days + 1

        if history.shape[0] > 0:
            history = history.rolling("%dD" % ds_len, min_periods=1).sum()
        return self.predictor(history, (ds_end, ds_end))

class HistoryDayOfWeek:
    def __init__(self, predictor):
        self.predictor = predictor

    def name(self):
        return "dow_%s" % self.predictor.name()

    def __call__(self, history, date_range):
        ds_start, ds_end = date_range
        ds_len = (ds_end - ds_start).days + 1

        yhat = 0
        for days in range(0, ds_len):
            target_ds = ds_start + timedelta(days=days)
            if history.shape[0] == 0:
                filtered_history = history
            else:
                filtered_history = history[history.index.weekday == target_ds.weekday()]
            n = self.predictor(history=filtered_history, date_range=[target_ds, target_ds])
            if not np.isnan(n):
                yhat += n
            eprint("HistoryDayOfWeek", yhat, filtered_history)
        return yhat

def constraint(type, min=None, max=None, values=None):
  return {"type": type, "min": min, "max": max, "values": values}

class HistoryThreshold:
    def __init__(self,
                 keep_threshold: constraint(float, min=0.2, max=1.0),
                 predictor):
        self.keep_threshold = keep_threshold
        self.predictor = predictor

    def name(self):
        if self.keep_threshold >= 1:
            return self.predictor.name()
        return ("keep-%.2f_" % self.keep_threshold) + self.predictor.name()

    def __call__(self, history=None, **kwargs):
        if history.shape[0] > 0:
            cumsum = history.cumsum()
            total = cumsum.tail(1)['y'][0]
            drop_threshold = 1 - self.keep_threshold
            to_drop = cumsum[cumsum['y'] < (total * drop_threshold)]
            if to_drop.shape[0] > 0:
                ds_keep_threshold = to_drop.index[0]
                history = history.ix[history.index >= ds_keep_threshold]
        return self.predictor(history=history, **kwargs)

class HistoryCliff:
    def __init__(self, max_days_back, predictor):
        self.max_days_back = max_days_back
        self.predictor = predictor

    def name(self):
        if self.max_days_back < 0:
            return self.predictor.name()
        return "%dd_%s" % (self.max_days_back, self.predictor.name())

    def __call__(self, history=None, **kwargs):
        if self.max_days_back >= 0:
            if "date_range" in kwargs:
                target_date = kwargs["date_range"][0]
            else:
                target_date = kwargs["arrival_ds"]
            horizon = target_date - timedelta(days=self.max_days_back)
            history = history[history.index >= horizon]
        return self.predictor(history=history, **kwargs)

class HistoryRelativeCliff:
    def __init__(self, max_days_back, predictor):
        self.max_days_back = max_days_back
        self.predictor = predictor

    def name(self):
        if self.max_days_back < 0:
            return self.predictor.name()
        return "%dd_%s" % (self.max_days_back, self.predictor.name())

    def __call__(self, history=None, **kwargs):
        if self.max_days_back >= 0 and history.shape[0] > 0:
            target_date = history.index[-1]
            horizon = target_date - timedelta(days=self.max_days_back)
            history = history[history.index >= horizon]
        return self.predictor(history=history, **kwargs)

class Amnesia:
  def __init__(self, today):
    self.today = today

  def __call__(self, history):
    if history.shape[0] > 0:
      history = history.ix[history.index < self.today]
    return history

class Trailing:
  def __init__(self,
               keep_threshold: constraint(float, min=0.2, max=1.0)):
    self.keep_threshold = keep_threshold

  def name(self):
      return "keep-%.2f_" % self.keep_threshold

  def __call__(self, history):
    if history.shape[0] > 0:
      cumsum = history.cumsum()
      total = cumsum.tail(1)['y'][0]
      drop_threshold = 1 - self.keep_threshold
      to_drop = cumsum[cumsum['y'] < (total * drop_threshold)]
      if to_drop.shape[0] > 0:
        ds_keep_threshold = to_drop.index[0]
        history = history.ix[history.index >= ds_keep_threshold]
      return history

class Max:
    def __init__(self, *alternatives):
        self.alternatives = alternatives

    def name(self):
        alt_names = [a.name() if callable(a) else str(a) for a in self.alternatives]
        return "max-%s" % "_".join(alt_names)

    def __call__(self, **kwargs):
        results = []
        for a in self.alternatives:
            result = a(**kwargs) if callable(a) else a
            if not np.isnan(result):
                results.append(result)

        results.sort()
        return results[-1]

def NoNan(a, b):
  if np.isnan(a):
    return b
  return a

def Max2(*alternatives):
      results = [a for a in alternatives if not np.isnan(a)]
      results.sort()
      return results[-1]

def Max3(*alternatives):
    if np.any(np.isnan(alternatives)):
        return np.nan
    alternatives = list(alternatives)
    alternatives.sort()
    return alternatives[-1]

def Sum(df):
  if df.hasnans:
    return np.nan
  return df.sum()

import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np

from fbprophet import Prophet
from datetime import timedelta

import multiprocessing

def Prophet3(hit_rate: constraint(float, min=0.4, max=0.99),
             history,
             start_ds,
             end_ds,
             changepoint_prior_scale: constraint(float, min=0.001, max=100) = 0.05,
             seasonality_prior_scale: constraint(float, min=0.001, max=100) = 10,
             holidays_prior_scale: constraint(float, min=0.001, max=100) = 10,
             log_bump=0,
             use_holidays=False,
             weekly_seasonality=True):

#     def _impl(queue, *args):
#       queue.put(_Prophet3(*args))

#     q = multiprocessing.Queue()
#     p = multiprocessing.Process(target=_impl, args=(
#         q,
#         hit_rate,
#         history,
#         start_ds,
#         end_ds,
#         changepoint_prior_scale,
#         seasonality_prior_scale,
#         holidays_prior_scale,
#         log_bump,
#         use_holidays,
#         weekly_seasonality))
#     p.start()
#     result = q.get()
#     p.join()
#     return result

# def _Prophet3(hit_rate,
#               history,
#               start_ds,
#               end_ds,
#               changepoint_prior_scale,
#               seasonality_prior_scale,
#               holidays_prior_scale,
#               log_bump,
#               use_holidays,
#               weekly_seasonality):
  holidays = None
  if use_holidays:
      holidays = pd.DataFrame(
          [
              # ['Migration', '2016-10-13', -2, 0],
              # ['Thanksgiving', '2016-11-24', -2, 1],
              # ['Christmas', '2016-12-25', -2, 1],
              ["New Year's", '2016-12-29', -2, 0],
              ["Jan Event 1", '2017-01-30', -2, 0],
              ["Feb Event 1", '2017-02-26', -1, 0],
              # ["Spring Break", '2017-04-20', -2, 2],
              # ["Spring Break", '2017-04-27', -2, 2],
          ],
          columns=['holiday', 'ds', 'lower_window', 'upper_window'])

  xform = lambda x: np.log(x + log_bump)
  unxform = lambda x: np.exp(x) - log_bump

  if hit_rate == 0.5:
      forecast_count_col = 'yhat'
      forecast_interval = 0.8 # this doesn't matter...
  else:
      if hit_rate < 0.5:
          forecast_count_col = 'yhat_lower'
          use_hit_rate = 1.0 - hit_rate
      else:
          forecast_count_col = 'yhat_upper'
          use_hit_rate = hit_rate

      forecast_interval = (2.0 * use_hit_rate) - 1.0 # To get interval width whose upper bound is hit_rate.

  def _fill(start_ds, end_ds, y):
      df = pd.DataFrame(data=[])
      df['ds'] = pd.date_range(start_ds, end_ds)
      df['y'] = y
      return df['y']

  if history is None:
      return _fill(start_ds, end_ds, np.nan)

  history = history[history['y'] != 0]
  if history.shape[0] <= 7:
      return _fill(start_ds, end_ds, np.nan)

  if np.all(history['y'] == unxform(0)):
      return _fill(start_ds, end_ds, unxform(0))

  df = history
  df = df.reset_index()
  # df = df.copy(deep=False)
  df['y'] = df['y'].apply(xform)

  m = Prophet(
      changepoint_prior_scale=changepoint_prior_scale,
      seasonality_prior_scale=seasonality_prior_scale,
      holidays_prior_scale=holidays_prior_scale,
      weekly_seasonality=weekly_seasonality,
      yearly_seasonality=False,
      holidays=holidays,
      interval_width=forecast_interval,
      # mcmc_samples=10,
  )

  try:
      m.fit(df)
  except Exception as e:
      eprint(e)
      eprint(df)
      raise e
      return _fill(start_ds, end_ds, np.nan)

  days_ahead = (end_ds - history.index.max()).days
  forecast = m.predict(m.make_future_dataframe(periods=days_ahead))

  range_forecast = forecast
  range_forecast = range_forecast[range_forecast['ds'] >= start_ds]
  range_forecast = range_forecast[range_forecast['ds'] <= end_ds]
  return range_forecast[forecast_count_col].apply(unxform)

