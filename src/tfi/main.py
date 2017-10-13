
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import importlib
import inspect
import os
import sys
import tensorflow as tf
import tfi


from collections import OrderedDict
from functools import partial

class ModelSpecifier(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 **kwargs):
        super(ModelSpecifier, self).__init__(
            option_strings=option_strings,
            dest=dest,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, None)
            return

        init = {}
        if values.startswith('@'):
            klass = tfi.saved_model.as_class(values[1:])
        else:
            # Expecting values to be something like "module.class(kwargs)"
            pre_initargs, *rest = values.split("(", 1)
            *module_fragments, classname = pre_initargs.split(".")
            module_name = ".".join(module_fragments)
            module = importlib.import_module(module_name)
            klass = getattr(module, classname)
            if rest:
                init = eval("dict(%s" % rest[0])

        sig = inspect.signature(klass)
        needed = OrderedDict(sig.parameters.items())
        result = klass
        # Only allow unspecified values to be given.
        for k in init.keys():
            del needed[k]
        result = partial(klass, **init)

        setattr(namespace, self.dest, result)
        setattr(namespace, "%s_raw" % self.dest, values)
        setattr(namespace, "%s_kwargs" % self.dest, needed)

class _HelpAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(_HelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        if namespace.specifier:
            setattr(namespace, 'help', True)
            return
        parser.print_help()
        parser.exit()

parser = argparse.ArgumentParser(prog='tfi', add_help=False)
parser.add_argument('--export', type=str, help='path to export to')
parser.add_argument('--interactive', '-i', default=False, action='store_true', help='Start interactive session')
parser.add_argument('specifier', type=str, default=None, nargs='?', action=ModelSpecifier, help='fully qualified class name to instantiate')
parser.add_argument('method', type=str, nargs='?', help='name of method to run')
parser.add_argument('--help', '-h', dest='help', default=None, action=_HelpAction, help="Show help")


# TODO(adamb)
#     And let's add basic text --doc output.
#     Then we'll add support for training a model locally ... (which?)
#     Then we'll add support for training a model ELSEWHERE.

def split_list(l, delim):
    for ix in range(0, len(l)):
        if l[ix] == '--':
            return l[:ix], l[ix+1:]
    return l, []

def argparser_for_fn(fn_name, needed_params, argparse_options_fn):
    empty = inspect.Parameter.empty
    parse = argparse.ArgumentParser(prog=fn_name)
    for name, param in needed_params.items():
        parse.add_argument(
                '--%s' % name,
                required=param.default is empty,
                default=None if param.default is empty else param.default,
                **argparse_options_fn(param))
    return parse

def apply_fn_args(fn_name, needed_params, param_types, fn, raw_args):
    p = argparser_for_fn(fn_name, needed_params, param_types)
    kw = vars(p.parse_args(raw_args))
    return fn(**kw)

def configure_repl(repl):
    from ptpython.prompt_style import PromptStyle
    from pygments.token import Token
    class TfiPrompt(PromptStyle):
        def in_tokens(self, cli):
            return [
                (Token.Prompt, 'tfi> '),
            ]
        def in2_tokens(self, cli, width):
            return [
                (Token.Prompt.Dots, '...'),
            ]
        def out_tokens(self, cli):
            return []
    repl.all_prompt_styles['tfi'] = TfiPrompt()
    repl.prompt_style = 'tfi'

    repl.show_status_bar = False
    repl.confirm_exit = False

    import six
    from prompt_toolkit.layout.utils import token_list_width
    from ptpython.repl import _lex_python_result
    def _execute(self, cli, line):
        """
        Evaluate the line and print the result.
        """
        output = cli.output

        def compile_with_flags(code, mode):
            " Compile code with the right compiler flags. "
            return compile(code, '<stdin>', mode,
                           flags=self.get_compiler_flags(),
                           dont_inherit=True)

        if line.lstrip().startswith('\x1a'):
            # When the input starts with Ctrl-Z, quit the REPL.
            cli.exit()

        elif line.lstrip().startswith('!'):
            # Run as shell command
            os.system(line[1:])
        else:
            # Try eval first
            try:
                code = compile_with_flags(line, 'eval')
                result = eval(code, self.get_globals(), self.get_locals())

                locals = self.get_locals()
                locals['_'] = locals['_%i' % self.current_statement_index] = result

                if result is not None:
                    out_tokens = self.get_output_prompt_tokens(cli)

                    tensor = tfi.data.as_tensor(result, None, None)
                    accept_mimetypes = {"image/png": tfi.data.terminal.imgcat, "text/plain": lambda x: x}
                    result_val = tfi.data._encode(tensor, accept_mimetypes)
                    if result_val is None:
                        result_val = result
                    try:
                        result_str = '%r\n' % (result_val, )
                    except UnicodeDecodeError:
                        # In Python 2: `__repr__` should return a bytestring,
                        # so to put it in a unicode context could raise an
                        # exception that the 'ascii' codec can't decode certain
                        # characters. Decode as utf-8 in that case.
                        result_str = '%s\n' % repr(result_val).decode('utf-8')

                    # Align every line to the first one.
                    line_sep = '\n' + ' ' * token_list_width(out_tokens)
                    result_str = line_sep.join(result_str.splitlines()) + '\n'

                    # Write output tokens.
                    if '\033' in result_str:
                        output.write_raw(result_str)
                    else:
                        out_tokens.extend(_lex_python_result(result_str))
                        cli.print_tokens(out_tokens)
            # If not a valid `eval` expression, run using `exec` instead.
            except SyntaxError:
                code = compile_with_flags(line, 'exec')
                six.exec_(code, self.get_globals(), self.get_locals())

            output.flush()

    repl._execute = partial(_execute, repl)

def run(argns, remaining_args):
    model = None
    exporting = argns.export is not None
    if argns.specifier:
        hparam_raw_args, method_raw_args = split_list(remaining_args, '--')

        model = apply_fn_args(
                argns.specifier_raw,
                argns.specifier_kwargs,
                lambda param: {'type': type(param.default) if param.annotation is inspect.Parameter.empty else param.annotation},
                argns.specifier,
                hparam_raw_args)

        if method_raw_args:
            method_name, method_raw_args = method_raw_args[0], method_raw_args[1:]
            method = getattr(model, method_name)
            apply_fn_args(
                    method_name,
                    inspect.signature(method).parameters,
                    lambda param: {'help': "%s %s" % (tf.as_dtype(param.annotation.dtype).name, tf.TensorShape(param.annotation.tensor_shape)) },
                    method,
                    method_raw_args)
        else:
            argns.interactive = not exporting
    else:
        argns.interactive = True

    if argns.interactive:
        from ptpython.repl import embed
        g = dict(globals())
        g['tfi'] = tfi
        tfi.display.install_hook()
        if model is not None:
            for n, m in inspect.getmembers(model, predicate=inspect.ismethod):
                g[n] = m
        embed(g, locals(), configure=configure_repl)

    if argns.export:
        tfi.saved_model.export(argns.export, model)

if __name__ == '__main__':
    run(*parser.parse_known_args(sys.argv[1:]))
