from ptpython.repl import embed
from ptpython.prompt_style import PromptStyle
from pygments.token import Token
import six
from functools import partial
import inspect
from prompt_toolkit.layout.utils import token_list_width
from ptpython.repl import _lex_python_result

from io import StringIO
import tfi.data
import tfi.format.iterm2
import tfi.tensor.codec

def _configure_repl(repl):
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

                    try:
                        accept_mimetypes = {"image/png": tfi.format.iterm2.imgcat, "text/plain": lambda x: x}
                        result = tfi.tensor.codec.encode(accept_mimetypes, result)
                    # except TypeError:
                    #     result_val = result
                    # except Exception:
                    #     result_val = result
                    finally:
                        pass

                    # output_s = StringIO()
                    # rprinter = _pretty.RepresentationPrinter(output_s)
                    # rprinter.type_pprinters[numpy.ndarray] = _ndarray_pprint
                    # rprinter.pretty(result)
                    # result_str = output_s.getvalue()

                    # if result_val is None:
                    #     result_val = result
                    try:
                        result_str = '%r\n' % (result, )
                    except UnicodeDecodeError:
                        # In Python 2: `__repr__` should return a bytestring,
                        # so to put it in a unicode context could raise an
                        # exception that the 'ascii' codec can't decode certain
                        # characters. Decode as utf-8 in that case.
                        result_str = '%s\n' % repr(result).decode('utf-8')

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

def run(globals=None, locals=None, history_filename=None, model=None, module=None):
    g = dict(globals)
    if module is not None:
        g['module'] = module

    if model is not None:
        g['m'] = model

        # Print hyperparameters and model methods
        ctorargs = "..."
        if hasattr(model, '__tfi_hyperparameters__'):
            ctorargs = ", ".join(["%s=%s" % (k, v) for k, v in model.__tfi_hyperparameters_dict__().items()])
        print("Initializing environment...")
        print("m = %s(%s)" % (model.__class__.__name__, ctorargs))
        methods = inspect.getmembers(model, predicate=inspect.ismethod)
        for i, (n, m) in enumerate(methods):
            if n.startswith('_'):
                continue
            leading = "└" if i == len(methods) - 1 else "├"
            print("%s m.%s(%s)" % (leading, n, ", ".join(inspect.signature(m).parameters.keys())))
        print()
    embed(g, locals,
          configure=_configure_repl,
          history_filename=history_filename)
