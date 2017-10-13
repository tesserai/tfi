from ptpython.repl import embed
from ptpython.prompt_style import PromptStyle
from pygments.token import Token
import six
from functools import partial
import inspect
from prompt_toolkit.layout.utils import token_list_width
from ptpython.repl import _lex_python_result

import tfi.data

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

                    tensor = tfi.maybe_as_tensor(result, None, None)
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

def run(globals=None, locals=None, history_filename=None, model=None):
    g = dict(globals)
    # g['tfi'] = tfi
    if model is not None:
        for n, m in inspect.getmembers(model, predicate=inspect.ismethod):
            g[n] = m
    embed(g, locals,
          configure=_configure_repl,
          history_filename=history_filename)