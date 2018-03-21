from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from pygments.lexers.sql import SqlLexer

import os, sys, time
import apsw

from collections import OrderedDict
import tensorflow as tf
from ast import literal_eval as _literal_eval
import math
import json
import tfi.tf
from tfi.resolve.model import resolve_auto as _resolve_auto
import tfi.watch
import traceback
from cli_helpers.tabular_output import TabularOutputFormatter

# /** .. method:: UpdateDeleteRow(rowid)

#   Delete the row with the specified *rowid*.

#   :param rowid: 64 bit integer
# */
# /** .. method:: UpdateInsertRow(rowid, fields)  -> newrowid

#   Insert a row with the specified *rowid*.

#   :param rowid: :const:`None` if you should choose the rowid yourself, else a 64 bit integer
#   :param fields: A tuple of values the same length and order as columns in your table

#   :returns: If *rowid* was :const:`None` then return the id you assigned
#     to the row.  If *rowid* was not :const:`None` then the return value
#     is ignored.
# */
# /** .. method:: UpdateChangeRow(row, newrowid, fields)

#   Change an existing row.  You may also need to change the rowid - for example if the query was
#   ``UPDATE table SET rowid=rowid+100 WHERE ...``

#   :param row: The existing 64 bit integer rowid
#   :param newrowid: If not the same as *row* then also change the rowid to this.
#   :param fields: A tuple of values the same length and order as columns in your table
# */

# /** .. method:: FindFunction(name, nargs)

#   Called to find if the virtual table has its own implementation of a
#   particular scalar function. You should return the function if you
#   have it, else return None. You do not have to provide this method.

#   This method is called while SQLite is `preparing
#   <https://sqlite.org/c3ref/prepare.html>`_ a query.  If a query is
#   in the :ref:`statement cache <statementcache>` then *FindFunction*
#   won't be called again.  If you want to return different
#   implementations for the same function over time then you will need
#   to disable the :ref:`statement cache <statementcache>`.

#   :param name: The function name
#   :param nargs: How many arguments the function takes

#   .. seealso::

#     * :meth:`Connection.overloadfunction`

# */

class GeneratorFnSource(object):
    _PythonTypeForDtype = {
        tf.float16: float, # 16-bit half-precision floating-point.
        tf.float32: float, # 32-bit single-precision floating-point.
        tf.float64: float, # 64-bit double-precision floating-point.
        tf.bfloat16: float, # 16-bit truncated floating-point.
        tf.complex64: complex, # 64-bit single-precision complex.
        tf.complex128: complex, # 128-bit double-precision complex.
        tf.int8: int, # 8-bit signed integer.
        tf.uint8: int, # 8-bit unsigned integer.
        tf.uint16: int, # 16-bit unsigned integer.
        tf.uint32: int, # 32-bit unsigned integer.
        tf.uint64: int, # 64-bit unsigned integer.
        tf.int16: int, # 16-bit signed integer.
        tf.int32: int, # 32-bit signed integer.
        tf.int64: int, # 64-bit signed integer.
        tf.bool: bool, # Boolean.
        tf.string: str, # String.
        tf.qint8: int, # Quantized 8-bit signed integer.
        tf.quint8: int, # Quantized 8-bit unsigned integer.
        tf.qint16: int, # Quantized 16-bit signed integer.
        tf.quint16: int, # Quantized 16-bit unsigned integer.
        tf.qint32: int, # Quantized 32-bit signed integer.
        tf.resource: str, # Handle to a mutable resource.
    }

    def __init__(self, implfn):
        self._implfn = implfn

    def Create(self, db, modulename, dbname, tablename, *args):
        args = [_literal_eval(arg) for arg in args]

        input_types, output_types, generatorfn = self._implfn(*args)
        schema = "CREATE TABLE %s(%s);" % (
            tablename,
            ", ".join([
                *[k for k in output_types.keys()],
                *["%s HIDDEN" % k for k in input_types.keys()],
            ]),
        )

        return schema, GeneratorFnTable(input_types, output_types, generatorfn)
    Connect = Create


constraint_ops = {
    2: 'SQLITE_INDEX_CONSTRAINT_EQ',
    4: 'SQLITE_INDEX_CONSTRAINT_GT',
    8: 'SQLITE_INDEX_CONSTRAINT_LE',
    16: 'SQLITE_INDEX_CONSTRAINT_LT',
    32: 'SQLITE_INDEX_CONSTRAINT_GE',
    64: 'SQLITE_INDEX_CONSTRAINT_MATCH',
    65: 'SQLITE_INDEX_CONSTRAINT_LIKE',        # 3.10.0 and later
    66: 'SQLITE_INDEX_CONSTRAINT_GLOB',        # 3.10.0 and later
    67: 'SQLITE_INDEX_CONSTRAINT_REGEXP',      # 3.10.0 and later
    68: 'SQLITE_INDEX_CONSTRAINT_NE',          # 3.21.0 and later
    69: 'SQLITE_INDEX_CONSTRAINT_ISNOT',       # 3.21.0 and later
    70: 'SQLITE_INDEX_CONSTRAINT_ISNOTNULL',   # 3.21.0 and later
    71: 'SQLITE_INDEX_CONSTRAINT_ISNULL',      # 3.21.0 and later
    72: 'SQLITE_INDEX_CONSTRAINT_IS',          # 3.21.0 and later
    1: 'SQLITE_INDEX_SCAN_UNIQUE',             # Scan visits at most 1 row
}

class GeneratorFnTable(object):
    def __init__(self, input_types, output_types, generatorfn):
        inputs = list(input_types.keys())
        outputs = list(output_types.keys())
        self._columns = outputs + inputs
        self._columntypes = [
            *[output_types[k] for k in outputs],
            *[input_types[k] for k in inputs],
        ]
        self._ishidden = ([False] * len(outputs)) + ([True] * len(inputs))
        self._generatorfn = generatorfn
        # print("GeneratorFnTable.__init__", self._columns, self._ishidden, self._generatorfn)

    def BestIndex(self, constraints, orderbys):
        try:
            # Return
            #   0: constraints used (default None)
            #     This must either be None or a sequence the same length as
            #     constraints passed in. Each item should be as specified above
            #     saying if that constraint is used, and if so which constraintarg
            #     to make the value be in your :meth:`VTCursor.Filter` function.

            #   1: index number (default zero)
            #     This value is passed as is to :meth:`VTCursor.Filter`

            #   2: index string (default None)
            #     This value is passed as is to :meth:`VTCursor.Filter`

            #   3: orderby consumed (default False)
            #     Return

            #   4: estimated cost (default a huge number)
            #     Approximately how many disk operations are needed to provide the
            #     results. SQLite uses the cost to optimise queries. For example if
            #     the query includes *A or B* and A has 2,000 operations and B has 100
            #     then it is best to evaluate B before A.

            constraints_used = list(range(len(constraints))) # None or [constraintarg, ...] with same length as constraints. Elements are None or value to pass to Filter
            index_num = len(constraints) # passed to Filter
            index_str = json.dumps(constraints) # passed to Filter
            orderby_consumed = False # True if your output will be in exactly the same order as the orderbys passed in
            unique_hidden_eq_constraints = len({
                col
                for col, op in constraints
                if op == 2 and self._ishidden[col]
            })
            estimated_cost = len(self._columns) - unique_hidden_eq_constraints # Approximately how many disk operations needed to provide the results. Used to optimize queries.
            best_index_result = (constraints_used, index_num, index_str, orderby_consumed, estimated_cost)
            # print("best_index_result", best_index_result)
            return best_index_result
        except:
            traceback.print_exc()

    def FindFunction(self, name, nargs):
        print("FindFunction", name, nargs)
        return None

    def Open(self):
        return GeneratorFnCursor(self._columns, self._columntypes, self._generatorfn)

    def Disconnect(self):
        pass

    Destroy=Disconnect


class ParallelDictWalker(object):
    @staticmethod
    def walk(d):
        w = ParallelDictWalker(d)
        length = len(d[list(d.keys())[0]])
        for index in range(length):
            w.index = index
            yield w

    def __init__(self, d):
        self.d = d
        self.index = -1

    def __contains__(self, key):
        return key in d

    def __getitem__(self, key):
        r = self.d[key][self.index]
        # print("%s[%s, %s] = %s" % (self.d, self.index, key, r))
        return r


class GeneratorFnCursor(object):
    def __init__(self, colnames, coltypes, generatorfn):
        self._colnames = colnames
        self._coltypes = coltypes
        self._generatorfn = generatorfn
        self._generator = None
        self._iseof = False
        self._current = None

    def Filter(self, indexnum, indexname, constraintargs):
        constraints = json.loads(indexname)
        self._constraint_map = {
            col: constraintarg
            for (col, constraint_op), constraintarg in zip(constraints, constraintargs)
        }

        kwarg = {
            self._colnames[k]: v
            for k, v in self._constraint_map.items()
        }
        self._generator = self._generatorfn(**kwarg)
        self._iseof = False
        self.Next()

        # print("GeneratorFnCursor.Filter", indexnum, indexname, constraintargs, self.constraint_map)

    def Eof(self):
        # print("GeneratorFnCursor.Eof")
        return self._iseof

    def Rowid(self):
        # print("GeneratorFnCursor.Rowid")
        return self._table.data[self._pos][0]

    def Column(self, col):
        if col in self._constraint_map:
            return self._constraint_map[col]

        r = self._current[self._colnames[col]]
        # print("GeneratorFnCursor.Column", col, r)
        r = self._coltypes[col](r)
        return r

    def Next(self):
        try:
            self._current = next(self._generator)
        except StopIteration:
            self._iseof = True

    def Close(self):
        # print("GeneratorFnCursor.Close")
        pass


def main():
    ar = tfi.watch.AutoRefresher()

    def tfi_model_sqlite_adapter(path, *args):
        resolution = _resolve_auto(path)
        m = resolution['model_fn'](*args)

        if 'refresh_fn' in resolution:
            ar.watch(resolution['source'], resolution['source_sha1hex'], resolution['refresh_fn'])

        fn = tfi.tf.estimator_method(m, 'infer')
        generatorfn = lambda **kwarg: ParallelDictWalker.walk(fn(**kwarg))

        annotations = fn.__annotations__
        return_annotation = annotations['return']
        output_types = OrderedDict([
            (k, GeneratorFnSource._PythonTypeForDtype[return_annotation[k].dtype])
            for k in sorted(return_annotation.keys())
        ])
        input_types = OrderedDict([
            (k, GeneratorFnSource._PythonTypeForDtype[annotations[k].dtype])
            for k in sorted(annotations.keys())
            if k != 'return'
        ])
        return input_types, output_types, generatorfn

    ar.start()

    connection = apsw.Connection("dbfile")
    cursor = connection.cursor()

    connection.createmodule("tfi", GeneratorFnSource(tfi_model_sqlite_adapter))

    formatter = TabularOutputFormatter()

    while 1:
        try:
            user_input = prompt(
                u'SQL> ',
                history=FileHistory('history.txt'),
                auto_suggest=AutoSuggestFromHistory(),
                # completer=SQLCompleter(),
                lexer=SqlLexer,
            )

            r = cursor.execute(user_input)
            headers = [h for h, d in cursor.getdescription()]
            for x in formatter.format_output(r, headers, format_name='fancy_grid'):
                print(x)
        except apsw.ExecutionCompleteError:
            # Our naive call to getdescription() can fail for commands that don't have results.
            continue
        except apsw.SQLError:
            traceback.print_exc()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == '__main__':
    main()