import sys

import numpy
import tfi.data
import tfi.data.terminal

def _wrap_displayhook(original):
    def _displayhook(o):
        tfi.data.terminal_write(o)
        __builtins__['_'] = o
        return

        # return original(o)
    return _displayhook

def install_hook():
    sys.displayhook = _wrap_displayhook(sys.displayhook)
