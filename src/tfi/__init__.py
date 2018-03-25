
__version__ = "0.6"

__title__ = "tfi"
__description__ = "Use any TensorFlow model in a single line of code"
__uri__ = "https://github.com/ajbouh/tfi"
__doc__ = __description__ + " <" + __uri__ + ">"

__author__ = "Adam Bouhenguel"
__email__ = "adam@bouhenguel.com"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2017 Adam Bouhenguel"

def tf_make_logdir_fn(datetime):
    dir = datetime.strftime("/tmp/tfi/tf/%F_%H-%M-%S")
    def logdir_fn(run_id=None):
        return dir
    return logdir_fn
