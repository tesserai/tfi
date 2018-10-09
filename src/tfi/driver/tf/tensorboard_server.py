import logging as base_logging
import os
import socket
import sys
from werkzeug import serving

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging

from tensorboard.program import TensorBoard

def main(tb_logdir, tb_host=None, tb_port=None, tb_debug=True, tb_purge_orphaned_data=False, tb_reload_interval=5, tb_on_ready_fn=None):
  if tb_host is None:
    tb_host = '127.0.0.1'

  if tb_port is None:
    tb_port = 6006

  argv = [
    sys.argv[0],
    "--logdir", tb_logdir,
    "--reload_interval", str(tb_reload_interval),
    "--purge_orphaned_data", str(tb_purge_orphaned_data),
    "--host", tb_host,
    "--port", str(tb_port),
  ]

  tb = TensorBoard()
  tb.configure(argv)
  url = tb.launch()
  if tb_on_ready_fn:
    tb_on_ready_fn(url)
