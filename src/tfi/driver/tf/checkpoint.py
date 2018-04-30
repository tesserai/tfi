import os
import glob
import tensorflow as tf

def restore(checkpoint_file_prefix):
    graph = tf.get_default_graph()
    session = tf.get_default_session()

    var_list = {}
    reader = tf.train.NewCheckpointReader(checkpoint_file_prefix)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
      try:
        tensor = graph.get_tensor_by_name(key + ":0")
      except KeyError:
        # This tensor doesn't exist in the graph (for example it's
        # 'global_step' or a similar housekeeping element) so skip it.
        continue
      var_list[key] = tensor
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(session, checkpoint_file_prefix)

# TODO(adamb) Add some reasonable testing.
def find(checkpoint_glob_prefix, ext=".ckpt"):
    checkpoint_files = glob.glob("%s*%s*" % (checkpoint_glob_prefix, ext))

    if len(checkpoint_files) == 0:
        return None

    if len(checkpoint_files) == 1:
        return checkpoint_files[0]

    # Expect to always find ext, since it was in glob.
    return list(set([f[:f.find(ext) + len(ext)] for f in checkpoint_files]))[0]
