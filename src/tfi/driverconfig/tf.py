def make_logdir_fn(datetime):
    dir = datetime.strftime("/tmp/tfi/tf/%F_%H-%M-%S")
    def logdir_fn(run_id=None):
        return dir
    return logdir_fn


def make_session(graph):
    import tensorflow as tf
    config = tf.ConfigProto(
        device_count={'CPU' : 1, 'GPU' : 0},
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options={'allow_growth': True},
    )
    return tf.Session(
        graph=graph,
        config=config,
    )
