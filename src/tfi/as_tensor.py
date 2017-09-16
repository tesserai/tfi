import tensorflow as tf
import collections

def as_tensor(object, shape, dtype):
    if isinstance(object, str):
        return object
    if isinstance(object, collections.Iterable):
        if shape is None:
            max_len = None
        elif len(shape) >= 1:
            max_len = shape[0]
        else:
            raise Exception("Shape is %s. Didn't expect iterable: %s" % (shape, object))
        sub_shape = shape[1:]
        r = []
        cur_len = 0
        for o in object:
            r.append(as_tensor(o, sub_shape, dtype))
            cur_len += 1
            if max_len is not None and cur_len > max_len:
                raise Exception("Iterable too long. Expected exactly %s elements from: %s" % (max_len, object))
        return tf.stack(r)
    if hasattr(object, '__tensor__'):
        return object.__tensor__(shape, dtype)
    return object
