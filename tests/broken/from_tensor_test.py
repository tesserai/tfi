import unittest

from collections import OrderedDict

# impls = OrderedDict()
# # try:
# from tfi.driver.pytorch.tensor_codec import from_tensor as pytorch_from_tensor
# impls['pytorch'] = pytorch_from_tensor
# # except ImportError:
# #     pass

# # try:
# from tfi.driver.tf.tensor_codec import from_tensor as tf_from_tensor
# impls['tf'] = tf_from_tensor
# # except ImportError:
# #     pass

from functools import partialmethod

import tfi.tensor.codec

class FromTensorTest(unittest.TestCase):
    pass

import numpy as np
import numpy.testing as npt
mappings = [
    (np.int8, (None, None, 3), lambda x: 'int8 image'),
    (np.float32, (None, None, 3), lambda x: 'float32 image'),
]

_FROM_TENSOR_FIXTURES = [
    # (name, (tensor, xfrm), score),
    ('nothing', (None, mappings), None),
    ('image_int8', (np.ones([8,8,3], np.int8), mappings), 'int8 image'),
    ('image_float32', (np.ones([8,8,3], np.float32), mappings), 'float32 image'),
    ('image_float64', (np.ones([8,8,3], np.float64), mappings), None),
    ('arr_image_float32', (np.ones([1,8,8,3], np.float32), mappings), np.array(['float32 image'])),
    ('arr2_image_float32', (np.ones([2,8,8,3], np.float32), mappings), np.array(['float32 image', 'float32 image'])),
    ('arr_2d_image_float32', (np.ones([1,1,8,8,3], np.float32), mappings), np.array([['float32 image']])),
    ('arr2_2d_image_float32', (np.ones([2,1,8,8,3], np.float32), mappings), np.array([['float32 image'], ['float32 image']])),
]
for (name, args, expect) in _FROM_TENSOR_FIXTURES:
    def do_test(self, args, expect):
        result = tfi.tensor.codec.encode(*args)
        npt.assert_array_equal(expect, result)

    setattr(FromTensorTest,
            'test_%s' % name,
            partialmethod(do_test, args, expect))



if __name__ == '__main__':
    unittest.main()
