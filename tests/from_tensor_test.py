import unittest

from tfi.as_tensor import from_tensor, _shape_match_score

from functools import partialmethod

class ShapeMatchScoreTest(unittest.TestCase):
    pass

_SHAPE_MATCH_SCORE_FIXTURES = [
    # (name, (src, xfrm), score),
    ('exact_any', (None, None), (2, 0)),
    ('exact_scalar', ([], []), (2, 0)),
    ('exact_1d', ([3], [3]), (2, 1)),
    ('exact_1d_unknown', ([None], [None]), (2, 1)),
    ('exact_2d_unknown', ([None, 3], [None, 3]), (2, 2)),

    ('partial_any_scalar', (None, []), (1, 0)),
    ('partial_any_1d', (None, [3]), (1, 0)),
    ('partial_any_1d_unknown', (None, [None]), (1, 0)),

    ('partial_scalar_any', ([], None), (1, 0)),
    ('partial_1d_any', ([3], None), (1, 0)),
    ('partial_1d_unknown_any', ([None], None), (1, 0)),

    ('partial_image', ([16, 16, 3], [None, None, 3]), (1, 1)),
    ('partial_image_array', ([6, 16, 16, 3], [None, None, 3]), (1, 1)),

    ('none_scalar_1d', ([], [3]), (0, 0)),
    ('none_scalar_2d', ([], [3, 2]), (0, 0)),
    ('none_scalar_1d_unknown', ([], [None]), (0, 0)),
    ('none_scalar_2d_unknown', ([], [None, None]), (0, 0)),
]
for (name, args, expect) in _SHAPE_MATCH_SCORE_FIXTURES:
    def do_test(self, args, expect):
        result = _shape_match_score(*args)
        self.assertEqual(expect, result)

    setattr(ShapeMatchScoreTest, 'test_%s' % name, partialmethod(do_test, args, expect))


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
        result = from_tensor(*args)
        npt.assert_array_equal(expect, result)

    setattr(FromTensorTest, 'test_%s' % name, partialmethod(do_test, args, expect))



if __name__ == '__main__':
    unittest.main()
