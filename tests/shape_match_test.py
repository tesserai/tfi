import unittest

from tfi.tensor.codec import _shape_match_score

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

    setattr(ShapeMatchScoreTest,
            'test_%s' % name,
            partialmethod(do_test, args, expect))

if __name__ == '__main__':
    unittest.main()
