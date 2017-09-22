import unittest

from tfi.as_tensor import as_tensor

from functools import partialmethod

class AsTensorTest(unittest.TestCase):
    pass

_FIXTURES = [
    ('string', 'string', [], str, 'string'),
    ('list', ['string'], [None], str, ['string']),
    ('list', ['string'], [1], str, ['string']),
    ('generator', (s for s in ['string']), [1], str, ['string']),
    ('emptylist', [], [None], float, []),
    ('emptylist', [], [0], float, []),
    ('nested_list', [['string'], ['foo']], [2,1], str, [['string'], ['foo']]),
]
for (name, *rest) in _FIXTURES:
    def do_test(self, expect, shape, dtype, data):
        result = as_tensor(data, shape, dtype)
        self.assertEqual(expect, result)

    setattr(AsTensorTest,
            'test_%s' % name,
            partialmethod(do_test, *rest))

if __name__ == '__main__':
    unittest.main()
