import unittest

from as_tensor import as_tensor

class AsTensorTest(unittest.TestCase):
    pass

_FIXTURES = [
    ('string', 'string', [], 'string'),
    ('list', ['string'], [None], ['string']),
    ('list', ['string'], [1], ['string']),
    ('generator', (s for s in ['string']), [1], ['string']),
    ('emptylist', [], [None], []),
    ('emptylist', [], [0], []),
    ('nested_list', [['string'], ['foo']], [2,1], [['string'], ['foo']]),
]
for (name, expect, shape, data) in _FIXTURES:
    def do_test(self):
        result = as_tensor(data, shape)
        self.assertEqual(expect, result)

    setattr(AsTensorTest, 'test_%s' % name, do_test)

if __name__ == '__main__':
    unittest.main()
