import unittest
from functools import partialmethod
from collections import OrderedDict

as_tensor_impls = OrderedDict()
# try:
from tfi.driver.pytorch.tensor_codec import as_tensor as pytorch_as_tensor
as_tensor_impls['pytorch'] = pytorch_as_tensor
# except ImportError:
#     pass

# try:
from tfi.driver.tf.tensor_codec import as_tensor as tf_as_tensor
as_tensor_impls['tf'] = tf_as_tensor
# except ImportError:
#     pass

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
for as_tensor_impl_name, as_tensor in as_tensor_impls.items():
    for (name, *rest) in _FIXTURES:
        def do_test(self, expect, shape, dtype, data):
            result = as_tensor(data, shape, dtype)
            self.assertEqual(expect, result)

        setattr(AsTensorTest,
                'test_%s_%s' % (as_tensor_impl_name, name),
                partialmethod(do_test, *rest))

if __name__ == '__main__':
    unittest.main()
