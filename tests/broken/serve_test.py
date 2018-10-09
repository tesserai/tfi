import unittest

from tfi.as_tensor import as_tensor

from functools import partialmethod

class ServeTest(unittest.TestCase):
    pass

_FIXTURES = [
    ('zoo/torchvision/resnet.py:Resnet50', )
]
for (name, *rest) in _FIXTURES:
    def do_model_test(self, path):
        
        # TODO(adamb) Load the model
        # TODO(adamb) For each instance method, run the example in Python

        # TODO(adamb) Export the model
        # TODO(adamb) Load the model
        # TODO(adamb) For each instance method, run the example in Python

        # TODO(adamb) Start the server
        # TODO(adamb) For each instance method, run the curl example
        # TODO(adamb) For each instance method, run the command line example
        # TODO(adamb) If example returns are given, check the result.
        result = as_tensor(data, shape, dtype)
        self.assertEqual(expect, result)

    setattr(AsTensorTest,
            'test_%s' % name,
            partialmethod(do_test, *rest))

if __name__ == '__main__':
    unittest.main()
