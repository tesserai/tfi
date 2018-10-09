import unittest

import tfi.driver.tf as tf_driver
import tensorflow as tf
import tempfile
import os.path

class SavedModelTest(unittest.TestCase):
    def test_simple(self):
        class Math(tf_driver.Model):
            def __init__(self):
                self._x = tf.placeholder(tf.float32)
                self._y = tf.placeholder(tf.float32)
                self._w = self._x + self._y
                self._z = self._x * self._y
            def add(self, *, x: self._x, y: self._y) -> {'sum': self._w}:
                pass
            def mult(self, *, x: self._x, y: self._y) -> {'prod': self._z}:
                pass

        m = Math()

        # Prove that simple add and multiply work.
        self.assertEqual(3.0, m.add(x=1.0, y=2.0)['sum'])
        self.assertEqual(2.0, m.mult(x=1.0, y=2.0)['prod'])

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "math.saved_model")
            tf_driver.dump(path, m)
            # Prove that we can save it.
            # Prove that we can restore it to a new class.

            m2 = tf_driver.load(path)
            # Prove that we can save and restore it again.
            self.assertEqual(3.0, m2.add(x=1.0, y=2.0)['sum'])
            self.assertEqual(2.0, m2.mult(x=1.0, y=2.0)['prod'])

    def test_variables(self):
        # Prove that variables are serialized and restored as expected.
        # Prove that variables are different in different instances.
        pass

if __name__ == '__main__':
    unittest.main()
