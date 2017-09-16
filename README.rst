======================================================
TFI: Use any TensorFlow model in a single line of code
======================================================

.. teaser-begin

TFI provides a simple Python interface to any TensorFlow model. It does this by automatically generating a Python class on the fly.

.. -spiel-end-

Here's an example of using TFI with a SavedModel based on `Inception v1 <https://github.com/tensorflow/models/blob/master/slim/nets/inception_v1.py>`_. This particular SavedModel has a single ``predict`` method and a `SignatureDef <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto>`_ that looks something like: ``predict(images float <?,224,224,3>) -> (categories string <1001>, scores float <?,1001>)``

TFI in Action
=============

.. code-block:: pycon

   >>> import tfi
   >>> InceptionV1 = tfi.saved_model.as_class("./inception_v1.saved_model")

Passing in data
===============

TFI can automatically adapt any data you provide to the shape expected by the graph. Let's take a random photo of a dog I found on the internet...

.. image:: https://www.royalcanin.com/~/media/Royal-Canin/Product-Categories/dog-medium-landing-hero.ashx
   :alt: dog

.. code-block:: pycon

   >>> model = InceptionV1()
   >>> image = tfi.data.file("./dog-medium-landing-hero.jpg")
   >>> result = model.predict(images=[image])
   >>> categories, scores = result.categories, result.scores[0]

If we print the top 5 probabilities, we see:

.. code-block:: pycon

   >>> [(scores[i], categories[i].decode()) for i in scores.argsort()[:-5:-1]]
   [(0.80796158, 'beagle'),
    (0.10305813, 'Walker hound, Walker foxhound'),
    (0.064740285, 'English foxhound'),
    (0.009166114, 'basset, basset hound')]

Not bad!

Creating SavedModel files
=========================

TFI uses the information in a SavedModel's SignatureDefs to automatically generate the proper methods on the resulting class. Method names, keyword argument names, and expected data types are all pulled from the SignatureDef.

In our example above, we start with a .saved_model we already have on disk. In practice .saved_model files are rare beasts that aren't commonly distributed online.

Luckily, TFI makes it easy to generate a SavedModel on disk. Let's start with a simple example and then graduate to the SavedModel used in the original example.

.. code-block:: pycon

   >>> import tfi.saved_model, tensorflow as tf
   >>> class Math(tfi.saved_model.Base):
   >>>     def __init__(self):
   >>>         self._x = tf.placeholder(tf.float32)
   >>>         self._y = tf.placeholder(tf.float32)
   >>>         self._w = self._x + self._y
   >>>         self._z = self._x * self._y
   >>>     def add(self, *, x: self._x, y: self._y) -> {'sum': self._w}:
   >>>         pass
   >>>     def mult(self, *, x: self._x, y: self._y) -> {'prod': self._z}:
   >>>         pass
   >>>
   >>> tfi.saved_model.export("./math.saved_model", Math)

To export a SavedModel in TFI:
1. Define a class that inherits from ``tfi.saved_model.Base``.
2. Within ``__init__``, build a graph using placeholders as input. Save inputs and outputs as attributes on self.
3. Define each method you'd like to be present in the SavedModel as a public instance method.
4. Call ``tfi.saved_model.export`` with the output path and your class

Using the resulting model with TFI is straightforward, even on another machine.

.. code-block:: pycon

   >>> import tfi.saved_model
   >>> math = tfi.saved_model.as_class("./math.saved_model")()
   >>> math.add(x=1.0, y=3.0).sum
   4
   >>> math.mult(x=1.0, y=3.0).prod
   3

If you have trouble with the above, please `file an issue <https://github.com/ajbouh/tfi/issues/new>`_ and ask for clarification.

Now let's see how this works for a larger model that's also been pre-trained, like Google's slim implementation of Inception v1.

The code below is spiritually equivalent to the ``Math`` example above: define a class that inherits from ``tfi.saved_model.Base``, build the graph and load a checkpoint in ``__init__``, add any instance methods you want, and export.

First, let's get the Python code for Inception v1 and put it on ``PYTHONPATH``.

.. code-block:: bash

   $ git clone https://github.com/tensorflow/models
   $ export PYTHONPATH=$PWD/models/slim


.. code-block:: python
   from datasets import dataset_factory
   from nets import nets_factory
   import os.path
   import tensorflow as tf
   import tfi
   from urllib.request import urlretrieve

   CHECKPOINT_URL = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
   CHECKPOINT_FILE = "inception_v1.ckpt"
   CHECKPOINT_SHA256 = "7a620c430fcaba8f8f716241f5148c4c47c035cce4e49ef02cfbe6cd1adf96a6"

   class InceptionV1(tfi.saved_model.Base):
       def __init__(self):
         dataset = dataset_factory.get_dataset('imagenet', 'train', '')
         category_items = list(dataset.labels_to_names.items())
         category_items.sort() # sort by index
         categories = [label for _, label in category_items]
         self._labels = tf.constant(categories)

         network_fn = nets_factory.get_network_fn(
             'inception_v1',
             num_classes=len(categories),
             is_training=False)

         image_size = network_fn.default_image_size
         self._placeholder = tf.placeholder(
                 name='input',
                 dtype=tf.float32,
                 shape=[None, image_size, image_size, 3])

         logits, _ = network_fn(self._placeholder)
         self._scores = tf.nn.softmax(logits)
         tfi.checkpoint.restore(CHECKPOINT_FILE)

       def predict(self, *, images: self._placeholder) -> {
             'scores': self._scores,
             'categories': self._labels,
          }:
          pass

   # Lazily download checkpoint file and verify its digest.
   if not os.path.exists(CHECKPOINT_FILE):
     import hashlib
     import tarfile

     downloaded = urlretrieve(CHECKPOINT_URL)[0]
     def sha256(filename, blocksize=65536):
         hash = hashlib.sha256()
         with open(filename, "rb") as f:
             for block in iter(lambda: f.read(blocksize), b""):
                 hash.update(block)
         return hash.hexdigest()
     s = sha256(downloaded)
     if s != CHECKPOINT_SHA256:
       print("invalid fetch of", CHECKPOINT_URL, s, "!=", CHECKPOINT_SHA256)
       exit(1)
     with tarfile.open(downloaded, 'r|gz') as tar:
       tar.extractall()

   # Do the actual export!
   tfi.saved_model.export("./inception_v1.saved_model", InceptionV1)


Image data
==========
The ``tf.data.file`` function uses `mimetypes <https://docs.python.org/3.6/library/mimetypes.html>`_ to discover the right data decoder to use. If an input to a graph is an ``"image/*"``, TFI will automatically decode and resize the image to the proper size. In the example above, the JPEG image of a dog is automatically decoded and resized to 224x224.

Batches
=======
If you look closely at the example code above, you'll see that the images argument is actually an array. The class generated by TFI is smart enough to convert an array of images to an appropriately sized batch of Tensors.

Graphs with variables
=====================
Each instance of the class has separate variables from other instances. If a graph's variables are mutated during a session in a useful way, you can continue to use those mutations by calling methods again on that same instance.

If you'd like to have multiple instances that do not interfere with one another, you can create a second instance and call methods on each of them separately.

Getting Started
===============
`TFI is on PyPI <https://pypi.python.org/pypi/tfi>`_, install it with ``pip install tfi``.

Future work
===========

Adapting ``tfi.data`` functions to handle queues and datasets wouldn't require much effort. If this is something you'd like me to do, please `file an issue <https://github.com/ajbouh/tfi/issues/new>`_ with your specific use case!

Extending `tfi.data` to support more formats is also quite straightforward. `File an issue <https://github.com/ajbouh/tfi/issues/new>`_ with a specific format you'd like to see. For bonus points, include the expected tensor dtype and shape. For double bonus points, include a way for me to test it in a real model.

Acknowledgements
================
If you're curious, the photo used above was from `a random Google image search <https://goo.gl/images/UNNf2W>`_.

PyPI packaging was way easier because of `this fantastic guide <https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/>`_.
