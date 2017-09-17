import collections
import importlib.util
import os
import pprint
import unittest

from tfi.doc.docstring import GoogleDocstring

def _load_fixture(basename):
    module_name = ".".join([*__name__.split(".")[:-1], os.path.splitext(basename)[0]])
    module_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            basename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    return m

class GoogleDocstringTest(unittest.TestCase):
    pass

f = _load_fixture("example_google_docstrings_fixture.py")
_FIXTURES = [
    ('toplevel', f,
        ([
            ('text', [
                'Example Google style docstrings.',
                '',
                'This module demonstrates documentation as specified by the `Google Python',
                'Style Guide`_. Docstrings may extend over multiple lines. Sections are '
                'created',
                'with a section header and a colon followed by a block of indented text.',
                '',
            ]),
            ('example', [
                'Examples can be given using either the ``Example`` or ``Examples``',
                'sections. Sections support any reStructuredText formatting, including',
                'literal blocks::',
                '',
                '    $ python example_google.py',
            ]),
            ('text', [
                'Section breaks are created by resuming unindented text. Section breaks',
                'are also implicitly created anytime a new section starts.',
                '',
            ]),
            ('todo', [
                '* For module TODOs',
                '* You have to also use ``sphinx.ext.todo`` extension',
            ]),
            ('text', [
                '.. _Google Python Style Guide:',
                '   http://google.github.io/styleguide/pyguide.html',
                ''
            ])
        ], {
            "args": [],
            "attributes": [
                ("module_level_variable1",
                 'int',
                 ['Module level variables may be documented in',
                  'either the ``Attributes`` section of the module docstring, '
                  'or in an',
                  'inline docstring immediately following the variable.',
                  '',
                  'Either form is acceptable, but the two should not be '
                  'mixed. Choose',
                  'one convention to document module level variables and be '
                  'consistent',
                  'with it.'])
            ],
            "returns": [],
            "yields": [],
         })),
    ('toplevel_func', f.function_with_pep484_type_annotations,
        ([
            ('text', [
                'Example function with PEP 484 type annotations.',
                '',
            ]),
        ], {
            "args": [
                 ('param1', '', ['The first parameter.']),
                 ('param2', '', ['The second parameter.']),
            ],
            "attributes": [],
            "returns": [
                ('result', '', ['The return value. True for success, False otherwise.'])
            ],
            "yields": [],
         })),
    ('exampleclass_init', f.ExampleClass.__init__,
        ([
            ('text', [
                'Example of docstring on the __init__ method.',
                '',
                'The __init__ method may be documented in either the class level',
                'docstring, or as a docstring on the __init__ method itself.',
                '',
                'Either form is acceptable, but the two should not be mixed. Choose one',
                'convention to document the __init__ method and be consistent with it.',
                '',
            ]),
            ('note', [
                'Do not include the `self` parameter in the ``Args`` section.',
            ]),
         ], {
            "args": [
                ('param1', 'str', ['Description of `param1`.']),
                ('param2', ':obj:`int`, optional', ['Description of `param2`. Multiple', 'lines are supported.']),
                ('param3', ':obj:`list` of :obj:`str`', ['Description of `param3`.']),
            ],
            "attributes": [],
            "returns": [],
            "yields": [],
         })),
]

from functools import partialmethod

pp = pprint.PrettyPrinter(indent=1)
for name, obj, expect in _FIXTURES:
    def do_test(self, obj, expect):
        result = GoogleDocstring(obj=obj).result()
        if result != expect:
            pp.pprint(result)
        self.assertEqual(expect, result)

    setattr(GoogleDocstringTest,
            'test_%s' % name,
            partialmethod(do_test, obj=obj, expect=expect))

if __name__ == '__main__':
    unittest.main()
