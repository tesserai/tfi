# -*- coding: utf-8 -*-
"""
    sphinx.ext.napoleon.docstring
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    Classes for docstring parsing and formatting.


    :copyright: Copyright 2007-2017 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import collections
import inspect
import re

# from six import string_types, u
# from six.moves import range

from tfi.doc.iterators import modify_iter

import sys
def _prepare_docstring(s, ignore=1):
    # type: (unicode, int) -> List[unicode]
    """Convert a docstring into lines of parseable reST.  Remove common leading
    indentation, where the indentation of a given number of lines (usually just
    one) is ignored.
    Return the docstring as a list of lines usable for inserting into a docutils
    ViewList (used as argument of nested_parse().)  An empty line is added to
    act as a separator between this docstring and following content.
    """
    lines = s.expandtabs().splitlines()
    # Find minimum indentation of any non-blank lines after ignored lines.
    margin = sys.maxsize
    for line in lines[ignore:]:
        content = len(line.lstrip())
        if content:
            indent = len(line) - content
            margin = min(margin, indent)
    # Remove indentation from ignored lines.
    for i in range(ignore):
        if i < len(lines):
            lines[i] = lines[i].lstrip()
    if margin < sys.maxsize:
        for i in range(ignore, len(lines)):
            lines[i] = lines[i][margin:]
    # Remove any leading blank lines.
    while lines and not lines[0]:
        lines.pop(0)
    # make sure there is an empty line at the end
    if lines and lines[-1]:
        lines.append('')
    return lines

_directive_regex = re.compile(r'\.\. \S+::')
_google_section_regex = re.compile(r'^(\s|\w)+:\s*$')
_google_typed_arg_regex = re.compile(r'\s*(.+?)\s*\(\s*(.*[^\s]+)\s*\)')
_single_colon_regex = re.compile(r'(?<!:):(?!:)')
_xref_regex = re.compile(r'(:(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:`.+?`)')
_bullet_list_regex = re.compile(r'^(\*|\+|\-)(\s+\S|\s*$)')
_enumerated_list_regex = re.compile(
    r'^(?P<paren>\()?'
    r'(\d+|#|[ivxlcdm]+|[IVXLCDM]+|[a-zA-Z])'
    r'(?(paren)\)|\.)(\s+\S|\s*$)')

class GoogleDocstring(object):
    """Convert Google style docstrings to reStructuredText.

    Parameters
    ----------
    docstring : :obj:`str` or :obj:`list` of :obj:`str`
        The docstring to parse, given either as a string or split into
        individual lines.


    Other Parameters
    ----------------
    what : :obj:`str`, optional
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : :obj:`str`, optional
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.


    Example
    -------
    >>> from sphinx.ext.napoleon import Config
    >>> config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
    >>> docstring = '''One line summary.
    ...
    ... Extended description.
    ...
    ... Args:
    ...   arg1(int): Description of `arg1`
    ...   arg2(str): Description of `arg2`
    ... Returns:
    ...   str: Description of return value.
    ... '''
    >>> print(GoogleDocstring(docstring, config))
    One line summary.
    <BLANKLINE>
    Extended description.
    <BLANKLINE>
    :param arg1: Description of `arg1`
    :type arg1: int
    :param arg2: Description of `arg2`
    :type arg2: str
    <BLANKLINE>
    :returns: Description of return value.
    :rtype: str
    <BLANKLINE>

    """
    def __init__(self, docstring=None, what='', name='',
                 obj=None, options=None):

        if not what:
            if inspect.isclass(obj):
                what = 'class'
            elif inspect.ismodule(obj):
                what = 'module'
            elif isinstance(obj, collections.Callable):  # type: ignore
                what = 'function'
            else:
                what = 'object'

        if docstring is None:
            if obj is None:
                raise "If docstring is None, obj may not be"
            docstring = obj.__doc__

        self._what = what
        self._name = name
        self._obj = obj
        if isinstance(docstring, str):
            docstring = _prepare_docstring(docstring)
            print("prepared docstring...", docstring)
        else:
            print("didn't prepare docstring...", docstring)
        self._lines = docstring
        self._line_iter = modify_iter(docstring, modifier=lambda s: s.rstrip())
        self._parsed_lines = []  # type: List[unicode]
        self._is_in_section = False
        self._section_indent = 0
        self._directive_sections = []  # type: List[unicode]
        self._dict_sections = {
            'args': self._parse_fields_section,
            'attributes': self._parse_fields_section,
            'returns': self._parse_fields_section,
            'yields': self._parse_fields_section,
        }  # type: Dict[unicode, Callable]

        self._list_sections = {
            'example': self._parse_generic_section,
            'examples': self._parse_generic_section,
            'note': self._parse_generic_section,
            'references': self._parse_generic_section,
            'see also': self._parse_generic_section,
            'todo': self._parse_generic_section,
        }  # type: Dict[unicode, Callable]

        self._sections = {
            name: value
            for name, value in [*self._dict_sections.items(), *self._list_sections.items()]
        }

        self._parsed_dicts = {
            name: []
            for name in self._dict_sections.keys()
        }
        self._parse()

    def lines(self):
        # type: () -> List[unicode]
        """Return the parsed lines of the docstring in reStructuredText format.

        Returns
        -------
        list(str)
            The lines of the docstring in a list.

        """
        return self._parsed_lines

    def result(self):
        # type: () -> List[unicode]
        """Return the parsed lines of the docstring in reStructuredText format.

        Returns
        -------
        list(str)
            The lines of the docstring in a list.

        """
        return self._parsed_lines, self._parsed_dicts

    def _consume_indented_block(self, indent=1):
        # type: (int) -> List[unicode]
        lines = []
        line = self._line_iter.peek()
        while(not self._is_section_break() and
              (not line or self._is_indented(line, indent))):
            lines.append(next(self._line_iter))  # type: ignore
            line = self._line_iter.peek()
        return lines

    def _consume_contiguous(self):
        # type: () -> List[unicode]
        lines = []
        while (self._line_iter.has_next() and
               self._line_iter.peek() and
               not self._is_section_header()):
            lines.append(next(self._line_iter))  # type: ignore
        return lines

    def _consume_empty(self):
        # type: () -> List[unicode]
        lines = []
        line = self._line_iter.peek()
        while self._line_iter.has_next() and not line:
            lines.append(next(self._line_iter))  # type: ignore
            line = self._line_iter.peek()
        return lines

    def _consume_field(self, parse_type=True, prefer_type=False):
        # type: (bool, bool) -> Tuple[unicode, unicode, List[unicode]]
        line = next(self._line_iter)  # type: ignore

        before, colon, after = self._partition_field_on_colon(line)
        _name, _type, _desc = before, '', after  # type: unicode, unicode, unicode

        if parse_type:
            match = _google_typed_arg_regex.match(before)  # type: ignore
            if match:
                _name = match.group(1)
                _type = match.group(2)

        _name = self._escape_args_and_kwargs(_name)

        if prefer_type and not _type:
            _type, _name = _name, _type
        indent = self._get_indent(line) + 1
        _descs = [_desc] + self._dedent(self._consume_indented_block(indent))
        return _name, _type, _descs

    def _consume_fields(self, parse_type=True, prefer_type=False):
        # type: (bool, bool) -> List[Tuple[unicode, unicode, List[unicode]]]
        self._consume_empty()
        fields = []
        while not self._is_section_break():
            _name, _type, _desc = self._consume_field(parse_type, prefer_type)
            if _name or _type or _desc:
                fields.append((_name, _type, _desc,))
        return fields

    def _consume_section_header(self):
        # type: () -> unicode
        section = next(self._line_iter)  # type: ignore
        stripped_section = section.strip(':')
        if stripped_section.lower() in self._sections:
            section = stripped_section
        return section

    def _consume_to_end(self):
        # type: () -> List[unicode]
        lines = []
        while self._line_iter.has_next():
            lines.append(next(self._line_iter))  # type: ignore
        return lines

    def _consume_to_next_section(self):
        # type: () -> List[unicode]
        self._consume_empty()
        lines = []
        while not self._is_section_break():
            lines.append(next(self._line_iter))  # type: ignore
        return lines + self._consume_empty()

    def _dedent(self, lines, full=False):
        # type: (List[unicode], bool) -> List[unicode]
        if full:
            return [line.lstrip() for line in lines]
        else:
            min_indent = self._get_min_indent(lines)
            return [line[min_indent:] for line in lines]

    def _escape_args_and_kwargs(self, name):
        # type: (unicode) -> unicode
        if name[:2] == '**':
            return r'\*\*' + name[2:]
        elif name[:1] == '*':
            return r'\*' + name[1:]
        else:
            return name

    def _fix_field_desc(self, desc):
        # type: (List[unicode]) -> List[unicode]
        if self._is_list(desc):
            desc = [u''] + desc
        elif desc[0].endswith('::'):
            desc_block = desc[1:]
            indent = self._get_indent(desc[0])
            block_indent = self._get_initial_indent(desc_block)
            if block_indent > indent:
                desc = [u''] + desc
            else:
                desc = ['', desc[0]] + self._indent(desc_block, 4)
        return desc

    def _get_current_indent(self, peek_ahead=0):
        # type: (int) -> int
        line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        while line != self._line_iter.sentinel:
            if line:
                return self._get_indent(line)
            peek_ahead += 1
            line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        return 0

    def _get_indent(self, line):
        # type: (unicode) -> int
        for i, s in enumerate(line):
            if not s.isspace():
                return i
        return len(line)

    def _get_initial_indent(self, lines):
        # type: (List[unicode]) -> int
        for line in lines:
            if line:
                return self._get_indent(line)
        return 0

    def _get_min_indent(self, lines):
        # type: (List[unicode]) -> int
        min_indent = None
        for line in lines:
            if line:
                indent = self._get_indent(line)
                if min_indent is None:
                    min_indent = indent
                elif indent < min_indent:
                    min_indent = indent
        return min_indent or 0

    def _indent(self, lines, n=4):
        # type: (List[unicode], int) -> List[unicode]
        return [(' ' * n) + line for line in lines]

    def _is_indented(self, line, indent=1):
        # type: (unicode, int) -> bool
        for i, s in enumerate(line):
            if i >= indent:
                return True
            elif not s.isspace():
                return False
        return False

    def _is_list(self, lines):
        # type: (List[unicode]) -> bool
        if not lines:
            return False
        if _bullet_list_regex.match(lines[0]):  # type: ignore
            return True
        if _enumerated_list_regex.match(lines[0]):  # type: ignore
            return True
        if len(lines) < 2 or lines[0].endswith('::'):
            return False
        indent = self._get_indent(lines[0])
        next_indent = indent
        for line in lines[1:]:
            if line:
                next_indent = self._get_indent(line)
                break
        return next_indent > indent

    def _is_section_header(self):
        # type: () -> bool
        section = self._line_iter.peek().lower()
        match = _google_section_regex.match(section)
        if match and section.strip(':') in self._sections:
            header_indent = self._get_indent(section)
            section_indent = self._get_current_indent(peek_ahead=1)
            return section_indent > header_indent
        elif self._directive_sections:
            if _directive_regex.match(section):
                for directive_section in self._directive_sections:
                    if section.startswith(directive_section):
                        return True
        return False

    def _is_section_break(self):
        # type: () -> bool
        line = self._line_iter.peek()
        return (not self._line_iter.has_next() or
                self._is_section_header() or
                (self._is_in_section and
                    line and
                    not self._is_indented(line, self._section_indent)))

    def _parse(self):
        # type: () -> None
        self._parsed_lines = self._consume_empty()

        while self._line_iter.has_next():
            if self._is_section_header():
                try:
                    section = self._consume_section_header()
                    self._is_in_section = True
                    self._section_indent = self._get_current_indent()
                    if _directive_regex.match(section):  # type: ignore
                        lines = [section] + self._consume_to_next_section()
                    else:
                        section_key = section.lower()
                        parse_section = self._sections[section_key]
                        if section_key in self._parsed_dicts:
                            self._parsed_dicts[section_key].extend(
                                    parse_section())
                        else:
                            self._parsed_lines.append(
                                    (section_key, parse_section()))
                finally:
                    self._is_in_section = False
                    self._section_indent = 0
            else:
                if not self._parsed_lines:
                    self._parsed_lines.append(('text', self._consume_contiguous() + self._consume_empty()))
                else:
                    self._parsed_lines.append(('text', self._consume_to_next_section()))

        # Multiline docstrings often begin right after the """ and then continue
        # with appropriate indentation at the next line break. The above algorithm
        # splits a single text section into two. Merge them here if that happens.
        if len(self._parsed_lines) >= 2:
            first = self._parsed_lines[0]
            second = self._parsed_lines[1]
            if first[0] == 'text' and second[0] == 'text':
                self._parsed_lines = self._parsed_lines[1:]
                self._parsed_lines[0] = ('text', first[1] + second[1])

    def _parse_fields_section(self):
        # type: (unicode) -> List[unicode]
        fields = self._consume_fields()

        # type: (List[Tuple[unicode, unicode, List[unicode]]], unicode, unicode) -> List[unicode]  # NOQA
        lines = []
        for _name, _type, _desc in fields:
            _desc = self._strip_empty(_desc)
            if any(_desc):
                _desc = self._fix_field_desc(_desc)

            lines.append((_name, _type, _desc))
        return lines

    def _parse_generic_section(self):
        # type: (unicode, bool) -> List[unicode]
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        if lines:
            return lines
        else:
            return ['']


    def _partition_field_on_colon(self, line):
        # type: (unicode) -> Tuple[unicode, unicode, unicode]
        before_colon = []
        after_colon = []
        colon = ''
        found_colon = False
        for i, source in enumerate(_xref_regex.split(line)):  # type: ignore
            if found_colon:
                after_colon.append(source)
            else:
                m = _single_colon_regex.search(source)
                if (i % 2) == 0 and m:
                    found_colon = True
                    colon = source[m.start(): m.end()]
                    before_colon.append(source[:m.start()])
                    after_colon.append(source[m.end():])
                else:
                    before_colon.append(source)

        return ("".join(before_colon).strip(),
                colon,
                "".join(after_colon).strip())

    def _strip_empty(self, lines):
        # type: (List[unicode]) -> List[unicode]
        if lines:
            start = -1
            for i, line in enumerate(lines):
                if line:
                    start = i
                    break
            if start == -1:
                lines = []
            end = -1
            for i in reversed(range(len(lines))):
                line = lines[i]
                if line:
                    end = i
                    break
            if start > 0 or end + 1 < len(lines):
                lines = lines[start:end + 1]
        return lines
