import codecs
import os
import re

from setuptools import setup, find_packages

# Make any resulting zip files reproducible.
os.environ["SOURCE_DATE_EPOCH"] = '315532800'

###################################################################

NAME = "tfi"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", NAME, "__init__.py")
KEYWORDS = ["tensorflow", "savedmodel", "boilerplate", "pytorch"]
# List of all classifiers: https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
INSTALL_REQUIRES = [
]
DEPENDENCY_LINKS = [
    "git+https://github.com/facebook/prophet.git@master#egg=fbprophet&subdirectory=python",
]
EXTRAS_REQUIRE = {
    "cli": [
        "ptpython==0.41",
        "prompt-toolkit<2.0.0",
        "pywatchman",
    ],
    "msp": [
        "fbprophet",
        "gpyopt",
        "numpy",
        "pandas",
        "pystan",  
    ],
    "tensorflow": [
        "numpy",
        "tensorflow",
    ],
    "torch": [
        "cloudpickle",
        "torch",
        "torchvision",
    ],
    "spacy": [
        "spacy",
    ],
    "prophet": [
        "fbprophet",
        "pandas",
        "numpy",
        "pyarrow",
        "pystan",
    ],
    "facets": [
        "facets_overview",
    ],
    "serve": [
        "pysquashfsimage",
        "beautifulsoup4",
        "bjoern>=2.2.3",
        "docutils",
        "Flask",
        "requests",
        "tinydb",
        "yapf",
        "opentracing<2,>=1.2.2",
        "Flask-Opentracing",
        "jaeger-client",
        "pillow",
    ],
}
ENTRY_POINTS = {
    'console_scripts': [
        'tfi = tfi.main:main',
    ],
}

###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        description=find_meta("description"),
        license=find_meta("license"),
        url=find_meta("uri"),
        version=find_meta("version"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        keywords=KEYWORDS,
        long_description=read("README.rst"),
        entry_points=ENTRY_POINTS,
        dependency_links=DEPENDENCY_LINKS,
        packages=PACKAGES,
        package_dir={"": "src"},
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        include_package_data=True,
    )
