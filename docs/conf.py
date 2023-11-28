import os
import shutil
import sys

__location__ = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(__location__, "../src"))
sys.path.insert(0, os.path.join(__location__, "../src/reglm"))

try:  # for Sphinx >= 1.7
    from sphinx.ext import apidoc
except ImportError:
    from sphinx import apidoc

output_dir = os.path.join(__location__, "api")
module_dir = os.path.join(__location__, "../src/reglm")
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass

try:
    import sphinx

    cmd_line = f"sphinx-apidoc --implicit-namespaces -f -o {output_dir} {module_dir}"

    args = cmd_line.split(" ")
    if tuple(sphinx.__version__.split(".")) >= ("1", "7"):
        # This is a rudimentary parse_version to avoid external dependencies
        args = args[1:]

    apidoc.main(args)
except Exception as e:
    print("Running `sphinx-apidoc` failed!\n{}".format(e))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Enable markdown
extensions.append("myst_parser")

# Configure MyST-Parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "regLM"
copyright = "2023, Genentech, Inc."

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# version: The short X.Y version.
# release: The full version, including alpha/beta/rc tags.
# If you don’t need the separation provided between version and release,
# just set them both to the same value.
try:
    from reglm import __version__ as version
except ImportError:
    version = ""

if not version or version.lower() == "unknown":
    version = os.getenv("READTHEDOCS_VERSION", "unknown")  # automatically set by RTD

release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_emit_warnings = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "alabaster"
html_theme_options = {"sidebar_width": "300px", "page_width": "1200px"}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "reglm-doc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ("letterpaper" or "a4paper").
    # "papersize": "letterpaper",
    # The font size ("10pt", "11pt" or "12pt").
    # "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    # "preamble": "",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", "user_guide.tex", "reglm Documentation", "lala8", "manual")
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = ""

# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": ("https://docs.python.org/" + python_version, None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "setuptools": ("https://setuptools.pypa.io/en/stable/", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
}

print(f"loading configurations for {project} {version} ...", file=sys.stderr)
