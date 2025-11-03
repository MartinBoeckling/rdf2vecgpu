# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Make both the repo root and src discoverable
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_DIR)

project = "rdf2vecgpu"
copyright = "2025, Martin Boeckling"
author = "Martin Boeckling"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",  # cross-project links
    "sphinx.ext.autosectionlabel",
    "sphinxemoji.sphinxemoji",
]
extensions += ["sphinx.ext.autosummary"]
autosummary_generate = True
# Make index.rst the landing page (avoid README as root)
root_doc = "index"
# Prefix section labels with the document path to avoid duplicates
autosectionlabel_prefix_document = True


# Sensible defaults for autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}
add_module_names = False
autodoc_typehints = "description"

# Mock heavy/optional GPU deps so autodoc doesn’t fail on missing libs
autodoc_mock_imports = [
    "cudf",
    "cupy",
    "cugraph",
    "dask_cudf",
    "dask_cuda",
    "dask",
    "torch",
    "lightning",
    "pytorch_lightning",
    "igraph",
    "pyspark",
    "sparkkgml",
]

# Optionally, pick up package version automatically
try:
    from rdf2vecgpu import __version__ as release
except Exception:
    try:
        from src import __version__ as release  # src/__init__.py exports __version__
    except Exception:
        release = "0.0.0"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
