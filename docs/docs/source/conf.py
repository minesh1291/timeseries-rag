import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))

project = 'Time Series RAG'
copyright = '2024, Minesh A. Jethva'
author = 'Minesh A. Jethva'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
    'nbsphinx',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'nbsphinx',
}