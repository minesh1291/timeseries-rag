import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))

project = 'Time Series RAG'
copyright = '2024, OpenHands'
author = 'OpenHands'

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