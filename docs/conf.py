# conf.py
# Sphinx configuration file
# https://www.sphinx-doc.org/en/master/usage/configuration.html

### import setup ##################################################################################

import datetime

### project information ###########################################################################

project = "maxent_disaggregation"
author = "Arthur Jakobs & Simon Schulte"
copyright = datetime.date.today().strftime("%Y") + ' Arthur Jakobs & Simon Schulte'

### project configuration #########################################################################

extensions = [
    # native extensions
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx.ext.inheritance_diagram',
    # iPython extensions
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    # theme
    'sphinx_rtd_theme',
    # Markdown support
    # 'myst_parser', # do not enable separately if using myst_nb, compare: https://github.com/executablebooks/MyST-NB/issues/421#issuecomment-1164427544
    # Jupyter Notebook support
    "myst_nb",
    # mermaid support
    "sphinxcontrib.mermaid",
    # API documentation support
    'autoapi',
    # responsive web component support
    'sphinx_design',
    # copy button on code blocks
    "sphinx_copybutton",
]

napoleon_numpy_docstring = True

exclude_patterns = ['_build']

# The master toctree document.
master_doc = 'index'

### intersphinx configuration ######################################################################

intersphinx_mapping = {
    "bw": ("https://docs.brightway.dev/en/latest/", None),
}    

### theme configuration ############################################################################

html_theme = "sphinx_rtd_theme"
html_title = "maxent_disaggregation"
html_show_sphinx = False

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_logo = 'https://raw.githubusercontent.com/brightway-lca/brightway-documentation/main/source/_static/logo/BW_all_white_transparent_landscape_wide.svg'
html_favicon = 'https://github.com/brightway-lca/brightway-documentation/blob/main/source/_static/logo/BW_favicon_500x500.png'

### extension configuration ########################################################################

## myst_parser configuration ############################################
## https://myst-parser.readthedocs.io/en/latest/configuration.html

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

## autoapi configuration ################################################
## https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#customisation-options

autoapi_options = [
    'members',
    'undoc-members',
    'private-members',
    'show-inheritance',
    'show-module-summary',
]

autoapi_python_class_content = 'both'
autoapi_member_order = 'bysource'
autoapi_root = 'content/api'
autoapi_keep_files = False

autoapi_dirs = [
    '../maxent_disaggregation',
]

autoapi_ignore = [
    '*/data/*',
    '*tests/*',
    '*tests.py',
    '*validation.py',
    '*version.py',
    '*.rst',
    '*.yml',
    '*.md',
    '*.json',
    '*.data'
]