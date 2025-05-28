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

needs_sphinx = '7.3.0'


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
    # 'sphinx_rtd_theme',
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

## autoapi configuration ################################################
## https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#customisation-options

autoapi_type = 'python'
autoapi_dirs = ['../maxent_disaggregation']
autoapi_options = [
    'members',
    'undoc-members',
    'private-members',
    'show-inheritance',
    'show-module-summary',
    "show-inheritance-diagram",
]

autoapi_python_class_content = 'both'
autoapi_member_order = 'bysource'
autoapi_root = 'content/api'
autoapi_template_dir = "_templates/autoapi_templates/"
autoapi_keep_files = False

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

graphviz_output_format = "svg"  # https://pydata-sphinx-theme.readthedocs.io/en/stable/examples/graphviz.html#inheritance-diagram


# Inject custom JavaScript to handle theme switching
mermaid_init_js = """
    function initializeMermaidBasedOnTheme() {
        const theme = document.documentElement.dataset.theme;

        if (theme === 'dark') {
            mermaid.initialize({
                startOnLoad: true,
                theme: 'base',
                themeVariables: {
                    edgeLabelBackground: '#14181E',
                    defaultLinkColor: '#ced6dd',
                    titleColor: '#ced6dd',
                    nodeTextColor: '#ced6dd',
                    lineColor: '#ced6dd',
                }
            });
        } else {
            mermaid.initialize({
                startOnLoad: true,
                theme: 'base',
                themeVariables: {
                    edgeLabelBackground: '#FFFFFF',
                    defaultLinkColor: '#222832',
                    titleColor: '#222832',
                    nodeTextColor: '#222832',
                    lineColor: '#222832',
                }
            });
        }

        // Re-render all Mermaid diagrams
        mermaid.contentLoaded();
    }

    // Observer to detect changes to the data-theme attribute
    const themeObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
            initializeMermaidBasedOnTheme();
            }
        });
    });

    themeObserver.observe(document.documentElement, { attributes: true });

    initializeMermaidBasedOnTheme();
"""


# napoleon_numpy_docstring = True


# The master toctree document.

master_doc = "index"

root_doc = "index"
html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"

suppress_warnings = [
    "myst.header"  # suppress warnings of the kind "WARNING: Non-consecutive header level increase; H1 to H3"
]


### theme configuration ############################################################################

html_title = "maxent_disaggregation"
html_show_sphinx = False
html_show_copyright = True

html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",  # for https://fontawesome.com/ icons
]


html_sidebars = {
    "**": [
        "sidebar-nav-bs.html",
    ],
    "content/index": [],
    "content/installation": [],
    "content/theory": [],
    "content/contributing": [],
    "content/codeofconduct": [],
    "content/license": [],
    "content/changelog": [],
}

html_theme_options = {
    # page elements
    "announcement": "⚠️ BETA VERSION ⚠️ This is a beta version of the maxent_disaggregation package. It is not yet fully tested and may contain bugs. Use at your own risk.",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links.html"],
    "navbar_align": "content",
    # "navbar_persistent": ["theme-switcher"], # this is where the search button is usually placed
    "footer_start": ["copyright"],
    "footer_end": ["footer"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink", "support"],
    "header_links_before_dropdown": 5,
    # page elements content
    "icon_links": [
        {
            "name": "Open this Repo on GitHub",
            "url": "https://github.com/jakobsarthur/maxent_disaggregation",
            "icon": "fab fa-brands fa-github",
        },
    ],
    # various settings
    "collapse_navigation": True,
    # "show_prev_next": False,
    "use_edit_page_button": True,
    "navigation_with_keys": True,
}

# required by html_theme_options: "use_edit_page_button"
html_context = {
    "github_user": "jakobsarthur",
    "github_repo": "maxent_disaggregation",
    "github_version": "main",
    "doc_path": "docs",
}


### extension configuration ########################################################################

## myst_parser configuration ############################################
## https://myst-parser.readthedocs.io/en/latest/configuration.html
source_suffix = {".rst": "restructuredtext", 
                 ".md": "myst-nb", 
                 ".ipynb": "myst-nb"}


myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# myst-nb configuration ################################################
# https://myst-nb.readthedocs.io/en/latest/configuration.html

nb_execution_mode = "off"

