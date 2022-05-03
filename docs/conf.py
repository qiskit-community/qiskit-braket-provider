# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Sphinx documentation builder
"""

# General options:
import os

project = "Qiskit-Braket provider"
copyright = "2022"  # pylint: disable=redefined-builtin
author = "Qiskit team"

# The full version, including alpha/beta/rc tags
with open(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "qiskit_braket_provider",
        "VERSION.txt",
    ),
    "r",
) as f:
    release = f.read()
# The short X.Y version
version = release

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "nbsphinx",
]
templates_path = ["_templates"]
numfig = True
numfig_format = {"table": "Table %s"}
language = None
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["qiskit_braket_provider."]

# html theme options
html_static_path = ["_static"]
html_logo = "_static/images/logo.png"

# autodoc/autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"

# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
nbsphinx_execute = "never"
nbsphinx_widgets_path = ""
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme_options = {
    "github_user": "qiskit-community",
    "github_repo": "qiskit-braket-provider",
    "github_button": True,
    "github_type": "star",
    "github_count": False,
}
