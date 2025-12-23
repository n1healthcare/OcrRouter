"""MinerU post-processing submodules for equation and table fixes.

This module exports individual fix functions that are used by MinerUPostprocessor.
The main post-processing logic is in postprocessor.py.
"""

from .equation_unbalanced_braces import try_fix_unbalanced_braces
from .equation_block import do_handle_equation_block
from .equation_double_subscript import try_fix_equation_double_subscript
from .equation_fix_eqqcolon import try_fix_equation_eqqcolon
from .equation_big import try_fix_equation_big
from .equation_leq import try_fix_equation_leq
from .equation_left_right import try_match_equation_left_right
from .otsl2html import convert_otsl_to_html

__all__ = [
    "try_fix_unbalanced_braces",
    "do_handle_equation_block",
    "try_fix_equation_double_subscript",
    "try_fix_equation_eqqcolon",
    "try_fix_equation_big",
    "try_fix_equation_leq",
    "try_match_equation_left_right",
    "convert_otsl_to_html",
]
