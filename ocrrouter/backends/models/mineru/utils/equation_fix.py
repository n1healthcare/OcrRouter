"""MinerU equation post-processing utilities.

This module provides equation-specific fixes for LaTeX content extracted from documents.
Re-exports from the existing post_process module for backward compatibility.
"""

# Re-export from the existing post_process module
from ocrrouter.backends.models.mineru.post_process import (
    try_fix_unbalanced_braces,
    try_fix_equation_double_subscript,
    try_fix_equation_eqqcolon,
    try_fix_equation_big,
    try_fix_equation_leq,
    try_match_equation_left_right,
    convert_otsl_to_html,
    do_handle_equation_block,
)


def apply_equation_fixes(content: str, debug: bool = False) -> str:
    """Apply all equation fixes to LaTeX content.

    This is a convenience function that applies all equation fixes in order.

    Args:
        content: LaTeX string to fix.
        debug: Enable debug output.

    Returns:
        Fixed LaTeX string.
    """
    content = try_match_equation_left_right(content, debug=debug)
    content = try_fix_equation_double_subscript(content, debug=debug)
    content = try_fix_equation_eqqcolon(content, debug=debug)
    content = try_fix_equation_big(content, debug=debug)
    content = try_fix_equation_leq(content, debug=debug)
    content = try_fix_unbalanced_braces(content, debug=debug)
    return content


__all__ = [
    "apply_equation_fixes",
    "try_fix_unbalanced_braces",
    "try_fix_equation_double_subscript",
    "try_fix_equation_eqqcolon",
    "try_fix_equation_big",
    "try_fix_equation_leq",
    "try_match_equation_left_right",
    "convert_otsl_to_html",
    "do_handle_equation_block",
]
