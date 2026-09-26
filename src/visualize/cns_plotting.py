"""Shared Cell/Nature/Science plotting setup for visualization scripts.

The project follows the publication-plotting defaults provided by cnsplots:
https://github.com/faridrashidi/cnsplots

Call :func:`setup_cnsplots_style` before creating Matplotlib figures. Individual
plots may still override sizes, colors, or other settings when their scientific
content requires it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cnsplots as cns
import matplotlib as mpl


CNSPLOTS_GUIDELINES_URL = "https://github.com/faridrashidi/cnsplots"


def setup_cnsplots_style(
    overrides: Mapping[str, Any] | None = None,
) -> None:
    """Apply cnsplots' publication defaults plus optional local overrides."""

    cns.setup_matplotlib()
    if overrides:
        mpl.rcParams.update(overrides)


def main() -> None:
    """Apply the style and report its key publication export settings."""

    setup_cnsplots_style()
    print(f"cnsplots {cns.__version__} plotting style is ready.")
    print(
        "SVG text: "
        f"{mpl.rcParams['svg.fonttype']}; "
        f"PDF fonts: type {mpl.rcParams['pdf.fonttype']}; "
        f"save DPI: {mpl.rcParams['savefig.dpi']:g}."
    )


if __name__ == "__main__":
    main()
