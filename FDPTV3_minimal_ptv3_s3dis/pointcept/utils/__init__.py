"""Utility package exports used by the minimal project.

The upstream launcher imports ``comm`` from ``pointcept.utils`` directly, so we
re-export it here to keep that import contract intact.
"""

from . import comm

__all__ = ["comm"]
