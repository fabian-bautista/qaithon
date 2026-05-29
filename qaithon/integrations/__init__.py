"""Adapters bridging Qaithon-compiled models to popular serving frameworks.

Each integration is opt-in (controlled by a pip extra) and isolated so
that breakage in one framework's API does not cascade into the rest of
the library.
"""

from __future__ import annotations
