#!/usr/bin/env python3
"""Alias controller for the stationary egg-pour flow.

This file intentionally reuses ``egg_pour_controller`` so both entry points run
the same stationary behavior:
  - shared pan+bowl detection pose
  - precise-mode pour
  - pour-up (+Z) finish
  - extra post-grasp lift
  - place bowl back at pick pose
"""

from __future__ import annotations

from egg_pour_controller import main


if __name__ == "__main__":
    raise SystemExit(main())
