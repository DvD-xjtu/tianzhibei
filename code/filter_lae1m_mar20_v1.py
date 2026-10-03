#!/usr/bin/env python3
"""Run the frozen engineering-v7 gate on one MAR20→LAE preview batch."""
from __future__ import annotations

import filter_lae1m_v9 as gate
from verify_lae1m_mar20_v1 import audit

gate.audit = audit

if __name__ == '__main__':
    gate.main()
