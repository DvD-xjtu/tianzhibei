#!/usr/bin/env python3
"""Apply engineering-v7 quality gate to a 13-class LAE batch."""
from __future__ import annotations
import filter_lae1m_v9 as gate
from verify_lae1m_13class_batch import audit
gate.audit=audit
if __name__=='__main__': gate.main()
