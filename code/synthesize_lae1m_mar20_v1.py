#!/usr/bin/env python3
"""LAE-1M preview synthesis using augmented, accepted MAR20 foregrounds."""
from __future__ import annotations

import synthesize_lae1m_v9 as pipeline
import synthesize_copypaste_v6 as paste

CLASS_NAMES = [
    'A1 SU-35', 'A2 C-130', 'A3 C-17', 'A4 C-5', 'A5 F-16',
    'A6 TU-160', 'A7 E-3', 'A8 B-52', 'A9 P-3C', 'A10 B-1B',
    'A11 E-8', 'A12 TU-22', 'A13 F-15', 'A14 KC-135', 'A15 F-22',
    'A16 F/A-18', 'A17 TU-95', 'A18 KC-10', 'A19 SU-34', 'A20 SU-24',
]
CLASS_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def main() -> None:
    pipeline.CLASSES = CLASS_NAMES
    pipeline.CLASS_ID = CLASS_ID
    pipeline.MAR20_NAMES = CLASS_NAMES
    pipeline.PIPELINE_VERSION = 'copy-paste-mar-20-v1-lae-preview-1'
    paste.CLASS_ID = CLASS_ID
    paste.CLASS_CYCLE = CLASS_NAMES
    paste.MAR20_NAMES = CLASS_NAMES
    pipeline.main()


if __name__ == '__main__':
    main()
