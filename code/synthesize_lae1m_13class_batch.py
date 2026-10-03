#!/usr/bin/env python3
"""Generate one 10k LAE-1M-subset batch with the frozen 13-class taxonomy."""
from __future__ import annotations

import argparse
import synthesize_lae1m_v9 as pipeline
import synthesize_copypaste_v6 as paste

CLASSES_13=['C-17','C-130','KC-135','KC-10','A-10','F-15','F-16','F/A-18','F-35','B-1B','B-52H','E-2','E-3']
MTARSI_CLASSES=['C-17','C-130','A-10','F-15','F-16','F/A-18','F-35','B-1B','B-52H','E-2','E-3']
MAR20_CLASSES=['C-17','C-130','KC-135','KC-10','F-15','F-16','F/A-18','B-1B','B-52H','E-3']

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',choices=['mtarsi','mar20'],required=True)
    ap.add_argument('--dataset',choices=sorted(pipeline.SPECS),required=True)
    ap.add_argument('--count',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=20260928)
    ap.add_argument('--foreground-manifest',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    classes=MTARSI_CLASSES if args.source=='mtarsi' else MAR20_CLASSES
    mapping={name:CLASSES_13.index(name) for name in classes}
    pipeline.CLASSES=classes
    pipeline.CLASS_ID=mapping
    pipeline.MAR20_NAMES=CLASSES_13
    pipeline.PIPELINE_VERSION=f'copy-paste-{args.source}-13class-lae-v1'
    paste.CLASS_ID=mapping
    paste.CLASS_CYCLE=classes
    paste.MAR20_NAMES=CLASSES_13
    import sys
    sys.argv=[sys.argv[0],'--dataset',args.dataset,'--count',str(args.count),'--seed',str(args.seed),
              '--foreground-manifest',args.foreground_manifest,'--foreground-source',args.source,
              '--output',args.output]
    pipeline.main()

if __name__=='__main__':main()
