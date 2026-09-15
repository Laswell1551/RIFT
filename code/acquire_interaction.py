"""Acquire only the sixteen trusted source arrays required by the road protocol."""
from pathlib import Path
import argparse
import hashlib
import json
import gdown
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-data-terms',action='store_true',required=True,
                        help='Confirm permission to use the original INTERACTION/H2C processed data.')
    parser.parse_args()
    rows=json.loads((ROOT/'provenance/interaction_downloads.json').read_text())
    hashes=pd.read_csv(ROOT/'provenance/interaction_source_sha256.csv')
    expected={Path(r.file.replace('\\','/')).name:r.sha256 for r in hashes.itertuples()}
    for row in rows:
        dst=ROOT/'data/interaction'/row['path'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            gdown.download(id=row['id'],output=str(dst),quiet=False,resume=True,use_cookies=False)
        if not dst.is_file():raise RuntimeError(f'Download failed: {dst.name}')
        with dst.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=expected[dst.name]:raise RuntimeError(f'Checksum mismatch: {dst.name}; file was not loaded')
        print('Verified',dst.name)

if __name__=='__main__':main()
