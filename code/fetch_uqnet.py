"""Fetch pinned external encoder files; no third-party source is redistributed."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accept-upstream-terms',action='store_true',required=True,
                   help='Confirm that your use is permitted under the upstream source terms.')
    p.parse_args()
    manifest=json.loads((ROOT/'provenance/model_source_manifest.json').read_text())
    out=ROOT/'code/uqnet_vendor';out.mkdir(exist_ok=True)
    for row in manifest['files']:
        url=f"https://raw.githubusercontent.com/BIT-Jack/H2C-lifelong/{manifest['upstream_commit']}/traj_predictor/{row['file']}"
        raw=urllib.request.urlopen(url,timeout=60).read().replace(b'\r\n',b'\n')
        assert hashlib.sha256(raw).hexdigest()==row['upstream_lf_sha256'], 'Upstream source checksum mismatch'
        if row['file']=='encoder.py':
            raw=raw.replace(b'from traj_predictor.baselayers import *',b'from .baselayers import *')
        else:
            raw=raw.replace(b'from traj_predictor.utils import *',b'')
        assert hashlib.sha256(raw).hexdigest()==row['local_lf_sha256'], 'Import adaptation checksum mismatch'
        (out/row['file']).write_bytes(raw)
        print('Verified and installed',row['file'])
    (out/'__init__.py').write_text('"""Locally acquired external encoder; see THIRD_PARTY.md."""\n')

if __name__=='__main__':main()
