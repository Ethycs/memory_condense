"""Bundle a locally built Pixi artifact with its installer and instructions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('dist'))
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    output=args.output_dir.resolve()
    packages=list(output.glob('memory_condense-0.2.0-*.conda'))
    if len(packages)!=1:
        raise ValueError('Run pixi build --output-dir dist first; expected one 0.2.0 package')
    package=packages[0]
    installer=output/'install-proxy.ps1'
    shutil.copyfile(repo/'packaging/install-proxy.ps1',installer)
    guide=repo/'docs/02 - Implementation/06 - Provider Proxy and Transcript Import.md'
    files={package.name:package,installer.name:installer,'README.md':repo/'README.md',
           'docs/02 - Implementation/06 - Provider Proxy and Transcript Import.md':guide}
    manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    (output/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    archive=output/'memory-condense-0.2.0-win64.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name,path in files.items(): bundle.write(path,name)
        bundle.write(output/'SHA256SUMS.json','SHA256SUMS.json')
    print(json.dumps(dict(package=str(package),bundle=str(archive),bytes=archive.stat().st_size,
                         sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))


if __name__=='__main__': main()
