"""Collect the files to hand in under submit/<ID>_<name>/ and zip them.

Run after the release build, the audit and the English PDFs:
    python scripts/package_submission.py
The folder holds the Parts 1-4 PDF, the Part 5 PDF, the executable, the
source tree, a short RUN.txt and results/: every viewable output of the audited
run (bitmaps, SVGs, JSON summaries; the raw CSVs are left out for size) and the
high-fidelity exports of both paintings, so the results can be seen without
running anything. The ZIP next to it is what gets uploaded.

The high-fidelity exports come from
    multimedia hifi assets/paintings/<name>.bmp --out output/hifi_export
    multimedia render output/hifi_export/export_<name>_refined_coefficients.csv --svg --out output/hifi_export
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Personal details live in personal.json (git-ignored); see personal.example.json.
_personal = ROOT / 'personal.json'
if not _personal.is_file():
    raise SystemExit('copy personal.example.json to personal.json and fill in your details')
ME = json.loads(_personal.read_text(encoding='utf-8'))
ID = ME['student_id']
FOLDER = f"{ID}_{ME['name']}"
destination = ROOT / 'submit' / FOLDER
# Emptied rather than removed, so a file browser showing the folder does not block it.
destination.mkdir(parents=True, exist_ok=True)
for old in destination.iterdir():
    shutil.rmtree(old) if old.is_dir() else old.unlink()

files = {
    ROOT / 'output' / 'pdf' / 'assignment_1151_parts_1_4_en.pdf': f'{ID}_Parts1-4_Report.pdf',
    ROOT / 'output' / 'pdf' / 'assignment_1151_part5_en.pdf': f'{ID}_Part5_FinalProject.pdf',
    ROOT / 'target' / 'release' / 'multimedia.exe': 'multimedia.exe',
}
for source, name in files.items():
    if not source.is_file():
        raise SystemExit(f'missing {source}; build, audit and render the English PDFs first')
    shutil.copy2(source, destination / name)

source_dir = destination / 'source'
source_dir.mkdir()
for name in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'README.md'):
    shutil.copy2(ROOT / name, source_dir / name)
for name in ('src', 'tests', 'scripts'):
    shutil.copytree(ROOT / name, source_dir / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
# The two Part 4 paintings: the JPEGs only; decode_paintings.py makes the BMPs.
(source_dir / 'assets' / 'paintings').mkdir(parents=True)
for jpeg in sorted((ROOT / 'assets' / 'paintings').glob('*.jpg')):
    shutil.copy2(jpeg, source_dir / 'assets' / 'paintings' / jpeg.name)

# Results: the viewable files of the audited run, each checked against the
# audit's SHA-256 manifest so that nothing stale is shipped.
OUTPUT = ROOT / 'output'
manifest = json.loads((OUTPUT / 'evidence_manifest.json').read_text(encoding='utf-8'))
if manifest.get('status') != 'passed':
    raise SystemExit('the audit has not passed; run scripts/audit.py output first')
VIEWABLE = {'.bmp', '.svg', '.json', '.md'}
results = destination / 'results'
results.mkdir()
shipped = 0
for name, digest in sorted(manifest['artifact_sha256'].items()):
    path = OUTPUT / name
    if path.suffix not in VIEWABLE:
        continue
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f'{name} changed since the audit; rerun the program and the audit')
    shutil.copy2(path, results / name)
    shipped += 1
for name in ('evidence_manifest.json', 'verification.md'):
    shutil.copy2(OUTPUT / name, results / name)
exports = results / 'hifi_export'
exports.mkdir()
for painting in sorted(p.stem for p in (ROOT / 'assets' / 'paintings').glob('*.jpg')):
    for suffix in ('.svg', '.json'):
        source = OUTPUT / 'hifi_export' / f'export_{painting}_refined{suffix}'
        if not source.is_file():
            raise SystemExit(f'missing {source}; run hifi and render for {painting} (see this script\'s docstring)')
        shutil.copy2(source, exports / source.name)
(results / 'README.txt').write_text(f'''Results of the run the report is built from ({shipped} files from output/).

Bitmaps (.bmp), vector files (.svg) and summaries (.json) as the program wrote
them; the report cites every number from these summaries. The raw CSV tables
(coverage, labels, coordinates; about 800 MB) are left out for size: rerun the
program (RUN.txt) to get them and to rerun the audit.
verification.md lists the audit's checks; evidence_manifest.json holds the
SHA-256 of every output of that run, including the files here.

Where to look:
  star*, ring*, curve*, joins*, wheel*      Parts 1-3
  vectorize_*, recover_*                    Part 4: the star, three shapes
  holes_*, color_*                          Part 4 extended: holes, color
  paint_<painting>_*                        Part 4 extended: paintings
                                            (*_vector.bmp is each method's refill,
                                             *_browser.svg opens in a browser)
  hifi_<painting>_*                         Part 4 extended: high fidelity
  hifi_export/export_<painting>_refined.svg each painting drawn from Fourier
                                            series (smooth outlines, DFT + descent),
                                            viewable in a browser; .json gives its
                                            mean dE and SSIM
  transform_*                               the rotation experiment (not in the report)
''', encoding='utf-8')

(destination / 'RUN.txt').write_text(f'''Assignment 1151 - {ID} {ME['name']}

Environment: Windows 11 x86_64, Rust 1.92.0 (stable, MSVC toolchain).
No external Rust crates and no environment variables are needed.

One-line command (from the source folder):
    cargo run --release -- all --out output --size 384 --wedges 96 --paintings assets/paintings

The Part 4 painting showcase reads assets/paintings/*.bmp; create them once by
decoding the included JPEGs (Pillow decodes only, no resizing):
    python scripts/decode_paintings.py
The high-fidelity experiment runs separately (about 15 minutes on 20 threads):
    cargo run --release -- hifi --out output --paintings assets/paintings
Other entry points (one part, one painting, exporting a picture as Fourier
coefficients and a browser SVG) are listed by: multimedia.exe --help
Without --paintings every other experiment still runs. The Lab results also come
as output/*_lab*_browser.svg, which open directly in a web browser.

Prebuilt executable (same program, same arguments):
    multimedia.exe all --out output --size 384 --wedges 96 --paintings source/assets/paintings

All figures and numbers in the report are written to output/; the viewable ones
(and both paintings drawn from Fourier series) are already in results/.
Optional independent audit and PDF layout (Python 3.12, Pillow 12.3, ReportLab 4.4):
    python scripts/audit.py output
    python scripts/build_report.py output --lang en
    python scripts/build_part5.py output --lang en
''', encoding='utf-8')

archive = ROOT / 'submit' / f'{FOLDER}.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(destination.rglob('*')):
        zf.write(path, path.relative_to(destination.parent))
print(destination)
print(archive)
