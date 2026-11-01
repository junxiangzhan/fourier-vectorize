"""Decode the Part 4 painting JPEGs into 24-bit BMP for the Rust program.

Run: python scripts/decode_paintings.py [folder]   (default assets/paintings)
Pillow only decodes the files here: no resizing, filtering or colour
change. Cropping, downsampling and every later step happen in Rust.
"""
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
folder = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'assets' / 'paintings')
found = sorted(folder.glob('*.jpg'))
if not found:
    raise SystemExit(f'no .jpg files in {folder}')
for source in found:
    with Image.open(source) as image:
        image.convert('RGB').save(source.with_suffix('.bmp'), format='BMP')
    print(source.with_suffix('.bmp'))
