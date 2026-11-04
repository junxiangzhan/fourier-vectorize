"""Decode JPEG or PNG files into the 24-bit BMP that `fourier-vectorize fit` reads.

Run: python scripts/decode_images.py IMAGE... (each BMP lands next to its source)
Pillow only decodes the files here: no resizing, filtering or colour change.
"""
import sys
from pathlib import Path
from PIL import Image

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
for name in sys.argv[1:]:
    source = Path(name)
    with Image.open(source) as image:
        image.convert('RGB').save(source.with_suffix('.bmp'), format='BMP')
    print(source.with_suffix('.bmp'))
