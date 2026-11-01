"""Shared layout for the Parts 1-4 report and the separate Transformations document.

Parses the command line, registers the fonts, defines the styles and the
flowables, checks the audit manifest, and reads artifacts from output/. Every
artifact a document reads goes through data(), rows(), points() or art(), so
the document can check that its traceability table names each of them.
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Flowable, Image, KeepTogether


def require(condition, message):
    """An explicit check that survives python -O, unlike assert."""
    if not condition:
        raise ValueError(message)


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=(
    'Lay out the Parts 1-4 report (--part main, the submitted PDF) or the separate Transformations '
    'experiment (--part transformations, not submitted) from audited Rust artifacts; neither renders coverage. '
    'Run after audit.py. Requires reportlab and Pillow. On Windows the default font is Microsoft '
    'JhengHei; pass --font and --bold-font for another TrueType font.'))
parser.add_argument('output', nargs='?', default=str(ROOT / 'output'))
parser.add_argument('--font', default='C:/Windows/Fonts/msjh.ttc')
parser.add_argument('--bold-font', default='C:/Windows/Fonts/msjhbd.ttc')
parser.add_argument('--lang', choices=['zh', 'en'], default='zh')
parser.add_argument('--part', choices=['main', 'transformations'], default='main')
args = parser.parse_args()


def T(zh, en):
    """Pick the report language; both versions stay side by side in the source."""
    return en if args.lang == 'en' else zh


OUT = Path(args.output).resolve()
pdfmetrics.registerFont(TTFont('CJK', args.font, subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJKB', args.bold_font, subfontIndex=0))
pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJKB')
INK = colors.HexColor('#17354a')
BLUE = colors.HexColor('#147ca3')
ORANGE = colors.HexColor('#d26225')
GRAY = colors.HexColor('#78909c')
HIGHLIGHT = colors.HexColor('#fdf0e2')
W = A4[0] - 84
# CJK wrapping breaks anywhere; English text must wrap at spaces instead. English runs
# longer than Chinese, so its type is set slightly tighter to keep each section on its page.
body = ParagraphStyle('body', fontName='CJK', fontSize=T(10.2, 9.7), leading=T(16, 14.4), wordWrap=T('CJK', None), textColor=INK, spaceAfter=T(8, 7))
small = ParagraphStyle('small', parent=body, fontSize=T(8.6, 8.2), leading=T(13, 11.8), spaceAfter=6)
heading = ParagraphStyle('heading', parent=body, fontName='CJKB', fontSize=22, leading=30, spaceAfter=14)
subheading = ParagraphStyle('subheading', parent=body, fontName='CJKB', fontSize=13, leading=19, spaceAfter=9)
caption = ParagraphStyle('caption', parent=small, alignment=1)
code = ParagraphStyle('code', parent=small, fontName='Courier', fontSize=7, leading=10)
story = []

# ---------------------------------------------------------------- artifacts --
USED = set()


def art(name):
    """Path of an output artifact, recorded for the traceability check."""
    USED.add(name)
    return str(OUT / name)


def exists(name):
    return (OUT / name).exists()


def data(name):
    USED.add(name)
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def rows(name):
    USED.add(name)
    with (OUT / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def points(name):
    return [(float(r['x']), float(r['y'])) for r in rows(name)]


def load_evidence():
    """The run and audit manifests, after checking that nothing has changed since the audit."""
    run = data('run.json')
    audit = data('evidence_manifest.json')
    require(run['status'] == 'complete' and audit['status'] == 'passed' and audit['run'] == run, 'Run audit.py first')
    for name, digest in audit['artifact_sha256'].items():
        require(hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, f'Changed artifact: {name}; rerun audit')
    for name, digest in audit['source_sha256'].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Changed source: {name}; rebuild and audit')
    return run, audit


# ------------------------------------------------------------------- layout --
def p(text, style=body):
    return Paragraph(text, style)


def add(text, style=body):
    story.append(p(text, style))


def qa(question, answer):
    """An answer to one of the assignment's questions, led by the question in bold."""
    story.append(p(f'<b>{question}</b> {answer}'))


def bullet(text, style=body):
    story.append(p('• ' + text, style))


SECTION = [0, '']


def title(text, tag, new_page=True):
    """Starts a section, on a new page unless new_page is False; sections are numbered in order
    of appearance, with the document's prefix."""
    if story and new_page:
        story.append(PageBreak())
    elif story:
        story.append(Spacer(1, 10))
    SECTION[0] += 1
    story.append(p(f'{SECTION[1]}{SECTION[0]:02d} / {tag}', small))
    story.append(p(text, heading))


def table(values, widths=None, highlight=(), style=small, keep=False):
    """A table of text; rows listed in `highlight` (1 = first body row) get a tinted background.
    With keep, the table is never split across pages."""
    formatted = [[p(escape(str(cell)), style) for cell in row] for row in values]
    t = Table(formatted, colWidths=widths or [W / len(values[0])] * len(values[0]), hAlign='LEFT', repeatRows=1)
    commands = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5eff4')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.7, GRAY), ('LINEBELOW', (0, 1), (-1, -1), 0.3, colors.HexColor('#d9e2e6')), ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    commands += [('BACKGROUND', (0, r), (-1, r), HIGHLIGHT) for r in highlight]
    t.setStyle(TableStyle(commands))
    story.append(KeepTogether([t, Spacer(1, 9)]) if keep else t)
    if not keep:
        story.append(Spacer(1, 9))


def grid(cells, columns, valign='TOP', pad=2):
    """A borderless layout table of flowables, centered in equal columns."""
    t = Table(cells, colWidths=[W / columns] * columns)
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), valign), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('BOTTOMPADDING', (0, 0), (-1, -1), pad)]))
    story.append(t)


def gallery(names, captions, size):
    """One row of Rust BMPs with captions underneath."""
    grid([[Image(art(n), width=size, height=size) for n in names], [p(c, caption) for c in captions]], len(names))


def build(destination, header, footer, pdf_title):
    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont('CJK', 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(42, A4[1] - 27, header)
        canvas.drawString(42, 24, footer)
        canvas.drawRightString(A4[0] - 42, 24, str(doc.page))
        canvas.restoreState()
    destination.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=48, bottomMargin=42, title=pdf_title, author='Multimedia experiments', pageCompression=1)
    doc.build(story, onFirstPage=page, onLaterPages=page)


def check_traced(trace):
    """Every artifact this document read must match a pattern of its traceability table."""
    patterns = [pattern for _, _, pats in trace for pattern in pats]
    missing = sorted(name for name in USED if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns))
    require(not missing, f'Artifacts used but not listed in the traceability table: {missing}')


# ---------------------------------------------------------------- flowables --
class LogChart(Flowable):
    """Line chart with x = log2(N), so 9..16385 fit; with logy the y axis is log10 too."""

    def __init__(self, series, width=W, height=190, ylabel='', xlabel='', logy=False):
        Flowable.__init__(self)
        self.series = series
        self.width = width
        self.height = height
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.logy = logy

    def draw(self):
        c = self.canv
        pts = [q for _, _, s in self.series for q in s]
        xmin, xmax = (min((x for x, _ in pts)), max((x for x, _ in pts)))
        left, bottom, right, top = (52, 34, self.width - 12, self.height - 30)
        px = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
        if self.logy:
            lo = math.floor(math.log10(min(y for _, y in pts)))
            hi = math.ceil(math.log10(max(y for _, y in pts)))
            py = lambda y: bottom + (math.log10(y) - lo) / (hi - lo) * (top - bottom)
            ticks = [m * 10 ** e for e in range(lo, hi + 1) for m in (1, 3) if lo <= math.log10(m * 10 ** e) <= hi]
        else:
            ymax = max((y for _, y in pts)) * 1.06
            py = lambda y: bottom + y / ymax * (top - bottom)
            ticks = [ymax * i / 4 for i in range(5)]
        c.setFont('CJK', 8)
        c.setStrokeColor(GRAY)
        c.setLineWidth(0.5)
        for y in ticks:
            c.line(left, py(y), right, py(y))
            c.setFillColor(INK)
            c.drawRightString(left - 6, py(y) - 3, f'{y:,.0f}')
        for n in (9, 17, 33, 65, 129, 257, 1025, 4097, 16385):
            x = math.log2(n)
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                c.drawCentredString(px(x), bottom - 13, f'{n:,}')
        for i, (name, color, s) in enumerate(self.series):
            path = c.beginPath()
            path.moveTo(px(s[0][0]), py(s[0][1]))
            for x, y in s[1:]:
                path.lineTo(px(x), py(y))
            c.setStrokeColor(color)
            c.setLineWidth(1.5)
            c.drawPath(path)
            c.setFillColor(color)
            for x, y in s:
                c.circle(px(x), py(y), 2, stroke=0, fill=1)
            c.drawString(left + i * 170, self.height - 13, name)
        c.setFillColor(INK)
        c.drawCentredString((left + right) / 2, 4, self.xlabel)
        c.saveState()
        c.translate(11, (bottom + top) / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, self.ylabel)
        c.restoreState()


class Geometry(Flowable):
    """Computed polylines over control polygons, inside a frame showing the canvas."""

    def __init__(self, series, controls, width, height, size, labels=()):
        Flowable.__init__(self)
        self.series = series
        self.controls = controls
        self.width = width
        self.height = height
        self.size = size
        self.labels = labels

    def draw(self):
        c = self.canv
        c.setFillColor(colors.white)
        c.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        scale = min(self.width, self.height) / self.size
        offset = (self.width - self.size * scale) / 2
        c.setStrokeColor(colors.HexColor('#d9e2e6'))
        c.setLineWidth(0.6)
        c.rect(offset, self.height - self.size * scale, self.size * scale, self.size * scale, stroke=1, fill=0)

        def line(points, color, line_width):
            path = c.beginPath()
            path.moveTo(offset + points[0][0] * scale, self.height - points[0][1] * scale)
            for x, y in points[1:]:
                path.lineTo(offset + x * scale, self.height - y * scale)
            c.setStrokeColor(color)
            c.setLineWidth(line_width)
            c.drawPath(path)
        for control in self.controls:
            line(control, GRAY, 0.7)
            c.setFillColor(GRAY)
            for x, y in control:
                c.circle(offset + x * scale, self.height - y * scale, 2, stroke=0, fill=1)
        for pts, color in self.series:
            line(pts, color, 1.1)
        c.setFont('CJK', 8)
        c.setFillColor(ORANGE)
        for x, y, text in self.labels:
            c.drawString(offset + x * scale + 5, self.height - y * scale + 5, text)


class Chart(Flowable):

    def __init__(self, series, width=W, height=170, ymax=None, xlabel='', ylabel=''):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.series = series
        self.ymax = ymax
        self.xlabel = xlabel
        self.ylabel = ylabel

    def draw(self):
        c = self.canv
        allpoints = [q for _, _, ps in self.series for q in ps]
        xmin = min((x for x, y in allpoints))
        xmax = max((x for x, y in allpoints))
        ymax = self.ymax or max((y for x, y in allpoints)) * 1.06
        left = 48
        bottom = 32
        right = self.width - 12
        top = self.height - 30
        px = lambda x: left + (x - xmin) / (xmax - xmin or 1) * (right - left)
        py = lambda y: bottom + y / (ymax or 1) * (top - bottom)
        c.setFont('CJK', 8)
        c.setStrokeColor(GRAY)
        c.setLineWidth(0.5)
        for i in range(5):
            y = ymax * i / 4
            c.line(left, py(y), right, py(y))
            c.setFillColor(INK)
            c.drawRightString(left - 6, py(y) - 3, f'{y:.2f}')
            x = xmin + (xmax - xmin) * i / 4
            c.drawCentredString(px(x), bottom - 13, f'{x:.1f}')
        for i, (name, color, pts) in enumerate(self.series):
            path = c.beginPath()
            path.moveTo(px(pts[0][0]), py(pts[0][1]))
            for x, y in pts[1:]:
                path.lineTo(px(x), py(y))
            c.setStrokeColor(color)
            c.setLineWidth(1.5)
            c.drawPath(path)
            c.setFillColor(color)
            c.drawString(left + i * 180, self.height - 13, name)
        c.setFillColor(INK)
        c.drawCentredString((left + right) / 2, 2, self.xlabel)
        c.saveState()
        c.translate(10, (bottom + top) / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, self.ylabel)
        c.restoreState()


class Crop(Flowable):
    """Enlarged clip (x0, y0, x1, y1) of an existing bitmap; no coverage computation or resampling."""

    def __init__(self, name, box, width):
        Flowable.__init__(self)
        self.name = name
        self.box = box
        self.width = width
        self.height = width * (box[3] - box[1]) / (box[2] - box[0])

    def draw(self):
        c = self.canv
        x0, y0, x1, y1 = self.box
        reader = ImageReader(art(self.name))
        iw, ih = reader.getSize()
        factor = self.width / (x1 - x0)
        c.saveState()
        clip = c.beginPath()
        clip.rect(0, 0, self.width, self.height)
        c.clipPath(clip, stroke=0)
        c.drawImage(reader, -x0 * factor, (y1 - ih) * factor, width=iw * factor, height=ih * factor)
        c.restoreState()


class RowLocation(Flowable):
    """The k=8 star with the measured stretch of the pixel row marked in orange."""

    def __init__(self, selection, width, size):
        Flowable.__init__(self)
        self.selection = selection
        self.width = width
        self.height = width
        self.size = size

    def draw(self):
        c = self.canv
        s = self.selection
        scale = self.width / self.size
        c.drawImage(ImageReader(art('star_k8.bmp')), 0, 0, self.width, self.height)
        c.setStrokeColor(ORANGE)
        c.setLineWidth(2)
        y = self.height - (s['row'] + 0.5) * scale
        c.line(s['x_start'] * scale, y, s['x_end_exclusive'] * scale, y)
        c.circle(s['crossing_x'] * scale, y, 5, stroke=1, fill=0)
