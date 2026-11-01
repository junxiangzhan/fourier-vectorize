"""Part 5: team member list and final project description, as one PDF.

Run: python scripts/build_part5.py [output-directory] --lang zh|en
Requires ReportLab; fonts as in build_report.py.
"""
import argparse
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
# Personal details live in personal.json (git-ignored); see personal.example.json.
_personal = ROOT / 'personal.json'
if not _personal.is_file():
    raise SystemExit('copy personal.example.json to personal.json and fill in your details')
ME = json.loads(_personal.read_text(encoding='utf-8'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', nargs='?', default=str(ROOT / 'output'))
parser.add_argument('--lang', choices=['zh', 'en'], default='zh')
parser.add_argument('--font', default='C:/Windows/Fonts/msjh.ttc')
parser.add_argument('--bold-font', default='C:/Windows/Fonts/msjhbd.ttc')
args = parser.parse_args()
OUT = Path(args.output).resolve()
pdfmetrics.registerFont(TTFont('CJK', args.font, subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJKB', args.bold_font, subfontIndex=0))
pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJKB')

def T(zh, en):
    return en if args.lang == 'en' else zh

INK = colors.HexColor('#17354a')
GRAY = colors.HexColor('#78909c')
W = A4[0] - 84
body = ParagraphStyle('body', fontName='CJK', fontSize=10.2 if args.lang == 'zh' else 9.8, leading=16 if args.lang == 'zh' else 14.5, wordWrap='CJK' if args.lang == 'zh' else None, textColor=INK, spaceAfter=7)
small = ParagraphStyle('small', parent=body, fontSize=8.8, leading=13, spaceAfter=4)
item = ParagraphStyle('item', parent=body, leftIndent=10, spaceAfter=2)
heading = ParagraphStyle('heading', parent=body, fontName='CJKB', fontSize=20, leading=27, spaceAfter=12)
subheading = ParagraphStyle('subheading', parent=body, fontName='CJKB', fontSize=12.5, leading=18, spaceBefore=4, spaceAfter=6)
story = []

def p(text, style=body):
    return Paragraph(text, style)

def bullets(items):
    for text in items:
        story.append(p('• ' + text, item))

story.append(p(T('Part 5 · 期末專題與組員名單', 'Part 5 · Final Project & Team Member List'), small))
story.append(p(T('期末專題與組員名單', 'Final Project and Team Member List'), heading))
story.append(p(T('組員名單', 'Team member list'), subheading))
member = [
    [T('學號', 'Student ID'), ME['student_id']],
    [T('班級（學校、學院、系所）', 'Class (University, College/School, Department)'),
     T(ME['programme_zh'], ME['programme_en'])],
    [T('姓名', 'Name'), ME['name']],
    ['Email', ME['email']],
    [T('期末專題主題', 'Topic of the final project'),
     T('升維計算的事前成本估算：一維長序列運算的多維切分決策', 'Estimating before running: a cost model for lifting long one-dimensional computations into multi-dimensional layouts')],
]
t = Table([[p(f'<b>{k}</b>', small), p(v, small)] for k, v in member], colWidths=[150, W - 150], hAlign='LEFT')
t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -1), 0.3, colors.HexColor('#d9e2e6')), ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e5eff4')), ('LEFTPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
story.extend([t, Spacer(1, 6), p(T('本組僅一名組員。', 'This team has one member.'), small)])

story.append(p(T('專題說明', 'Project description'), subheading))
story.append(p(T('<b>目的。</b>許多常見計算可以把一條很長的一維資料重新排列成多維陣列來處理，本專題稱之為「升維」。例如 AI 模型中的長序列卷積、快速傅立葉轉換（FFT 的 Cooley–Tukey 分解即為一例）、多項式乘法與大整數乘法。升維不一定更快：多維計算的各步驟之間必須搬移資料、轉換格式，部分方法還需要事先建立輔助結構，例如預先計算的旋轉因子表。實務上，工程師多半以反覆試錯決定切分方式。本專題提出一套「執行前估算」的成本模型，在實際執行前預測各候選切法的總成本，並判斷升維在該情境是否值得。',
                 '<b>Purpose.</b> Many common computations can be carried out by rearranging one long sequence of data into a multi-dimensional array, which we call dimension lifting. Examples include long-sequence convolution in AI models, the fast Fourier transform (the Cooley-Tukey decomposition is one instance), polynomial multiplication and big-integer multiplication. Lifting does not always pay off: the stages of a multi-dimensional computation must move data between them and convert its layout, and some methods also need auxiliary structures built in advance, such as precomputed twiddle-factor tables. In practice, engineers usually choose the decomposition by trial and error. This project proposes an estimate-before-you-run cost model that predicts the total cost of each candidate decomposition before execution and judges whether lifting is worthwhile in the given setting.')))
story.append(p(T('<b>方法。</b>模型把總成本拆成三部分：真正的計算、資料搬運與格式轉換，以及依重複使用次數攤提的一次性建置。各部分以目標硬體上實測的單位成本為參數，代入每一種候選切法後，推薦成本最低者。',
                 '<b>Approach.</b> The model splits the total cost into three parts: the actual computation, data movement and layout conversion, and one-time setup amortized over the number of reuses. Each part is parameterized by unit costs measured on the target hardware; every candidate decomposition is evaluated with these parameters and the cheapest one is recommended.')))
story.append(p(T('<b>輸入</b>', '<b>Input</b>')))
bullets([T('要執行的運算與問題規模（例如序列長度、多項式次數、整數位數）。', 'The operation to perform and the problem size (for example sequence length, polynomial degree or number of digits).'),
         T('目標硬體上各種切法的實測單位成本：計算、搬運與轉換。', 'Measured unit costs of each candidate decomposition on the target hardware: computation, movement and conversion.'),
         T('預先建立的結構會被重複使用幾次，用於攤提其建置成本。', 'How many times precomputed structures will be reused, used to amortize their setup cost.')])
story.append(p(T('<b>輸出</b>', '<b>Output</b>')))
bullets([T('建議的切法：切成幾維、每一維的長度。', 'A recommended decomposition: how many dimensions, and the length of each.'),
         T('預估的總成本，以及成本組成：計算、搬運與轉換、一次性建置。', 'The predicted total cost and its breakdown into computation, movement and conversion, and one-time setup.'),
         T('適用性判斷：此方法在該情境是否適用、預估是否可信；模型假設不成立時給出警示。', 'An applicability judgment: whether the method suits this setting and how far the prediction can be trusted, with a warning when the model\'s assumptions do not hold.')])
story.append(p(T('<b>驗證方式。</b>在 CPU 與一張筆電 GPU 上比較預估成本與實測執行時間，並分別檢視以搬運為主與以計算為主的步驟，如實列出模型準確與不準確之處。',
                 '<b>Validation.</b> Predicted costs are compared with measured run times on a CPU and on a laptop GPU. Movement-dominated and computation-dominated steps are examined separately, and the report states plainly where the model is accurate and where it is not.')))

def page(canvas, doc):
    canvas.saveState()
    canvas.setFont('CJK', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(42, A4[1] - 27, 'MULTIMEDIA SYSTEMS / ASSIGNMENT 1151')
    canvas.drawString(42, 24, 'Part 5 · ' + ME['student_id'])
    canvas.drawRightString(A4[0] - 42, 24, str(doc.page))
    canvas.restoreState()
pdf_dir = OUT / 'pdf'
pdf_dir.mkdir(parents=True, exist_ok=True)
destination = pdf_dir / f'assignment_1151_part5_{args.lang}.pdf'
doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=48, bottomMargin=42, title='Assignment 1151: Part 5', author=ME['student_id'], pageCompression=1)
doc.build(story, onFirstPage=page, onLaterPages=page)
print(destination)
