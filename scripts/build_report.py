"""Lay out the Parts 1-4 report, or the separate Transformations experiment, from audited
Rust artifacts.

--part main (default) writes the submitted report, output/pdf/assignment_1151_parts_1_4_{lang}.pdf:
Parts 1-3, Part 4 (vectorizing the star), the extended Part 4 experiments (several shapes,
holes, color, paintings, high fidelity, related work, development notes) and traceability.
--part transformations writes output/pdf/assignment_1151_transformations_{lang}.pdf, which is not
submitted: the rotation experiment, another option on the Part 4 list.

Run after audit.py; nothing here renders coverage. Requires reportlab and Pillow; layout
helpers live in report_common.py. Pass --lang zh (default) or --lang en.
"""
import itertools
import math
import statistics
import sys
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage
from reportlab.platypus import Image, Spacer, Table, TableStyle

# Also under python -I, which leaves this folder off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_common import (  # noqa: E402
    OUT, SECTION, args, T, W, require, story, art, exists, data, rows, points,
                           load_evidence, p, add, qa, bullet, title, table, grid, gallery, build,
                           check_traced, body, small, subheading, caption, code, BLUE, ORANGE,
                           LogChart, Geometry, Chart, Crop, RowLocation)

run, audit = load_evidence()
size = run['canvas']


# =============================================================== main report ==

def cover():
    wheel = data('wheel_summary.json')
    title(T('多媒體系統實驗報告', 'Multimedia Systems Lab Report'), 'Assignment 1151 · Parts 1-4')
    add(T('本報告涵蓋第一至四題：Bézier 曲線、non-zero winding 填色、supersampling，以及 Part 4 的向量化（Vectorizing）。Part 4 先回答作業的問題（還原星形輪廓），之後是延伸實驗：多個形狀、挖洞、彩色、兩幅名畫與高精度重建。所有曲線座標、覆蓋率、影像及計時來自同一個 Rust 程式；Part 5 另附 PDF。',
          'This report covers Parts 1 to 4: Bézier curves, non-zero winding fill, supersampling, and vectorizing for Part 4. Part 4 first answers the assignment\'s question (recovering the star\'s outline), then extends the work: several shapes, holes, color, two paintings and a high-fidelity reconstruction. All curve coordinates, coverage values, images and timings come from a single Rust program; Part 5 is submitted as a separate PDF.'))
    story.append(p(T('重現方式', 'How to reproduce'), subheading))
    for command in ['cargo build --offline --release', 'python scripts/decode_paintings.py', 'cargo run --offline --release -- all --out output --size 384 --wedges 96 --paintings assets/paintings', 'cargo run --offline --release -- hifi --out output --paintings assets/paintings', 'cargo test --offline', 'python scripts/audit.py output', T('python scripts/build_report.py output', 'python scripts/build_report.py output --lang en')]:
        story.append(p(escape(command), small))
    add(T('第 1 與第 3 行完成全部實驗，Rust 程式不需 Python 或 Node.js。第 2 行只用 Pillow 把兩幅畫作的 JPEG 解碼成 BMP（只解碼、不縮放），供 Part 4 延伸實驗中的名畫使用；第 4 行另外執行高精度實驗（約 15 分鐘）。最後兩行分別使用 Pillow 獨立解碼與驗證、使用 ReportLab 排版 PDF；它們不參與覆蓋率計算。需將 Rust 的 cargo 與 rustc 放在 PATH；沒有專案必填環境變數。',
          "Lines 1 and 3 run every experiment; the Rust program needs neither Python nor Node.js. Line 2 only uses Pillow to decode the two painting JPEGs into BMP (decoding only, no resizing) for the paintings in the extended Part 4 experiments; line 4 runs the high-fidelity experiment separately (about 15 minutes). The last two lines use Pillow for independent decoding and verification and ReportLab to lay out this PDF; neither takes part in computing coverage. Rust's cargo and rustc must be on PATH; the project requires no environment variables."))
    table([[T('本次環境／參數', 'Environment / parameter'), T('記錄', 'Value')], ['OS / architecture', f"{run['os']} / {run['architecture']}"], ['Rust', run['rustc']], [T('畫布與輪盤', 'Canvas and wheel'), T(f"{size} × {size}；{wheel['wedges']} 個楔形", f"{size} × {size}; {wheel['wedges']} wedges")], [T('程式版本', 'Program version'), run['package_version']], [T('建置輸入指紋', 'Build-input fingerprint'), run['source_fingerprint']]], [150, W - 150])
    story.append(p(T('共用工具庫與座標慣例', 'Shared library and conventions'), subheading))
    add(T('Matrix&lt;R,C&gt; 用 [[f64; C]; R] 儲存，尺寸不相容的乘法不能編譯；Vector&lt;N&gt; 共用 N×1 矩陣表示。Point2 區分位置與位移。Image&lt;T&gt; 儲存動態畫布，Image&lt;f64&gt; 保留覆蓋率，Color 使用浮點 RGB；BMP 存檔才量化。曲線、路徑、scanline 填色、區塊平均與 24-bit BMP 編碼均自行實作，無外部 Rust crate。',
          'Matrix&lt;R,C&gt; is stored as [[f64; C]; R], so a multiplication with incompatible dimensions does not compile; Vector&lt;N&gt; shares the N×1 matrix representation. Point2 distinguishes positions from displacements. Image&lt;T&gt; holds a dynamically sized canvas, Image&lt;f64&gt; keeps coverage values, and Color uses floating-point RGB; values are quantized only when a BMP is saved. Curves, paths, the scanline fill, block averaging and 24-bit BMP encoding are all implemented from scratch, with no external Rust crates.'))
    add(T('座標原點在左上，x 向右、y 向下；像素中心為 (x+0.5,y+0.5)，正旋轉視覺上為順時針。每題的作答以粗體問句起頭。',
          'The origin is at the top left, with x pointing right and y pointing down; pixel centers are at (x+0.5,y+0.5), and a positive rotation appears clockwise. Each answer to a question of the assignment starts with that question in bold.'))


def part1_curves():
    curve = data('curve_summary.json')
    conv = rows('convergence.csv')
    title(T('兩種曲線構造與收斂', 'Two constructions and their convergence'), 'Part 1 · P1-1 / P1-2')
    add(T('公式使用 [1,t,t²,t³] B P，自製矩陣乘法算出位置。中點構造獨立使用 de Casteljau 的 t=0.5 分割；深度 d 的折線有 2^d 段，深度 0 僅連兩端點。',
          'The formula evaluates [1,t,t²,t³] B P, computing positions with our own matrix multiplication. The mid-point construction independently uses de Casteljau splitting at t=0.5; the polyline at depth d has 2^d segments, and depth 0 only joins the two endpoints.'))
    story.append(Geometry([(points('curve_formula.csv'), BLUE), (points('curve_depth2.csv'), ORANGE)], [points('curve_controls.csv')], W, 220, size))
    story.append(p(T('藍：公式密集取樣；橘：深度 2 中點折線；灰：控制多邊形；外框為整個畫布。僅顯示已計算的點與線段。',
                     'Blue: dense sampling of the formula; orange: depth-2 mid-point polyline; gray: control polygon; the frame is the whole canvas. Only computed points and segments are drawn.'), small))
    first = curve['first_depth_below_half_pixel']
    add(T(f"公式共有 {curve['formula_samples']:,} 個樣本，實測最大相鄰間距 {curve['max_spacing_px']:.9f} px &lt; 0.1 px。對每個公式樣本取到最近折線線段的距離，再取最大值；首次 &lt; 0.5 px 是<b>深度 {first}</b>（表中標色列）。",
          f"The formula has {curve['formula_samples']:,} samples, with a measured maximum spacing between neighbors of {curve['max_spacing_px']:.9f} px &lt; 0.1 px. For every formula sample we take its distance to the nearest polyline segment and report the largest; the gap first falls below 0.5 px at <b>depth {first}</b> (the tinted row)."))
    table([[T('深度', 'Depth'), T('線段數', 'Segments'), T('最大誤差 / px', 'Max error / px'), T('相對前層', 'Ratio to previous')]] + [[r['depth'], r['segments'], f"{float(r['max_gap_px']):.6f}", r['ratio_to_previous'][:8] or '-'] for r in conv],
          highlight=[i + 1 for i, r in enumerate(conv) if int(r['depth']) == first])
    qa(T('每加深一層，誤差遵循什麼規則？淺層是否同樣符合？', 'What rule do the errors follow per extra depth, and do the shallow depths obey it?'),
       T('實測深層比值接近 0.25，即線段數加倍後誤差約剩四分之一。淺層較偏離，因長弦尚不能代表局部曲率。',
         'At the deeper levels the measured ratio is close to 0.25: each time the number of segments doubles, the error drops to about a quarter. The shallow depths deviate more, because long chords cannot yet represent the local curvature.'))
    qa(T('中點構造能做到、而公式做不到的一件事？', 'One thing the mid-point construction allows that the formula does not?'),
       T('局部自適應細分：只分割控制多邊形不夠平坦的區間，不必事先決定 t 的取樣數。本次為比較收斂而固定使用均勻深度。',
         'Locally adaptive subdivision: split only the intervals whose control polygon is not flat enough, instead of deciding the number of t values in advance. Here a uniform depth is used throughout so that convergence can be compared.'))


def part1_joins():
    join_controls = rows('joins_controls.csv')
    metrics = rows('joins_metrics.csv')
    metric = {(r['variant'], r['join']): r for r in metrics}
    corner_angle = float(metric[('equal', '1')]['tangent_angle_deg'])
    title(T('方向控制線與三段接點', 'Direction lines and the joins of three segments'), 'Part 1 · P1-3')
    geometries = []
    for variant in ('equal', 'unequal'):
        controls = [[(float(r['x']), float(r['y'])) for r in join_controls if r['variant'] == variant and r['segment'] == segment] for segment in ('0', '1', '2')]
        labels = [(controls[0][-1][0], controls[0][-1][1], T('接點 0', 'join 0')), (controls[1][-1][0], controls[1][-1][1], T('接點 1', 'join 1'))]
        geometries.append(Geometry([(points(f'joins_{variant}.csv'), BLUE)], controls, W / 2 - 7, 230, size, labels))
    t = Table([[p(T('等長方向線', 'Equal-length direction lines'), subheading), p(T('不等長方向線（3 倍）', 'Unequal direction lines (3×)'), subheading)], geometries], colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t)
    add(T('同一條開放路徑串接三段 cubic，接點 0 平滑、接點 1 為角點。灰色線段連接控制點，藍色線由深度 7 細分的已計算座標構成。不等長版本只將接點 0 的離開方向線拉長為 3 倍，接點位置與其他控制點保持相同。',
          'One open path chains three cubic segments; join 0 is smooth and join 1 is a corner point. Gray segments connect the control points, and the blue line consists of computed coordinates from subdivision depth 7. The unequal version only stretches the outgoing direction line of join 0 to 3 times its length; the join positions and all other control points stay the same.'))
    qa(T('角點為什麼是角點？', 'What makes the corner a corner?'),
       T(f'接點 1 兩側的方向控制線不共線，進入與離開切線夾角約 {corner_angle:.2f}°。',
         f'At join 1 the two direction lines are not collinear: the incoming and outgoing tangents meet at about {corner_angle:.2f}°.'))
    table([[T('版本 / 接點', 'Version / join'), T('進入線長', 'Incoming length'), T('離開線長', 'Outgoing length'), T('夾角 / °', 'Angle / °')]] + [[f"{r['variant']} / {r['join']}", f"{float(r['incoming_length']):.3f}", f"{float(r['outgoing_length']):.3f}", f"{float(r['tangent_angle_deg']):.6f}"] for r in metrics])
    unequal_join = metric[('unequal', '0')]
    speed_ratio = float(unequal_join['outgoing_length']) / float(unequal_join['incoming_length'])
    qa(T('不等長的接點看起來還平滑嗎？', 'Does the unequal-length join still look smooth?'),
       T('是。接點 0 的方向線共線，沿路徑前進的切向量同方向，兩個版本都具幾何平滑性 G1；切線夾角為 0°，沒有折角，只是第二段曲線因方向線拉長而改變形狀。',
         'Yes. At join 0 the direction lines are collinear and the tangent vectors along the direction of travel point the same way, so both versions are geometrically smooth (G1): the angle between the tangents is 0° and there is no kink; only the shape of the second segment changes, because its direction line is longer.'))
    qa(T('形狀不需要等長，那等長換到什麼？', 'Since the shape does not need equal lengths, what do they buy?'),
       T(f"參數上的連續。兩段使用相同參數區間時，等長方向線保證一階導數 C′(t) 的大小與方向都相等，即 C1。實際差別在沿 t 前進的速度：不等長版本在接點 0 的速度瞬間變為 {speed_ratio:.2f} 倍；沿路徑做動畫或以均勻 t 取樣時，點的疏密會在接點突變，等長則兩側一致。",
         f"Parametric continuity. When both segments use the same parameter interval, equal direction lines guarantee that the first derivative C′(t) has the same magnitude and direction on both sides, i.e. C1. The practical difference is the speed of travel along t: in the unequal version the speed at join 0 jumps instantly by a factor of {speed_ratio:.2f}, so when animating along the path, or sampling at uniform t, the spacing of the points changes abruptly at the join; with equal lengths it is the same on both sides."))


def part2_fill():
    star = data('star_summary.json')
    title(T('帶符號繞數的 scanline 填色', 'Scanline fill with signed winding numbers'), 'Part 2 · P2-1 / P2-2')
    t = Table([[Image(art('star.bmp'), width=250, height=250), p(T('星形使用五個圓周頂點，以 0→2→4→1→3 的隔點順序訪問，形成單一自交封閉路徑。',
                                                                   'The star uses five vertices on a circle, visited in "every other one" order 0→2→4→1→3, forming a single self-intersecting closed path.'), body)]], colWidths=[270, W - 270])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    story.append(t)
    qa(T('星形中心的繞數？', 'The winding number at the center of the star?'),
       T(f"<b>{star['winding']:+d}</b>，因此中央填黑；even-odd 在此會留下洞。",
         f"<b>{star['winding']:+d}</b>, so the center is filled black; even-odd would leave a hole there."))
    story.append(p(T('每列的判定方式', 'How each row is decided'), subheading))
    add(T('在 y+0.5 求各非水平邊的交點，排序並合併相同 x 的貢獻；從左到右累加 signed crossing，將 running winding 非零的區間寫入影像。查詢單點時使用朝右的射線，順時針為正；兩條流程符號一致。',
          'At y+0.5 we find the crossings of every non-horizontal edge, sort them and merge contributions at the same x; walking from left to right we accumulate the signed crossings and write every span where the running winding number is non-zero into the image. Single-point queries use a ray pointing right, with clockwise positive; both code paths use the same sign.'))
    add(T('<b>符號慣例固定為：</b>座標 y 向下；由左往右掃描時，<b>向上穿過該列的邊（終點 y 小於起點 y）記 +1，向下穿過的邊記 -1</b>。因此螢幕上順時針的輪廓內部為 +1，全程不再更改。',
          '<b>The sign convention, settled once:</b> y points down; scanning from left to right, <b>an edge that passes upward through the row (end y smaller than start y) counts +1, and one that passes downward counts -1</b>. The inside of an outline that is clockwise on screen is therefore +1, and this never changes.'))
    add(T('邊的 y 範圍採下界包含、上界排除，避免共用頂點重複計入。忽略水平／零長度邊；畫布外的交點仍參與繞數累加，僅寫入像素時裁切。像素中心位於左邊界時填入，右邊界時排除。',
          "Each edge's y range includes its lower bound and excludes its upper bound, so a shared vertex is not counted twice. Horizontal and zero-length edges are ignored; crossings outside the canvas still contribute to the winding sum and are clipped only when pixels are written. A pixel center lying exactly on a left boundary is filled; one on a right boundary is not."))
    add(T('Rust 測試涵蓋自交星形、多子路徑、同向與反向輪廓、頂點列、畫布外包圍輪廓及手算三角形。開放路徑、非有限座標與無效尺寸回傳錯誤，不靜默補成封閉輪廓。',
          'The Rust tests cover the self-intersecting star, multiple sub-paths, same- and opposite-direction outlines, rows through vertices, an enclosing outline outside the canvas, and a hand-computed triangle. Open paths, non-finite coordinates and invalid sizes return errors instead of being silently closed.'))


def part2_ring():
    ring = data('ring_summary.json')
    same, opposite = (ring['same_winding'], ring['opposite_winding'])
    title(T('只反轉內圈的圓環比較', 'Reversing only the inner circle'), 'Part 2 · P2-3')
    t = Table([[p(T(f'同向：中心 {same:+d}', f'Same direction: center {same:+d}'), subheading), p(T(f'反向內圈：中心 {opposite}', f'Inner circle reversed: center {opposite}'), subheading)], [Image(art('ring_same.bmp'), width=235, height=235), Image(art('ring_opposite.bmp'), width=235, height=235)]], colWidths=[W / 2, W / 2])
    story.append(t)
    add(T(f"內外圈各由四段 cubic 構成。外半徑 {ring['outer_radius']:.1f} px，內半徑 {ring['inner_radius']:.1f} px，切線方向的控制線長度為 {ring['handle_ratio']}×r；使用第一題中點細分，深度 {ring['flatten_depth']}。兩圖只差內圈的遍歷方向：四段控制點及所有幾何位置均未移動，只交換遍歷方向與控制點角色。",
          f"Each circle consists of four cubic segments. The outer radius is {ring['outer_radius']:.1f} px and the inner radius {ring['inner_radius']:.1f} px; the direction lines along the tangent are {ring['handle_ratio']}×r long. The circles are flattened with the Part 1 mid-point subdivision at depth {ring['flatten_depth']}. The two images differ only in the traversal direction of the inner circle: no control point and no other geometric position moves; only the direction, and with it the roles of the control points, is swapped."))
    qa(T('兩種情況下中心的繞數，以及差異的原因？', 'The winding number at the center in each case, and why they differ?'),
       T(f'兩圈同向時，中心繞數 +1 +1 = {same:+d}，整個內部仍填色。僅反轉內圈後，中心 +1 -1 = {opposite}，形成洞；兩圈之間仍為 +1。獨立解碼確認同向圖中心黑色、反向圖中心白色。',
         f'With both circles in the same direction, the center has +1 +1 = {same:+d} and the whole interior is filled. After reversing only the inner circle, the center has +1 -1 = {opposite} and becomes a hole; the band between the circles stays at +1. Independent decoding confirms that the center is black in the same-direction image and white in the reversed one.'))
    qa(T('這對字型中字母「O」的儲存方式意味著什麼？', 'What does this imply for how the letter "O" is stored in a font?'),
       T('以 non-zero winding 填色的字型中，O 的內外輪廓必須採相反方向，使字腔內的繞數相消。外圈的正負方向取決於座標慣例，重點是兩者相對方向相反。',
         'In a font filled with the non-zero winding rule, the inner and outer outlines of the "O" must run in opposite directions, so that the winding numbers cancel inside the counter. Which direction the outer outline takes depends on the coordinate convention; what matters is that the two are opposite.'))
    table([[T('驗證項目', 'Check'), T('結果', 'Result')], [T('各圈 cubic 數', 'Cubic segments per circle'), ring['segments_per_circle']], [T('控制線比例', 'Direction-line ratio'), ring['handle_ratio']], [T('唯一變因', 'Only variable'), T('內圈遍歷方向', 'Traversal direction of the inner circle')], [T('控制點座標多重集合保持不變', 'Multiset of control-point coordinates unchanged'), ring['positions_unchanged']]], [200, W - 200])


def part3_supersampling():
    title(T('相同畫布的四種 supersampling', 'Four supersampling factors, one canvas'), 'Part 3 · I')
    add(T(f'星形繞 ({size / 2:.1f},{size / 2:.1f}) 順時針旋轉 5°，所有邊皆非水平／垂直。每個倍率將幾何與畫布同時放大 k 倍，使用同一 scanline 填色，平均 k×k 區塊回到 {size}×{size}。最大倍率 k=8 即第 II、III 節所用的倍率。',
          f'The star is rotated 5° clockwise about ({size / 2:.1f},{size / 2:.1f}), so no edge is horizontal or vertical. For each factor, the geometry and the canvas are both scaled by k, filled with the same scanline fill, and averaged over k×k blocks back to {size}×{size}. The largest factor, k=8, is the one used in II and III.'))
    cells = []
    for pair in ((1, 2), (4, 8)):
        cells.append([p(f'k = {k}', subheading) for k in pair])
        cells.append([Image(art(f'star_k{k}.bmp'), width=218, height=218) for k in pair])
    t = Table(cells, colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    story.append(t)
    add(T('每個高解析像素取 0 或 1，平均後的覆蓋率仍為 f64。Color = 背景×(1-c) + 前景×c；本次白背景、黑前景。所有數值先保存於 CSV，最後 BMP 才將通道乘 255、四捨五入。',
          'Each high-resolution pixel is 0 or 1, and the averaged coverage stays an f64. Color = background×(1-c) + foreground×c; here the background is white and the foreground black. All values are first saved to CSV; only the final BMP multiplies the channels by 255 and rounds them.'))


def part3_row():
    selection = data('row_selection.json')
    times = rows('timing.csv')
    title(T('像素列過渡與成本', 'Transition along a pixel row, and cost'), 'Part 3 · II')
    add(T(f"最平緩邊為 edge {selection['edge_index']}。取 row {selection['row']}、x={selection['x_start']}..{selection['x_end_exclusive'] - 1}，交點 x={selection['crossing_x']:.3f}，距起點比例 {selection['fraction_from_start']:.4f}。最近其他交點距離 {selection['nearest_other_crossing_distance']:.3f} px，位於選段之外。",
          f"The shallowest edge is edge {selection['edge_index']}. We take row {selection['row']}, x={selection['x_start']}..{selection['x_end_exclusive'] - 1}; the crossing is at x={selection['crossing_x']:.3f}, at fraction {selection['fraction_from_start']:.4f} of the edge from its start. The nearest other crossing is {selection['nearest_other_crossing_distance']:.3f} px away, outside the selected stretch."))
    t = Table([[RowLocation(selection, 150, size), p(T('橘色標示沿像素列的測量區段。區段靠近邊的起點，兩側分別為內／外部；沒有選在自交星形兩側都填滿的中央段。圖像直接使用 Rust BMP，標記僅為報告註記。',
                                                       'Orange marks the measured stretch along the pixel row. It lies near the start of the edge, with the inside on one side and the outside on the other; it is not in the middle of the edge, where the self-intersecting star is filled on both sides. The image is the Rust BMP itself; the marking is only a report annotation.'), body)]], colWidths=[170, W - 170])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    story.append(t)
    sample_rows = rows('row_samples.csv')
    story.append(Chart([('k=1', ORANGE, [(float(r['x']), float(r['coverage_k1'])) for r in sample_rows]), ('k=8', BLUE, [(float(r['x']), float(r['coverage_k8'])) for r in sample_rows])], height=170, ymax=1, xlabel=T('像素欄 x', 'pixel column x'), ylabel=T('覆蓋率', 'coverage')))
    story.append(Spacer(1, 6))
    qa(T('k=1 與 k=8 的曲線形狀有何不同？', 'How do the k=1 and k=8 plots differ in shape?'),
       T(f"k=1 呈現 0/1 的陡跳；k=8 出現 {selection['partial_pixels_k8']} 個部分覆蓋像素，沿平緩邊產生漸變。數值直接讀取浮點 Image，未從 BMP 反推。",
         f"k=1 shows an abrupt 0/1 step; k=8 has {selection['partial_pixels_k8']} partially covered pixels, a gradual ramp along the shallow edge. Values come from the floating-point Image, not the BMP."))
    table([['k', T('畫布', 'Canvas'), T('中位渲染時間 / ms', 'Median render time / ms')]] + [[r['k'], f'{size}×{size}', f"{float(r['median_render_ms']):.6f}"] for r in times], [65, 150, W - 215])
    medians = {int(r['k']): float(r['median_render_ms']) for r in times}
    growth = [medians[b] / medians[a] for a, b in ((1, 2), (2, 4), (4, 8))]
    overall = medians[8] / medians[2]
    exponent = math.log(overall) / math.log(4)
    qa(T('成本如何隨 k 成長？為什麼？', 'How does the cost grow with k, and why?'),
       T(f"每個 k 重新渲染三次，以 Instant 單調時鐘取中位數；包含縮放、配置、填色與平均，排除存檔與圖表。k 由 1→2、2→4、4→8，中位時間分別變為前一級的 {growth[0]:.1f}、{growth[1]:.1f}、{growth[2]:.1f} 倍；單級倍率受計時雜訊影響，因此以 k=2 到 k=8 的總倍率 {overall:.1f} 判斷：與 k² 成正比應為 16、與 k 成正比為 4，實測相當於 k 的 {exponent:.2f} 次方，{'偏向 k²' if overall > 8 else '偏向 k'}。高解析填色與區塊平均處理 O(k²) 個樣本，交點計算只隨列數 O(k) 成長，固定成本在小 k 時占比也較高，所以未達 2 次方。1→2 的倍率偏高，因為 k=1 只做一次填色，k=2 起才多出放大畫布的配置與區塊平均。",
         f"Each k is rendered three times and timed with the monotonic Instant clock, taking the median; timing covers scaling, allocation, filling and averaging, but not saving files or charts. For k = 1→2, 2→4 and 4→8 the median time grows by factors of {growth[0]:.1f}, {growth[1]:.1f} and {growth[2]:.1f}. Single-step factors fluctuate with timing noise, so we judge by the overall factor from k=2 to k=8, {overall:.1f}: cost proportional to k² would give 16, proportional to k only 4; the measurement corresponds to k to the power {exponent:.2f}, {'closer to k²' if overall > 8 else 'closer to k'}. High-resolution filling and averaging handle O(k²) samples, while the crossing computation grows only with the row count, O(k), and fixed costs weigh more at small k, so the exponent falls short of 2. The 1→2 factor is high because k=1 does a single fill; the enlarged canvas and block averaging start at k=2."))


def part3_wheel():
    wheel = data('wheel_summary.json')
    title(T('輪盤中心的取樣限制', 'The sampling limit at the center of the wheel'), 'Part 3 · III')
    add(T(f"使用 {wheel['wedges']} 個三角形子路徑，半徑 {wheel['radius']:.2f} px、相位 {wheel['phase_radians']} rad。楔形角寬與間隙相同，均為 π/{wheel['wedges']} = {wheel['wedge_angle']:.6f} rad；兩圖使用相同幾何。",
          f"The wheel uses {wheel['wedges']} triangular sub-paths, with radius {wheel['radius']:.2f} px and phase {wheel['phase_radians']} rad. Each wedge and each gap have the same angular width, π/{wheel['wedges']} = {wheel['wedge_angle']:.6f} rad; both images use the same geometry."))
    c = size / 2
    box = (c - 16, c - 16, c + 16, c + 16)
    t = Table([[p('k=1', subheading), p('k=8', subheading)], [Image(art('wheel_k1.bmp'), width=212, height=212), Image(art('wheel_k8.bmp'), width=212, height=212)], [Crop('wheel_k1.bmp', box, 115), Crop('wheel_k8.bmp', box, 115)]], colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([('ALIGN', (0, 1), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(t)
    story.append(p(T('第二列為中心 32×32 像素放大顯示，僅裁切既有影像。', 'The second row enlarges the central 32×32 pixels, cropped from the existing images only.'), caption))
    qa(T('(a) k=1 時出現的，不只是鋸齒，那是什麼？', '(a) What appears at k=1 that is not merely jagged edges?'),
       T(f"Aliasing（moiré）：中心出現粗假紋理與放射／交錯帶。k=8 中心更接近灰色；中央 16×16 區域對 0.5 覆蓋率的 RMS 偏差由 {wheel['center_rms_from_half_k1']:.6f} 降至 {wheel['center_rms_from_half_k8']:.6f}。此數值是灰色均勻程度的代理指標，不是重建誤差真值。",
         f"Aliasing (moiré): the center shows coarse false texture and radiating, interleaved bands. At k=8 the center is closer to gray; the RMS deviation of the central 16×16 region from 0.5 coverage drops from {wheel['center_rms_from_half_k1']:.6f} to {wheel['center_rms_from_half_k8']:.6f}. This number is a proxy for how uniform the gray is, not a true reconstruction error."))
    qa(T('(b) 為什麼 k 再大也救不回中心的楔形？', '(b) Why does no k recover the wedges at the center?'),
       T(f"切向週期約 2πr/{wheel['wedges']}，空間頻率隨 1/r 增加，中心沒有有限上界。任何固定輸出網格都無法對所有中心細節滿足 Sampling Theorem。增加 k 可更準確估計面積覆蓋率，但最終網格解析度不變，不能恢復已無法分辨的楔形。",
         f"The tangential period is about 2πr/{wheel['wedges']}, so the spatial frequency grows as 1/r and has no finite upper bound at the center. No fixed output grid can satisfy the Sampling Theorem for all the detail at the center. Raising k estimates the area coverage more accurately, but the resolution of the final grid is unchanged, so wedges that can no longer be resolved cannot be recovered."))
    qa(T('(c) 哪一種濾波器相關？在何處施加？它做到與做不到什麼？', '(c) Which filter is relevant, where is it applied, and what does it achieve?'),
       T('低通濾波器。區塊平均相當於在降採樣前施加 box low-pass filter，壓低高頻假紋理，使不可分辨的交替楔形呈灰色。有限 box 濾波並非理想低通，因此仍可能有殘留 alias；它改善顯示，而不恢復超過最終 Nyquist 限制的資訊。',
         'A low-pass filter. Block averaging amounts to applying a box low-pass filter before downsampling: it suppresses the high-frequency false texture and turns the unresolvable alternating wedges into gray. A finite box filter is not an ideal low-pass filter, so some residual aliasing may remain; it improves the display but does not recover information beyond the final Nyquist limit.'))


# ---------------------------------------------------------------- Part 4 ----

def vectorize_data():
    """Part 4 A/B measures, shared by the Part 4 pages and its extended sections."""
    vec = data('vectorize_summary.json')
    metrics = rows('vectorize_metrics.csv')
    v = dict(vec=vec,
             A={int(r['budget']): r for r in metrics if r['method'] == 'hilbert'},
             B={int(r['harmonics']): r for r in metrics if r['method'] == 'fourier_fit'},
             O={int(r['harmonics']): r for r in metrics if r['method'] == 'oracle'},
             loss=rows('vectorize_b_loss.csv'))
    signal = [float(r['value']) for r in rows('vectorize_signal.csv')]
    # Times the 1D scan crosses 50% coverage: each is a step the spectrum must encode.
    v['crossings'] = sum(((a - 0.5) * (b - 0.5) < 0 for a, b in zip(signal, signal[1:])))
    v['wrong'] = lambda r: int(r['misclassified'])
    v['first_a'] = lambda target: next((n for n in sorted(v['A']) if v['wrong'](v['A'][n]) <= target), None)

    def settled(m):
        """First iteration whose loss is within 0.1% of the stage's final loss."""
        stage = [r for r in v['loss'] if int(r['harmonics']) == m]
        final = float(stage[-1]['loss'])
        return next((int(r['iteration']) for r in stage if float(r['loss']) <= final * 1.001))
    v['settled'] = settled
    pairs = [(2 * m + 1, v['wrong'](v['B'][m]), v['first_a'](v['wrong'](v['B'][m]))) for m in sorted(v['B'])]
    v['pairs'] = [(n, w, a) for n, w, a in pairs if a]
    v['ratios'] = [a / n for n, _, a in v['pairs']]
    return v


def recover_data():
    rec = data('recover_summary.json')
    rec['by'] = {s['truth']: s for s in rec['shapes']}
    return rec


def part4_overview():
    v = vectorize_data()
    vec, A, B, O = (v['vec'], v['A'], v['B'], v['O'])
    rec = recover_data()
    star = rec['by']['star']
    top = max(B)
    tips = ' → '.join(f"{float(B[m]['tip_px']):.2f}" for m in sorted(B))
    m8 = 8
    a_match = v['first_a'](v['wrong'](B[m8]))
    title(T('向量化：還原星形輪廓', "Vectorizing: recovering the star's outline"), T('Part 4 · 選題、問題與結論', 'Part 4 · Topic, question and findings'))
    add(T(f"<b>選題：Vectorizing（第 3 章）。</b>第 3 章說 \"for pixel to vector is more problematical\"（由像素轉回向量比較麻煩）。本題要回答：從本程式自己畫出的點陣圖，能把星形輪廓還原成多準確的向量路徑？確切是什麼讓它無法完全還原？對象是 Part 3 旋轉 5° 的星形，以 k={vec['k']} 覆蓋率畫在 {vec['canvas']}×{vec['canvas']} 的畫布上。",
          f"<b>Topic: Vectorizing (Chapter 3).</b> Chapter 3 says \"for pixel to vector is more problematical\". The question here: starting from the bitmap our own renderer produced, how accurately can the star's outline be recovered as a vector path, and what specifically stops the recovery from being exact? The subject is the Part 3 star rotated by 5°, rendered with k={vec['k']} coverage on a {vec['canvas']}×{vec['canvas']} canvas."))
    story.append(p(T('結論', 'Findings'), subheading))
    bullet(T(f"<b>做得多好。</b>以梯度下降把傅立葉輪廓擬合到 {vec['edge_points']} 個未排序的次像素邊緣點，M={top}（{2 * top + 1} 個係數）時與邊緣點平均相距 {float(B[top]['mean_edge_px']):.3f} px。在角點切開、改以直線擬合並取交點後，星形還原為 {star['lines']} 條直線，角點離真實頂點 {star['corner_error_px']:.3f} px；在星形、圓與 D 字形的場景中，還原的向量填回後只有 {rec['rebuilt_misclassified']} 個像素與原圖不同（共 {rec['canvas'] ** 2:,} 個）。",
             f"<b>How well it worked.</b> A Fourier outline fitted by gradient descent to {vec['edge_points']} unsorted sub-pixel edge points stays on average {float(B[top]['mean_edge_px']):.3f} px from them at M={top} ({2 * top + 1} coefficients). Cut at its corners, refitted as straight lines and intersected, the star comes back as {star['lines']} lines whose corners lie {star['corner_error_px']:.3f} px from the true vertices; in a scene with the star, a circle and a D shape, the refilled vectors differ from the original in only {rec['rebuilt_misclassified']} of {rec['canvas'] ** 2:,} pixels."))
    bullet(T(f"<b>打敗它的：尖角。</b>傅立葉級數的每一項都是平滑的圓周運動，有限項畫不出真正的角：M 每加倍，尖角誤差只緩慢下降（{tips} px），這是 Gibbs 現象在形狀上的樣子。只用傅立葉輪廓時，星形角點離真實頂點 {star['fourier_corner_error_px']:.2f} px。<b>事前不知道的是，圓化只發生在角點附近：</b>在角點之間擬合直線再取交點，角點誤差小約 {star['fourier_corner_error_px'] / star['corner_error_px']:.0f} 倍。",
             f"<b>What defeated it: sharp tips.</b> Every term of a Fourier series is a smooth circular motion, so a finite series cannot draw a true corner: the tip error falls only slowly as M doubles ({tips} px), which is the Gibbs phenomenon seen in a shape. The Fourier outline alone puts the star's corners {star['fourier_corner_error_px']:.2f} px from the true vertices. <b>Not known beforehand: the rounding is local.</b> Lines fitted between the corners and intersected rebuild them about {star['fourier_corner_error_px'] / star['corner_error_px']:.0f} times more accurately."))
    better = [m for m in sorted(B) if float(B[m]['mean_edge_px']) < float(O[m]['mean_edge_px'])]
    bullet(T(f"<b>事前不知道的事：自由參數化在平均誤差上勝過標準截斷。</b>Oracle 是真實輪廓依弧長的傅立葉級數截斷到同樣的 M；M = {', '.join(str(m) for m in better)} 時，擬合輪廓的平均邊緣距離都比 Oracle 小，因為梯度下降可以決定參數 t 在哪裡走快、哪裡走慢。尖角上則仍是 Oracle 較準（M={top}：{float(O[top]['tip_px']):.2f} 對 {float(B[top]['tip_px']):.2f} px）。",
             f"<b>Not known beforehand: free parameterization beats the textbook truncation on average.</b> The Oracle is the true outline's arc-length Fourier series truncated at the same M; for M = {', '.join(str(m) for m in better)} the fitted outline has the smaller mean edge distance, because gradient descent decides where the parameter t runs fast and where slowly; at the tips the Oracle stays better (M={top}: {float(O[top]['tip_px']):.2f} vs. {float(B[top]['tip_px']):.2f} px)."))
    bullet(T(f"<b>描述什麼比怎麼擬合重要。</b>改把像素本身做傅立葉轉換（對照組 A）時，要達到同樣的誤判像素數，需要 {min(v['ratios']):.0f} 到 {max(v['ratios']):.0f} 倍的係數：B 用 {2 * m8 + 1} 個係數誤判 {v['wrong'](B[m8]):,} 像素，A 要 {a_match:,} 個係數才降到 {v['wrong'](A[a_match]):,}。",
             f"<b>What is described matters more than how it is fitted.</b> Transforming the pixels instead (baseline A) needs {min(v['ratios']):.0f} to {max(v['ratios']):.0f} times as many coefficients for the same number of misclassified pixels: B misclassifies {v['wrong'](B[m8]):,} pixels with {2 * m8 + 1} coefficients; A needs {a_match:,} coefficients to get down to {v['wrong'](A[a_match]):,}."))
    story.append(p(T('方法', 'Method'), subheading))
    add(T(f"<b>1. 邊緣點。</b>在水平或垂直相鄰的像素中心之間，覆蓋率跨過 0.5 處以線性內插取次像素邊緣點（星形共 {vec['edge_points']} 個），<b>不排序</b>。",
          f"<b>1. Edge points.</b> Between horizontally or vertically adjacent pixel centers where coverage crosses 0.5, linear interpolation gives a sub-pixel edge point ({vec['edge_points']} for the star). The points are <b>not sorted</b>."))
    add(T(f"<b>2. 傅立葉輪廓（B）。</b>z(t) = Σ c<sub>k</sub> e<super>ikt</super>，|k| 不超過 M，共 N = 2M+1 個複數係數。從邊緣點重心、平均半徑的圓出發，以梯度下降（手算梯度，學習率 {vec['b_rate']}）最小化雙向最近點距離（chamfer）：曲線上的點要靠近邊緣點，每個邊緣點也要被曲線靠近。M 依 {', '.join(str(m) for m in vec['b_harmonics'])} 由粗到細，每階段 {vec['b_iterations_per_stage']} 次，以前一階段的結果為起點。沿邊追蹤最費工的一步是替邊緣點排序；chamfer 損失讓曲線自己找最近的點，因此不需要這一步。",
          f"<b>2. Fourier outline (B).</b> z(t) = Σ c<sub>k</sub> e<super>ikt</super>, |k| at most M, N = 2M+1 complex coefficients. From a circle at the edge points' centroid with their mean radius, gradient descent (hand-derived gradient, rate {vec['b_rate']}) minimizes the two-way nearest-point (chamfer) distance: curve points should be near edge points, and every edge point near the curve. M runs coarse to fine through {', '.join(str(m) for m in vec['b_harmonics'])}, {vec['b_iterations_per_stage']} iterations per stage, each starting from the previous result. The step edge following spends most effort on, putting the edge points in order, is not needed: the loss lets the curve find its nearest points."))
    add(T(f"<b>3. 角點與圖元。</b>沿擬合輪廓，前後各 {rec['window_px']} px 的弦夾角超過 {rec['corner_degrees']}° 視為角點並切段。每段去掉兩端 {rec['trim_px']} px 後，對原始邊緣點依序嘗試直線、圓弧、三次 Bézier，取第一個所有點都在 {rec['tolerance_px']} px 內的；Bézier 不夠準就在最差點再切開。相鄰的直線／圓弧取交點，重建銳利角點。有多個形狀時，先以 DBSCAN（半徑 {rec['eps']} px、至少 {rec['min_points']} 點）把邊緣點分成每條輪廓一群。",
          f"<b>3. Corners and primitives.</b> Along the fitted outline, a point where the chords {rec['window_px']} px before and after it meet at more than {rec['corner_degrees']}° is a corner, and the outline is cut there. Each piece, minus {rec['trim_px']} px at both ends, is fitted to the original edge points as a line, a circular arc or a cubic Bézier, taking the first whose points all lie within {rec['tolerance_px']} px; a Bézier that misses is split again at its worst point. Adjacent lines and arcs are intersected to rebuild the sharp corner. With several shapes, DBSCAN (radius {rec['eps']} px, at least {rec['min_points']} points) first puts the edge points of each outline into one cluster."))
    add(T(f"<b>4. 對照組 A。</b>畫布邊長取 2 的次方，讓希爾伯特曲線以單位步長走遍每個像素一次，沿它讀出亮度得到長 {vec['signal_length']:,} 的 1D 訊號；FFT 後只保留能量最大的 N 個頻率（共軛成對只算一個），反轉換後沿同一條曲線放回，結果仍是點陣圖。FFT、希爾伯特曲線與圖中的線和圓都是自己的程式。",
          f"<b>4. Baseline A.</b> On the power-of-two canvas a Hilbert curve visits every pixel once with unit steps; reading the brightness along it gives a 1D signal of length {vec['signal_length']:,}. An FFT keeps its N strongest frequencies (a conjugate pair counts as one), and the inverse is put back along the same curve; the result is still a bitmap. The FFT, the Hilbert curve and the figures' lines and circles are our own code."))


def part4_results():
    v = vectorize_data()
    A, B, O = (v['A'], v['B'], v['O'])
    wrong = v['wrong']
    top = max(B)
    a_match = v['first_a'](wrong(B[8]))
    title(T('星形上的結果', 'Results on the star'), T('Part 4 · 結果', 'Part 4 · Results'))
    gallery(['vectorize_original.bmp', f'vectorize_a_n{a_match}.bmp', 'vectorize_b_m8.bmp', f'vectorize_b_overlay_m{top}.bmp'],
            [T('原圖', 'Original'), T(f'A，N={a_match:,}：誤判 {wrong(A[a_match]):,}', f'A, N={a_match:,}: {wrong(A[a_match]):,} misclassified'), T(f'B，N=17 填回：誤判 {wrong(B[8]):,}', f'B, N=17 filled back: {wrong(B[8]):,} misclassified'), T(f'B，N={2 * top + 1}：紅線為擬合輪廓', f'B, N={2 * top + 1}: fitted outline in red')], 100)
    story.append(LogChart([(T('A：希爾伯特掃描', 'A: Hilbert scan'), ORANGE, [(math.log2(n), wrong(A[n])) for n in sorted(A)]), (T('B：傅立葉輪廓（梯度下降）', 'B: Fourier outline (gradient descent)'), BLUE, [(math.log2(2 * m + 1), wrong(B[m])) for m in sorted(B)])],
                          height=150, ylabel=T('誤判像素（對數）', 'Misclassified pixels (log)'), xlabel=T('係數數量 N（對數刻度）', 'Number of coefficients N (log scale)'), logy=True))
    table([['N'] + [f'{n:,}' for n in sorted(A)], [T('A：誤判', 'A: misclassified')] + [f'{wrong(A[n]):,}' for n in sorted(A)]], [92] + [(W - 92) / len(A)] * len(A))
    table([['M', 'N', T('誤判', 'Misclassified'), 'RMS', T('平均邊緣距離 / px', 'Mean edge distance / px'), T('尖角 / px', 'Tip / px'), T('內凹角 / px', 'Inner corner / px'), T('Oracle 平均 / 尖角 / px', 'Oracle mean / tip / px')]] +
          [[m, 2 * m + 1, f'{wrong(B[m]):,}', f"{float(B[m]['rms']):.4f}", f"{float(B[m]['mean_edge_px']):.3f}", f"{float(B[m]['tip_px']):.2f}", f"{float(B[m]['corner_px']):.2f}", f"{float(O[m]['mean_edge_px']):.3f} / {float(O[m]['tip_px']):.2f}"] for m in sorted(B)],
          [30, 36, 72, 50, 80, 46, 62, W - 376])
    story.append(p(T('誤判像素：重建結果與原圖各以 50% 覆蓋率判定內外，判定不同的像素數；B 的輪廓先以 Part 2 填色器、k=8 填回。上表：對照組 A；下表：B 依 M 的結果。Oracle：真實星形輪廓依弧長取樣、做傅立葉級數後截斷到同樣的 M；向量化時拿不到這份資料，只作為參考。',
                     'Misclassified pixels: the reconstruction and the original are each classified inside or outside at 50% coverage and the disagreements counted; B is first filled back with the Part 2 fill at k=8. Top: baseline A. Bottom: B by M. Oracle: the true star outline sampled by arc length, expanded as a Fourier series and truncated at the same M; a vectorizer does not have this data, so it serves only as a reference.'), small))
    add(T(f"<b>同樣的係數，B 遠勝 A。</b>以測試過的 N 而言，A 需要 B 的 {min(v['ratios']):.0f} 到 {max(v['ratios']):.0f} 倍係數才不比它差（A 的 N 每級跳 4 倍，實際門檻可能略低）。A 的係數描述每個像素的亮度，但星形的資訊只在邊界上：這條 1D 掃描穿越 50% 覆蓋率 {v['crossings']:,} 次，每次穿越都是訊號中的一個階梯，需要很寬的頻帶；係數不夠時還會出現沿希爾伯特象限排列的方塊。B 只描述邊界這條 1D 曲線。",
          f"<b>With the same coefficients, B is far better than A.</b> Over the N tested, A needs {min(v['ratios']):.0f} to {max(v['ratios']):.0f} times as many coefficients as B to do no worse (A's N grows fourfold per step, so the true threshold may be lower). A's coefficients describe the brightness of every pixel, but the star's information lies on its boundary: the 1D scan crosses 50% coverage {v['crossings']:,} times, each crossing is a step in the signal, and steps need a wide band of frequencies; with too few coefficients, blocks along the Hilbert quadrants appear as well. B describes only the boundary, a 1D curve."))
    add(T(f"<b>平均誤差下降快，尖角誤差下降慢。</b>M 由 4 到 {top}，平均邊緣距離由 {float(B[4]['mean_edge_px']):.3f} 降到 {float(B[top]['mean_edge_px']):.3f} px，尖角誤差只由 {float(B[4]['tip_px']):.2f} 降到 {float(B[top]['tip_px']):.2f} px。chamfer 是平均值，少數尖角點對它影響很小；M=4 時曲線甚至在尖端繞出小圈，以平滑的方式伸進角落。每階段的損失在 {max(v['settled'](m) for m in B)} 次迭代內就落在最終值 0.1% 以內。",
          f"<b>Mean error falls fast, tip error slowly.</b> From M=4 to {top} the mean edge distance falls from {float(B[4]['mean_edge_px']):.3f} to {float(B[top]['mean_edge_px']):.3f} px, but the tip error only from {float(B[4]['tip_px']):.2f} to {float(B[top]['tip_px']):.2f} px. The chamfer loss is an average, and the few tip points have little influence on it; at M=4 the curve even loops at the tips, reaching into the corners smoothly. Every stage's loss is within 0.1% of its final value after at most {max(v['settled'](m) for m in B)} iterations."))


def part4_recovery():
    rec = recover_data()
    shapes = rec['by']
    star, ring, dee = (shapes['star'], shapes['circle'], shapes['d'])
    top = max(rec['harmonics'])
    label = T({'star': '星形', 'circle': '圓', 'd': 'D 字形'}, {'star': 'Star', 'circle': 'Circle', 'd': 'D shape'})

    def number(value, digits):
        return '-' if value is None else f'{value:.{digits}f}'
    title(T('從傅立葉輪廓到直線與圓弧', 'From a Fourier outline to lines and arcs'), T('Part 4 · 角點還原與限制', 'Part 4 · Corner recovery and limits'))
    add(T(f"為確認流程不是只會輸出直線，同一流程也用在 {rec['canvas']}×{rec['canvas']}、k={rec['k']} 的三形狀場景：只有直線的星形、只有曲線的圓，以及直線加半圓的 D 字形。程式不知道有幾個形狀；DBSCAN 分出 {rec['clusters']} 群、噪點 {rec['noise']}。",
          f"To check that the pipeline does not just output lines, it also runs on a {rec['canvas']}×{rec['canvas']} scene at k={rec['k']} with three shapes: a star of straight lines only, a circle of curves only, and a D shape made of a line and a semicircle. The program is not told how many shapes there are; DBSCAN finds {rec['clusters']} clusters with {rec['noise']} noise points."))
    gallery(['recover_fourier.bmp', 'recover_primitives.bmp', 'recover_rebuilt.bmp'],
            [T(f'每群的傅立葉輪廓（M={top}）：角是圓的', f'Fourier outline of each cluster (M={top}): the corners are rounded'), T('還原的圖元：藍＝直線、綠＝圓弧、紫＝Bézier、×＝重建角點', 'Recovered primitives: blue = line, green = arc, purple = Bézier, × = rebuilt corner'), T('還原的向量以 Part 2 填色器填回', 'Recovered vectors filled back with the Part 2 fill')], 112)
    rows_ = [[T('形狀', 'Shape'), T('邊緣點', 'Edge points'), T('還原結果', 'Recovered as'), T('角點（找到 / 實際）', 'Corners (found / true)'), T('角點誤差：還原 / 傅立葉 px', 'Corner error: recovered / Fourier px'), T('圓弧圓心 / 半徑誤差 px', 'Arc center / radius error px'), T('邊緣距離 平均 / 最大 px', 'Edge distance mean / max px')]]
    for key in ('star', 'circle', 'd'):
        s = shapes[key]
        # English pluralizes the primitive name; the count itself is unchanged.
        parts = [f"{s[k]} {n}{T('', '' if s[k] == 1 else 's')}" for k, n in (('lines', T('直線', 'line')), ('arcs', T('圓弧', 'arc')), ('cubics', 'Bézier')) if s[k]]
        rows_.append([label[key], s['edge_points'], T('、', ', ').join(parts), f"{s['sharp_corners']} / {s['true_corners']}", f"{number(s['corner_error_px'], 3)} / {number(s['fourier_corner_error_px'], 3)}", f"{number(s['arc_center_error_px'], 3)} / {number(s['arc_radius_error_px'], 3)}", f"{s['vector_mean_edge_px']:.3f} / {s['vector_max_edge_px']:.3f}"])
    table(rows_, [50, 44, 74, 70, 95, 90, W - 423])
    add(T(f"<b>傅立葉係數負責找，簡單圖元負責描述。</b>星形的角點由傅立葉輪廓的 {star['fourier_corner_error_px']:.2f} px 降到 {star['corner_error_px']:.3f} px；D 字形直邊與半圓的交會處以直線與圓的交點重建，誤差 {dee['corner_error_px']:.3f} px；圓還原為 1 段完整圓弧（圓心誤差 {ring['arc_center_error_px']:.4f} px、半徑誤差 {ring['arc_radius_error_px']:.3f} px）。星形的 {2 * top + 1} 個複數係數最後變成 {star['lines']} 條直線；輸出的 recover_vector.svg 是一般可編輯、可縮放的向量路徑，整個場景填回後 RMS {rec['rebuilt_rms']:.4f}。",
          f"<b>Fourier coefficients do the finding; simple primitives do the describing.</b> The star's corners improve from the Fourier outline's {star['fourier_corner_error_px']:.2f} px to {star['corner_error_px']:.3f} px; where the D shape's straight edge meets its semicircle, the corner is rebuilt from a line-circle intersection with an error of {dee['corner_error_px']:.3f} px; the circle comes back as one full arc (center error {ring['arc_center_error_px']:.4f} px, radius error {ring['arc_radius_error_px']:.3f} px). The star's {2 * top + 1} complex coefficients end up as {star['lines']} straight lines; recover_vector.svg is an ordinary, editable vector path, and the refilled scene differs from the original by an RMS of {rec['rebuilt_rms']:.4f}."))
    story.append(p(T('其他打敗它的因素', 'What else defeated it'), subheading))
    for text in [
        T('<b>覆蓋率剛好 0.5 的像素。</b>k=8 時 32/64 很常見；若以「大於 0.5」判為內部，這些像素兩側都不產生邊緣點，輪廓斷成碎片，DBSCAN 也會把一個形狀分成好幾群。因此規則定為「大於或等於 0.5 為內」。',
          '<b>Pixels with coverage of exactly 0.5.</b> At k=8, 32/64 is common; if only "more than 0.5" counted as inside, they would yield no edge point, the outline would break into fragments and DBSCAN would split one shape into several clusters; hence "0.5 or more is inside".'),
        T(f"<b>淺角與手調門檻。</b>角點門檻 {rec['corner_degrees']}°、修剪長度 {rec['trim_px']} px 與容差 {rec['tolerance_px']} px 都是針對這個尺度選定的；形狀更小或轉角更緩（例如鈍角多邊形）時，轉角量測可能分不出角點與彎曲。",
          f"<b>Shallow corners and hand-picked thresholds.</b> The corner threshold of {rec['corner_degrees']}°, the trim of {rec['trim_px']} px and the tolerance of {rec['tolerance_px']} px were chosen for this scale; with smaller shapes or gentler (obtuse) corners, the turning angle may not tell corners from bends."),
        T('<b>輸入精度。</b>邊緣點來自 k=8 的覆蓋率，次像素精度受 supersampling 限制。',
          '<b>Input precision.</b> The edge points come from k=8 coverage, so their sub-pixel accuracy is bounded by the supersampling.'),
        T('<b>順序被忽略。</b>chamfer 損失只問曲線與點是否接近、不問順序，不是凸函數。星形上由粗到細的排程能收斂，但對又長又曲折的輪廓沒有保證。',
          '<b>Order is ignored.</b> The chamfer loss asks only whether curve and points are close, not in which order, and is not convex; coarse to fine converges on the star, but nothing guarantees it for long, intricate outlines.'),
    ]:
        bullet(text)
    story.append(p(T('參考的已知方法', 'Known methods used'), subheading))
    add(T('以傅立葉級數描述輪廓：Zahn &amp; Roskies（1972）；Kuhl &amp; Giardina（1982）。以最佳化讓傅立葉參數化輪廓貼合邊緣，即 B 的核心：Staib &amp; Duncan（1992）。DBSCAN：Ester 等（1996）。圓擬合：Kåsa（1976）。三次 Bézier 擬合直接採用 Schneider（1990）。直線與圓弧切段：Rosin &amp; West（1995）。整體流程與 Potrace（Selinger 2003）相近；各元件都是已知方法，本實驗把它們串成同一流程並逐步量測。',
          'Fourier descriptions of outlines: Zahn &amp; Roskies (1972); Kuhl &amp; Giardina (1982). Fitting a Fourier-parameterized contour to edges by optimization, the core of B: Staib &amp; Duncan (1992). DBSCAN: Ester et al. (1996). Circle fitting: Kåsa (1976). The cubic Bézier fitting follows Schneider (1990) directly. Segmenting into lines and arcs: Rosin &amp; West (1995). The pipeline as a whole is close to Potrace (Selinger 2003); every component is known, and this experiment chains and measures them.'), small)


TRACE_MAIN = [
    ('BMP', T('diagnostic.bmp；獨立 Pillow 解碼', 'diagnostic.bmp; independent Pillow decoding'), ['diagnostic.bmp']),
    ('P1-1, P1-2', T('curve_controls.csv、curve_formula.csv、curve_depth2.csv、curve_overlay.svg、convergence.csv/.svg、curve_summary.json、curve_answers.md', 'curve_controls.csv, curve_formula.csv, curve_depth2.csv, curve_overlay.svg, convergence.csv/.svg, curve_summary.json, curve_answers.md'), ['curve_*', 'convergence.*']),
    ('P1-3', T('joins_*.csv/.svg、joins_controls.csv、joins_metrics.csv', 'joins_*.csv/.svg, joins_controls.csv, joins_metrics.csv'), ['joins_*']),
    ('P2-1, P2-2', T('star.bmp、star_vertices.csv、star_summary.json', 'star.bmp, star_vertices.csv, star_summary.json'), ['star.bmp', 'star_vertices.csv', 'star_summary.json']),
    ('P2-3', T('ring_same/opposite.bmp、ring_controls.csv、ring_summary.json', 'ring_same/opposite.bmp, ring_controls.csv, ring_summary.json'), ['ring_*']),
    ('P3-I', T('star_k1/2/4/8.bmp 與 *_coverage.csv、aa_summary.json', 'star_k1/2/4/8.bmp and *_coverage.csv, aa_summary.json'), ['star_k*', 'aa_summary.json']),
    ('P3-II', T('row_samples.csv、row_plot.svg、row_selection.json、timing.csv、timing_trials.csv', 'row_samples.csv, row_plot.svg, row_selection.json, timing.csv, timing_trials.csv'), ['row_*', 'timing*']),
    ('P3-III', T('wheel_k1/8.bmp 與 *_coverage.csv、wheel_triangles.csv、wheel_summary.json', 'wheel_k1/8.bmp and *_coverage.csv, wheel_triangles.csv, wheel_summary.json'), ['wheel_*']),
    (T('Part 4 星形', 'Part 4 star'), T('vectorize_*（希爾伯特訊號、A 重建、邊緣點、B 係數與損失、指標）', 'vectorize_* (Hilbert signal, A reconstructions, edge points, B coefficients and loss, metrics)'), ['vectorize_*']),
    (T('Part 4 角點還原', 'Part 4 corner recovery'), T('recover_*（分群、係數、圖元、角點、SVG、填回）', 'recover_* (clusters, coefficients, primitives, corners, SVG, fill-back)'), ['recover_*']),
    (T('整體', 'Overall'), T('run.json、evidence_manifest.json、verification.md、本 PDF', 'run.json, evidence_manifest.json, verification.md, this PDF'), ['run.json', 'evidence_manifest.json']),
]


def audit_main():
    title(T('成果、驗證與追溯', 'Results, verification and traceability'), 'Audit · Parts 1-4')
    table([[T('題目', 'Part'), T('原始資料與成果（output/）', 'Raw data and results (output/)')]] + [[a, b] for a, b, _ in TRACE_MAIN[:-1] + TRACE_EXTENDED + TRACE_MAIN[-1:]], [110, W - 110])
    add(T(f"獨立稽核（scripts/audit.py）通過全部 {len(audit['checks'])} 組檢查。與本報告相關的包括：Bernstein 對照所有公式樣本、逐深度重新計算點到線段誤差、BMP 解碼、覆蓋率量化、取樣列隔離、計時中位數、輪盤三角形；Part 4 則獨立重建希爾伯特曲線、以獨立 FFT 重算每個 A 重建、由保存的係數重算 chamfer 損失與各項誤差，並以暴力法 DBSCAN 重現分群、檢查還原圖元與填回結果；延伸實驗另有檢查：洞的深度與方向（以射線奇偶法）、彩色的調色盤、混色反解與色版分解、名畫的區塊平均、雙邊濾波、分階、間隙計數、色差與 SSIM、高精度的分階、各變體的色差與 SSIM、逐輪廓表的總和與面積項掃描表。附錄標示為人工觀察的內容不在稽核範圍內。幾何容差 2e-8 px，覆蓋率容差 1e-8；編譯期來源指紋與本次 run.json 一致。",
          f"The independent audit (scripts/audit.py) passes all {len(audit['checks'])} groups of checks. Those behind this report include every formula sample against the Bernstein form, point-to-segment errors recomputed at every depth, BMP decoding, coverage quantization, isolation of the sampled row, timing medians and the wheel triangles; for Part 4 the audit rebuilds the Hilbert curve, recomputes every A reconstruction with an independent FFT, recomputes the chamfer losses and all error measures from the saved coefficients, reproduces the clusters with a brute-force DBSCAN, and checks the recovered primitives and the refilled bitmap. The extended experiments have their own checks: for holes, depth and direction with an even-odd ray test; for color, the palette, unmixing and plate decomposition; for the paintings, the box averages, bilateral smoothing, levels, gap counts, color differences and SSIM; for high fidelity, the levels, the color differences and SSIM of every variant, the sums of the per-outline tables and the region-term sweep table. What the appendix marks as manual observation lies outside the audit. The geometric tolerance is 2e-8 px and the coverage tolerance 1e-8; the compile-time source fingerprint matches this run's run.json."))
    add(T('SHA-256 清單（evidence_manifest.json）記錄本次 Rust 建置輸入及輸出檔；報告產生器先核對清單，避免引用被改動的資料，並確認本 PDF 讀取的每個檔案都列在上表。',
          "A SHA-256 manifest (evidence_manifest.json) records this run's Rust build inputs and output files; the report generator checks it first, so that it never cites altered data, and checks that every file this PDF reads is listed in the table above."))
    add(T('程式庫拒絕非有限輸入、尺寸溢位、錯誤倍率與 I/O 失敗。為避免單次請求耗盡資源，均勻細分深度最多 20、密集公式樣本最多 2^20+1；實驗 CLI 的畫布限制為 128..2048，與一般 Image 的尺寸能力分開。',
          'The library rejects non-finite input, size overflow and invalid factors, and reports I/O failures as errors. To keep a single request from exhausting resources, the uniform subdivision depth is at most 20 and the dense formula samples at most 2^20+1; the experiment CLI limits the canvas to 128..2048, separately from the sizes a general Image supports.'))


# ========================================================= Part 4 extended ==

def part4_extended():
    title(T('Part 4 延伸：其他向量化實驗', 'Part 4, extended: further vectorizing experiments'), T('Part 4 延伸 · 導覽', 'Part 4 extended · Overview'))
    add(T('前面三節回答作業的問題：還原星形輪廓，以及它做得多好、被什麼打敗。以下各節把同一條流程延伸到作業沒有要求的情況，屬於額外的實驗：',
          "The three sections above answer the assignment's question: recovering the star's outline, how well that worked and what defeated it. The sections below take the same pipeline into cases the assignment does not ask about, as additional experiments:"))
    for text in [
        T('傅立葉描述的細節：希爾伯特掃描、A 與 B 的完整圖例、本輪（epicycles）與收斂。', 'Fourier descriptions in detail: the Hilbert scan, the full A and B galleries, epicycles and convergence.'),
        T('三形狀場景：分群與輸出的向量檔。', 'The three-shape scene: clustering and the vector file.'),
        T('挖洞：包含關係與方向。', 'Holes: containment and direction.'),
        T('彩色：依調色盤或依油墨色版。', 'Color: by palette or by ink plate.'),
        T('兩幅名畫：方法、各畫結果與比較。', 'Two paintings: methods, each painting, and a comparison.'),
        T('高精度：以梯度下降直接畫出名畫。', 'High fidelity: painting directly with gradient descent.'),
        T('與已知方法的關係；附錄：開發紀錄。', 'Relation to known methods; appendix: development notes.'),
    ]:
        bullet(text)


def supp_fourier():
    v = vectorize_data()
    vec, B = (v['vec'], v['B'])
    title(T('傅立葉描述的細節', 'Fourier descriptions in detail'), T('Part 4 延伸 · A 與 B', 'Part 4 extended · A and B'), new_page=False)
    add(T('前面的 Part 4 只列出結果；這裡補上兩種描述的圖例。A 把整張圖沿希爾伯特曲線讀成 1D 訊號再做傅立葉，N 越大，消去的殘差越多，結果仍是點陣圖。',
          "Part 4 above gives the results; here are the pictures behind them. A reads the whole image along a Hilbert curve as a 1D signal and takes its Fourier transform; a larger N removes more of the residual, and the result is still a bitmap."))
    gallery(['vectorize_original.bmp', 'vectorize_hilbert_order.bmp', 'vectorize_a_n65.bmp', 'vectorize_a_n1025.bmp'], [T('原圖', 'Original'), T('掃描順序：越黑越晚走到', 'Scan order: darker = visited later'), T('A，N=65', 'A, N=65'), T('A，N=1025', 'A, N=1025')], 100)
    gallery([f'vectorize_b_overlay_m{m}.bmp' for m in (4, 8, 16, 64)], [T(f'B，M={m}（N={2 * m + 1}）', f'B, M={m} (N={2 * m + 1})') for m in (4, 8, 16, 64)], 100)
    story.append(p(T('灰色為原圖，紅色為擬合輪廓。M=4 時曲線為了伸進尖角而在尖端繞出小圈。', 'Gray is the original image and red the fitted outline. At M=4 the curve forms small loops at the tips as it reaches into them.'), caption))
    gallery(['vectorize_epicycles_m8.bmp', 'vectorize_b_m8.bmp', 'vectorize_b_m64.bmp'], [T('本輪：M=8 的各項是串接旋轉的圓，大者在前，鏈尾即 z(t)', 'Epicycles: the M=8 terms are chained rotating circles, largest first; the end of the chain is z(t)'), T('B，M=8 填回點陣', 'B, M=8 filled back into a bitmap'), T('B，M=64 填回點陣', 'B, M=64 filled back into a bitmap')], 118)
    table([['M', 'N', T('最終損失', 'Final loss'), T('收斂於第幾次', 'Settled at iteration')]] + [[m, 2 * m + 1, f"{float(B[m]['final_loss']):.3f}", v['settled'](m)] for m in sorted(B)], [60, 60, 120, W - 240])
    story.append(p(T(f"學習率 {vec['b_rate']}，每階段 {vec['b_iterations_per_stage']} 次。收斂：損失首次落在該階段最終值 0.1% 以內的迭代次數。由粗到細很重要：先用少數項定出大致形狀，再逐步加入高頻。",
                     f"Rate {vec['b_rate']}, {vec['b_iterations_per_stage']} iterations per stage. Settled: the first iteration at which the loss is within 0.1% of the stage's final value. Going coarse to fine matters: a few terms fix the rough shape first, and higher frequencies are added step by step."), small))


def supp_shapes():
    rec = recover_data()
    clusters = rows('recover_clusters.csv')
    pts = [(float(r['x']), float(r['y']), r['cluster']) for r in clusters]
    # Largest nearest-neighbor gap within a cluster, and smallest gap between clusters.
    within, between = (0.0, math.inf)
    for i, (x, y, c) in enumerate(pts):
        nearest = math.inf
        for j, (u, w, d) in enumerate(pts):
            if i == j:
                continue
            dist = math.hypot(x - u, y - w)
            if d == c:
                nearest = min(nearest, dist)
            else:
                between = min(between, dist)
        within = max(within, nearest)
    title(T('三形狀場景：分群與向量檔', 'The three-shape scene: clustering and the vector file'), T('Part 4 延伸 · 形狀還原', 'Part 4 extended · Shape recovery'))
    gallery(['recover_scene.bmp', 'recover_clusters.bmp'], [T('原圖', 'Original'), T(f"DBSCAN：{rec['clusters']} 群（三種顏色），噪點 {rec['noise']}", f"DBSCAN: {rec['clusters']} clusters (three colors), {rec['noise']} noise points")], 170)
    add(T(f"<b>DBSCAN 自己數出形狀數。</b>同一條輪廓上，每個邊緣點到最近鄰的距離最多 {within:.2f} px；不同形狀的點彼此至少相距 {between:.1f} px。半徑 {rec['eps']} px 因此剛好把每條輪廓連成一群，沒有噪點。前提是每條輪廓都連續：報告 Part 4 列出了覆蓋率剛好 0.5 的像素如何讓輪廓斷裂。",
          f"<b>DBSCAN counts the shapes by itself.</b> On one outline, every edge point has its nearest neighbor within {within:.2f} px; points of different shapes are at least {between:.1f} px apart. A radius of {rec['eps']} px therefore links each outline into exactly one cluster, with no noise. This presumes every outline is continuous; the report's Part 4 shows how pixels with coverage of exactly 0.5 break outlines apart."))
    svg_line = next((line for line in (OUT / 'recover_vector.svg').read_text(encoding='utf-8').splitlines() if line.strip().startswith('<path') and ' C' not in line), '')
    art('recover_vector.svg')
    if svg_line:
        story.append(p(T('還原的星形（recover_vector.svg 原文）：', 'The recovered star (verbatim from recover_vector.svg):'), small))
        story.append(p(escape(svg_line.strip()), code))
    add(T('輸出的 SVG 只有幾個 L 與 C 指令，可以直接編輯、縮放。圓弧存成 Bézier 時，控制點距離為 4/3·tan(φ/4)·r；四分之一圓正是 Part 2-3 用的 0.5523·r。',
          "The SVG holds only a few L and C commands and can be edited and scaled directly. When an arc is stored as a Bézier, its control points lie 4/3·tan(φ/4)·r away; for a quarter circle this is exactly the 0.5523·r used in Part 2-3."))


def supp_holes():
    lay = data('layers_summary.json')
    holes = lay['holes']
    contour_rows = rows('holes_contours.csv')
    depth = {int(r['contour']): int(r['depth']) for r in contour_rows}
    title(T('挖洞：包含關係與方向', 'Holes: containment and direction'), T('Part 4 延伸 · 洞', 'Part 4 extended · Holes'))
    add(T(f"每條輪廓各自追蹤後，用 Part 2 的繞數判斷它落在哪些輪廓裡面，得到深度：偶數為外框，奇數為洞，洞的父輪廓是包住它的最小輪廓。場景含一個「O」（洞中還有一個小島，深度 0、1、2）、一個方框與一個細圓環，共找到 {holes['contours']} 條輪廓、{len(holes['shapes'])} 個形狀，最深為 {max(depth.values())}。",
          f"After each outline is traced, the Part 2 winding number determines which other outlines it lies inside, which gives its depth: even depths are outer outlines and odd depths are holes, and a hole's parent is the smallest outline enclosing it. The scene contains an \"O\" (with a small island inside its hole: depths 0, 1, 2), a square frame and a thin ring; in total {holes['contours']} outlines and {len(holes['shapes'])} shapes are found, with a maximum depth of {max(depth.values())}."))
    gallery(['holes_scene.bmp', 'holes_unoriented.bmp', 'holes_rebuilt.bmp'], [T('原圖', 'Original'), T('輪廓方向未調整：洞全被填滿', 'Directions not adjusted: every hole is filled'), T('洞與外框反向後填回', 'Filled after reversing holes against their outer outlines')], 150)
    add(T(f"<b>方向決定一切。</b>同樣的輪廓，若照追蹤出來的方向直接填色，洞與外框同向、繞數相加，洞全被填滿：{holes['unoriented_misclassified']:,} 個像素判斷錯誤。把每個洞改成與外框相反的方向（必要時用 Subpath::reversed 反轉，點的位置不動），繞數相消，只剩 {holes['rebuilt_misclassified']} 個像素不同。這與報告 Part 2-3 的結論一致：字型裡的「O」必須讓內圈反向。輸出的 holes_vector.svg 每個形狀是一條 path，外框與洞是同一條 path 的不同子路徑，用 SVG 預設的 nonzero 規則就能正確挖洞。",
          f"<b>Direction decides everything.</b> If the same outlines are filled in the directions in which they were traced, the holes run the same way as their outer outlines, the winding numbers add up, and every hole is filled: {holes['unoriented_misclassified']:,} pixels are wrong. Making each hole run opposite to its outer outline (reversing it with Subpath::reversed where needed, without moving any point) makes the winding numbers cancel, and only {holes['rebuilt_misclassified']} pixels differ. This matches the conclusion of Part 2-3 in the report: the inner outline of an \"O\" in a font must be reversed. In holes_vector.svg each shape is one path whose outer outline and holes are sub-paths of that same path, so SVG's default nonzero rule cuts the holes correctly."))
    add(T('<b>包含判斷用繞數，稽核則用不同的演算法。</b>獨立稽核以射線奇偶法重新判斷每條輪廓的深度、父輪廓與最終方向，結果一致。',
          "<b>Containment uses the winding number; the audit uses a different algorithm.</b> The independent audit re-derives each outline's depth, parent and final direction with the even-odd ray test, and the results agree."))


def supp_color():
    lay = data('layers_summary.json')
    color = lay['color']
    pm, cm = (color['palette_method'], color['channel_method'])
    names = T({'cyan': '青', 'magenta': '洋紅', 'yellow': '黃'}, {'cyan': 'cyan', 'magenta': 'magenta', 'yellow': 'yellow'})
    title(T('彩色：依調色盤，或拆成色版', 'Color: by palette or by ink plate'), T('Part 4 延伸 · 彩色', 'Part 4 extended · Color'))
    add(T(f"彩色場景：白底上有紅色圓盤、疊在其上的藍色星形、綠色圓環與帶洞的黃色方框，邊緣都經過 k=8 抗鋸齒。調色盤由程式自己找：8-bit 顏色至少涵蓋 {color['min_pixels']} 個像素才算平塗色，Lab 色差小於 {color['merge_delta_e']} 的合併，共找到 {len(color['palette'])} 色。",
          f"The color scene: on a white background, a red disc with a blue star on top of it, a green ring and a yellow square frame with a hole, all with edges anti-aliased at k=8. The program finds the palette by itself: an 8-bit color counts as a flat color only if it covers at least {color['min_pixels']} pixels, and colors whose Lab color difference is below {color['merge_delta_e']} are merged; {len(color['palette'])} colors are found in total."))
    add(T('<b>做法一：一種顏色一類。</b>抗鋸齒邊緣的像素是兩種顏色的混合；對每個像素找出最能解釋它的兩色組合與比例 t，把 1-t 與 t 分給這兩色的遮罩（混色反解）。每色的遮罩再走完整的向量化流程，大色塊先畫、小色塊後畫。',
          "<b>Approach 1: one class per color.</b> A pixel on an anti-aliased edge is a mix of two colors; for each pixel we find the pair of colors and the ratio t that best explain it, and assign 1-t and t to the masks of those two colors (unmixing). Each color's mask then goes through the complete vectorizing pipeline; larger color regions are drawn first and smaller ones after."))
    add(T('<b>做法二：拆成 CMY 色版，用 multiply 合回。</b>油墨量 = 1 - RGB。每個色版只有幾種油墨濃度，依濃度切成層層相套的遮罩（介於兩濃度之間取線性比例），各自向量化。每個色版是一個 mix-blend-mode: multiply 群組，三組相乘回到 RGB，如同印刷分色版。',
          '<b>Approach 2: split into CMY ink plates and recombine with multiply.</b> Ink amount = 1 - RGB. Each ink plate has only a few ink densities; it is cut by density into nested masks (pixels between two densities take a linear proportion), and each layer is vectorized separately. Each ink plate is a group with mix-blend-mode: multiply, and multiplying the three groups together gives back RGB, just like the color separations of color printing.'))
    gallery(['color_scene.bmp', 'color_palette_rebuilt.bmp', 'color_channels_rebuilt.bmp'], [T('原圖', 'Original'), T('做法一：調色盤', 'Approach 1: palette'), T('做法二：CMY 色版', 'Approach 2: CMY ink plates')], 100)
    gallery(['color_plate_cyan.bmp', 'color_plate_magenta.bmp', 'color_plate_yellow.bmp', 'color_palette_error.bmp', 'color_channels_error.bmp'], [T('青色版', 'Cyan plate'), T('洋紅色版', 'Magenta plate'), T('黃色版', 'Yellow plate'), T('做法一誤差（越黑越大）', 'Approach 1 error (darker = larger)'), T('做法二誤差', 'Approach 2 error')], 74)
    levels = T('、', ', ').join((f"{names[k]} {len(v) - 1}" for k, v in cm['levels'].items()))
    table([['', T('做法一：調色盤', 'Approach 1: palette'), T('做法二：CMY 色版', 'Approach 2: CMY ink plates')], [T('分出的單位', 'Units separated'), T(f"{len(color['palette'])} 色（含背景）", f"{len(color['palette'])} colors (including background)"), T(f"色版濃度層：{levels}", f"Ink plate density levels: {levels}")], [T('向量化前（只用遮罩）明顯色差像素', 'Visibly wrong pixels before vectorizing (masks only)'), f"{pm['masks_visible']}", T(f"{cm['masks_visible']}（分層無損）", f"{cm['masks_visible']} (layering is lossless)")], [T('輪廓 / path 數', 'Outlines / paths'), f"{pm['contours']} / {pm['paths']}", f"{cm['contours']} / {cm['paths']}"], [T('SVG 大小', 'SVG size'), f"{pm['svg_bytes']:,} bytes", f"{cm['svg_bytes']:,} bytes"], [T('向量化後 RMS', 'RMS after vectorizing'), f"{pm['rms']:.4f}", f"{cm['rms']:.4f}"], [T(f"明顯色差像素（任一通道差 > {color['visible']}）", f"Visibly wrong pixels (any channel differs by > {color['visible']})"), f"{pm['visible']}", f"{cm['visible']}"]], [170, 150, W - 320])
    add(T(f"<b>做法一：向量化幾乎不增加誤差。</b>明顯色差 {pm['visible']} 像素，與還沒向量化、只用遮罩時的 {pm['masks_visible']} 相同；剩下的誤差都在三種顏色交會的少數像素，因為混色反解假設每個像素最多只混兩色。",
          f"<b>Approach 1: vectorizing adds almost no error.</b> There are {pm['visible']} visibly wrong pixels, the same as the {pm['masks_visible']} obtained from the masks alone before vectorizing; the remaining errors all lie in the few pixels where three colors meet, because unmixing assumes that each pixel mixes at most two colors."))
    add(T(f"<b>做法二：分層無損，向量化後卻多出套色誤差。</b>只用遮罩合回時誤差為 0，但向量化後明顯色差 {cm['visible']} 像素，是做法一的 {cm['visible'] / max(pm['visible'], 1):.1f} 倍，SVG 也大 {cm['svg_bytes'] / pm['svg_bytes']:.1f} 倍。同一條顏色邊界出現在好幾個色版裡，每次都各自擬合、各有誤差；疊回去時邊界錯開，就像印刷時色版沒有對齊。最明顯的地方是藍色星形尖角穿出紅色圓盤處：洋紅油墨在紅、藍兩色都有，那裡的邊界只存在於青與黃兩個色版，兩者分別擬合後錯開，誤差圖上留下一小片色斑。",
          f"<b>Approach 2: the layering is lossless, but vectorizing adds misregistration.</b> Recombining the masks alone gives zero error, but after vectorizing there are {cm['visible']} visibly wrong pixels, {cm['visible'] / max(pm['visible'], 1):.1f} times as many as with Approach 1, and the SVG is {cm['svg_bytes'] / pm['svg_bytes']:.1f} times as large. The same color boundary appears in several ink plates and is fitted separately in each, each time with its own error; when the plates are stacked again the boundaries no longer line up, just like plates out of register in printing. The most visible spot is where the tips of the blue star stick out of the red disc: magenta ink is present in both red and blue, so the boundary there exists only in the cyan and yellow plates, which are fitted separately and drift apart, leaving a small colored blotch in the error image."))
    add(T('<b>兩種做法的取捨。</b>平塗色的圖適合做法一：每個色塊一條 path，最省也最準。做法二的價值在於不需要找出顏色有幾種，且每個色版只是單色濃度，適合輸出給網版、雷射雕刻等一次一色的設備；代價是必須處理套色。這也是印刷業為何要有「補漏白」（trapping）：讓相鄰色塊稍微重疊來遮住套色誤差。',
          '<b>Trade-offs between the two.</b> Images with flat colors suit Approach 1: one path per color region, the most compact and the most accurate. The value of Approach 2 is that it does not need to know how many colors there are, and each ink plate holds only the densities of a single ink, which suits devices that work one color at a time, such as screen printing and laser engraving; the price is having to deal with registration. This is also why the printing industry uses trapping: adjacent color regions are made to overlap slightly to hide misregistration.'))


PAINT_METHODS = ['k24', 'k24_stack', 'k24_stack_clean', 'cmy888', 'lab888', 'lab1266', 'lab1266_clean']


def paint_label(suffix):
    """Short bilingual name of one painting run, from its stem suffix."""
    labels = {
        'k8': T('調色盤 8 色（剪貼）', 'Palette 8, cut-out'),
        'k16': T('調色盤 16 色（剪貼）', 'Palette 16, cut-out'),
        'k24': T('調色盤 24 色（剪貼）', 'Palette 24, cut-out'),
        'k24_stack': T('調色盤 24 色（堆疊）', 'Palette 24, stacked'),
        'k24_stack_clean': T('調色盤 24 色（堆疊＋清理）', 'Palette 24, stacked, cleaned'),
        'cmy888': T('CMY 8+8+8', 'CMY 8+8+8'),
        'lab888': T('Lab 8+8+8', 'Lab 8+8+8'),
        'lab1266': T('Lab 12+6+6', 'Lab 12+6+6'),
        'lab1266_clean': T('Lab 12+6+6＋清理', 'Lab 12+6+6, cleaned'),
    }
    return labels[suffix]


def painting_table(runs, painting):
    header = [T('做法', 'Method'), T('輪廓', 'Outlines'), T('退回折線', 'Fallbacks'), T('SVG KB', 'SVG KB'), T('平均 ΔE：量化 → 向量', 'Mean ΔE: quantized → vector'), T('SSIM：量化 → 向量', 'SSIM: quantized → vector'), T('背景露出', 'Background exposed')]
    body_rows = []
    for suffix in ['k8', 'k16'] + PAINT_METHODS:
        r = runs[f'paint_{painting}_{suffix}']
        gap = '-' if r['background_exposed'] is None else f"{r['background_exposed']:,}"
        body_rows.append([paint_label(suffix), f"{r['loops']:,}", f"{r['fallbacks']:,}", f"{r['svg_bytes'] / 1000:.0f}", f"{r['quant_mean_delta_e']:.2f} → {r['vector_mean_delta_e']:.2f}", f"{r['quant_ssim']:.3f} → {r['vector_ssim']:.3f}", gap])
    table([header] + body_rows, [128, 48, 50, 44, 86, 86, W - 442], keep=True)


def picture_grid(painting, suffixes, columns, aspect):
    """Rust BMPs of one painting in a grid, captions underneath."""
    cell = (W - 6 * columns) / columns
    height = cell * aspect
    names = [('input', T('輸入（縮小後）', 'Input (shrunk)'))] + [(s, paint_label(s)) for s in suffixes]
    cells = []
    for start in range(0, len(names), columns):
        chunk = names[start:start + columns]
        cells.append([Image(art(f'paint_{painting}_input.bmp' if s == 'input' else f'paint_{painting}_{s}_vector.bmp'), width=cell, height=height) for s, _ in chunk])
        cells.append([p(label, caption) for _, label in chunk])
    grid(cells, columns)


def crop_row(painting, box, suffixes, zh, en):
    """Enlarged crops of the same region: input, then each method."""
    width = (W - 6 * (len(suffixes) + 1)) / (len(suffixes) + 1)
    names = [(f'paint_{painting}_input.bmp', T(f'{zh}：輸入', f'{en}: input'))] + [(f'paint_{painting}_{s}_vector.bmp', paint_label(s)) for s in suffixes]
    grid([[Crop(n, box, width) for n, _ in names], [p(c, caption) for _, c in names]], len(names))


def supp_paintings():
    """Paintings, grouped by artwork, then compared."""
    require(exists('paintings_summary.json'), 'The extended Part 4 needs the painting run: decode_paintings.py, then the program with all --paintings')
    ps = data('paintings_summary.json')
    runs = {r['stem']: r for r in ps['runs']}
    sizes = ps['inputs']
    gleizes, adam = (sizes['gleizes'], sizes['adam'])
    loops = sum((r['loops'] for r in ps['runs']))
    clusters = sum((r['dbscan_clusters'] for r in ps['runs']))
    R = lambda painting, suffix: runs[f'paint_{painting}_{suffix}']
    pct = lambda before, after: (after - before) / before * 100

    # ---- Methods and measures.
    title(T('名畫實測：方法與指標', 'Two paintings: methods and measures'), T('Part 4 延伸 · 名畫', 'Part 4 extended · Paintings'))
    add(T(f"測試圖形彼此相距 10 px 以上；真正的畫作裡，同色區塊互相貼近、交錯。這裡用兩幅風格相反的畫測試整條流程：Gleizes《女人與孩子》（1920）是由平塗色塊、筆直硬邊與細線構成的立體派畫作，米開朗基羅《創造亞當》是帶漸層與裂紋的濕壁畫。Pillow 只負責把 JPEG 解碼成 BMP；之後的裁切、以 Part 3 的區塊平均縮小（得到 {gleizes['size'][0]}×{gleizes['size'][1]} 與 {adam['size'][0]}×{adam['size'][1]}）與所有處理都由 Rust 程式完成。每幅畫跑同一組九種做法。",
          f"The test shapes were more than 10 px apart; in a real painting, regions of one color touch and interleave. The whole pipeline is tested here on two paintings of opposite styles: Albert Gleizes's Woman and Child (1920), a Cubist picture of flat color planes, straight hard edges and thin lines, and Michelangelo's The Creation of Adam, a fresco with gradients and cracks. Pillow only decodes the JPEG into BMP; cropping, shrinking with the Part 3 box average (to {gleizes['size'][0]}×{gleizes['size'][1]} and {adam['size'][0]}×{adam['size'][1]}) and every later step run in the Rust program. Each painting goes through the same nine methods."))
    story.append(p(T('分層的做法', 'How the picture is cut into layers'), subheading))
    for text in [
        T('<b>調色盤（剪貼）</b>：在 Lab 色彩空間做 k-means（K = 8、16、24），每色一個遮罩，疊在面積最大的顏色上。相鄰兩色的共同邊界各自擬合一次，兩條曲線之間可能露出底色。',
          '<b>Palette, cut-out</b>: k-means in Lab color space (K = 8, 16, 24), one mask per color, laid on the largest color. The edge two colors share is fitted twice, once per side, and the background can show between the two curves.'),
        T('<b>調色盤（堆疊）</b>：顏色由亮到暗排序，第 i 層涵蓋「第 i 色及所有更暗的顏色」，每層都延伸到上層底下，因此不會留縫。',
          '<b>Palette, stacked</b>: colors ordered light to dark; layer i covers color i and every darker one, so each layer runs under the layers above and no gap can open.'),
        T('<b>CMY 8+8+8</b>：青、洋紅、黃三個油墨色版，各用一維 k-means 分 8 階，層層相套，三版相乘（SVG 的 multiply 混合模式）。',
          '<b>CMY 8+8+8</b>: cyan, magenta and yellow ink plates, each cut by 1-D k-means into 8 nested levels and multiplied together (SVG multiply blending).'),
        T('<b>Lab 8+8+8、12+6+6</b>：亮度 L* 與兩個色差 a*、b* 各自分階後轉回 RGB。12+6+6 把較多階分給肉眼較敏感的亮度。SVG 的混合模式只能在 RGB 中運算，無法直接重組 Lab；另一份以 SVG 濾鏡重組、可直接用瀏覽器開啟的版本見比較頁的「Lab 在瀏覽器中」。',
          '<b>Lab 8+8+8 and 12+6+6</b>: lightness L* and the two color-difference channels a*, b* are levelled separately and converted back to RGB. 12+6+6 gives more levels to lightness, to which the eye is more sensitive. SVG blend modes only compute in RGB and cannot recombine Lab directly; a second version rebuilt with an SVG filter, which opens directly in a browser, is described under "Lab in a browser" on the comparison page.'),
        T(f"<b>清理</b>：先以自寫的雙邊濾波（半徑 {ps['bilateral']['radius']} px、ΔE 權重 {ps['bilateral']['sigma_delta_e']}）抹平裂紋與斑駁、保留邊緣，再把小於 {ps['clean_area']} 像素的區塊併入相近的鄰居（調色盤 ΔE &lt; {ps['clean_delta_e']:.0f}，色版只能併入相鄰的階）。",
          f"<b>Cleanup</b>: a self-written bilateral filter (radius {ps['bilateral']['radius']} px, ΔE weight {ps['bilateral']['sigma_delta_e']}) first smooths cracks and mottling while keeping edges; then regions under {ps['clean_area']} pixels merge into a similar neighbor (palette ΔE &lt; {ps['clean_delta_e']:.0f}; a plate only into the adjacent level)."),
    ]:
        bullet(text)
    add(T(f"<b>共同流程。</b>小於 {ps['min_area']} 像素的區塊先併入鄰居；每個遮罩沿像素邊界追蹤出<b>有順序</b>的輪廓（若改把未排序的邊緣點交給 DBSCAN，{loops:,} 條輪廓只被分成 {clusters:,} 群，相鄰輪廓大量被併在一起），以 DFT 取得傅立葉係數後，與 Part 4 相同地找角點、擬合直線／圓弧／Bézier，並如前節挖洞；包含關係在像素邊界輪廓上判斷。還原輪廓與像素輪廓的面積差超過 {ps['fallback_area'] * 100:.0f}% 時改用 Douglas-Peucker 折線。最後以 Part 2 填色器（k={ps['fill_k']}）填回。",
          f"<b>Common pipeline.</b> Regions under {ps['min_area']} pixels first merge into a neighbor. Every mask is traced along the pixel cracks into <b>ordered</b> outlines (given the unordered edge points instead, DBSCAN puts {loops:,} outlines into only {clusters:,} clusters, merging many neighbors). A DFT gives the Fourier coefficients; corners and line/arc/Bézier fitting work as in Part 4 and holes as in the holes section, with containment decided on the pixel-crack outlines. When a recovered outline's area differs from the traced one's by more than {ps['fallback_area'] * 100:.0f}%, a Douglas-Peucker polygon is used instead. The Part 2 rasterizer (k={ps['fill_k']}) refills the result."))
    add(T('<b>三個指標，都與未平滑的原圖比較。</b>平均 ΔE：每個像素的色差（CIE76）。SSIM：亮度 L* 在 8×8 區塊上的結構相似度，1 為完全相同，看明暗結構是否保留。背景露出：調色盤做法中，本該是其他顏色、填回後卻最接近底色的像素數，用來量測色塊之間的空隙。',
          '<b>Three measures, all against the unsmoothed original.</b> Mean ΔE: the color difference of each pixel (CIE76). SSIM: structural similarity of lightness L* over 8×8 blocks, 1 = identical, showing whether light and shade survive. Background exposed: in the palette methods, pixels of another color that come back closest to the canvas color, measuring the gaps between color regions.'))

    # ---- Woman and Child.
    title(T('Gleizes《女人與孩子》：平塗色塊與細線', "Gleizes's Woman and Child: flat color planes and thin lines"), T('Part 4 延伸 · 《女人與孩子》', 'Part 4 extended · Woman and Child'))
    picture_grid('gleizes', ['k24', 'k24_stack', 'cmy888', 'lab1266', 'lab1266_clean'], 3, gleizes['size'][1] / gleizes['size'][0])
    crop_row('gleizes', (200, 80, 340, 220), ['k24_stack', 'cmy888', 'lab1266'], '臉部與細線', 'Face and thin lines')
    painting_table(runs, 'gleizes')
    cut, stack, cmy, lab12, lab8 = (R('gleizes', 'k24'), R('gleizes', 'k24_stack'), R('gleizes', 'cmy888'), R('gleizes', 'lab1266'), R('gleizes', 'lab888'))
    add(T(f"<b>堆疊消除了空隙。</b>剪貼式的 24 色有 {cut['background_exposed']:,} 個像素露出底色，堆疊後只剩 {stack['background_exposed']:,}（{pct(cut['background_exposed'], stack['background_exposed']):.0f}%），平均 ΔE 由 {cut['vector_mean_delta_e']:.2f} 降到 {stack['vector_mean_delta_e']:.2f}，SSIM 由 {cut['vector_ssim']:.3f} 升到 {stack['vector_ssim']:.3f}；代價是每層都延伸到上層底下，SVG 由 {cut['svg_bytes'] / 1000:.0f} 增為 {stack['svg_bytes'] / 1000:.0f} KB。",
          f"<b>Stacking closes the gaps.</b> Cut-out with 24 colors leaves {cut['background_exposed']:,} pixels showing the background; stacked, only {stack['background_exposed']:,} ({pct(cut['background_exposed'], stack['background_exposed']):.0f}%). Mean ΔE falls from {cut['vector_mean_delta_e']:.2f} to {stack['vector_mean_delta_e']:.2f} and SSIM rises from {cut['vector_ssim']:.3f} to {stack['vector_ssim']:.3f}. The price: every layer runs under the ones above, and the SVG grows from {cut['svg_bytes'] / 1000:.0f} to {stack['svg_bytes'] / 1000:.0f} KB."))
    best = min(PAINT_METHODS, key=lambda m: R('gleizes', m)['vector_mean_delta_e'])
    add(T(f"<b>平塗色塊的畫，調色盤保得住細線。</b>臉部的白色細線與眼睛在堆疊調色盤中保持完整；Lab 12+6+6 的白色細線斷成幾段；CMY 則沿著細線出現彩虹色的鑲邊，因為三個色版的邊緣各自擬合、彼此對不齊。這幅畫上平均 ΔE 最低的是{paint_label(best)}（{R('gleizes', best)['vector_mean_delta_e']:.2f}；堆疊調色盤 {stack['vector_mean_delta_e']:.2f}、Lab 8+8+8 {lab8['vector_mean_delta_e']:.2f}、Lab 12+6+6 {lab12['vector_mean_delta_e']:.2f}），CMY 的 SSIM 最高（{cmy['vector_ssim']:.3f}）。",
          f"<b>For flat color planes, the palette keeps the thin lines.</b> The thin white line along the face and the eye stay whole with the stacked palette; with Lab 12+6+6 the white line breaks into pieces; CMY draws rainbow fringes along the thin lines, because the edges of the three plates are fitted separately and do not coincide. On this picture the lowest mean ΔE belongs to {paint_label(best)} ({R('gleizes', best)['vector_mean_delta_e']:.2f}; stacked palette {stack['vector_mean_delta_e']:.2f}, Lab 8+8+8 {lab8['vector_mean_delta_e']:.2f}, Lab 12+6+6 {lab12['vector_mean_delta_e']:.2f}), and CMY has the highest SSIM ({cmy['vector_ssim']:.3f})."))

    # ---- The Creation of Adam.
    title(T('米開朗基羅《創造亞當》：漸層與裂紋', "Michelangelo's The Creation of Adam: gradients and cracks"), T('Part 4 延伸 · 《創造亞當》', 'Part 4 extended · The Creation of Adam'))
    picture_grid('adam', ['k24', 'k24_stack', 'cmy888', 'lab1266', 'lab1266_clean'], 2, adam['size'][1] / adam['size'][0])
    crop_row('adam', (285, 150, 395, 215), ['k24_stack', 'cmy888', 'lab1266'], '手指', 'Fingers')
    painting_table(runs, 'adam')
    cut, stack, cmy, lab12, cleaned = (R('adam', 'k24'), R('adam', 'k24_stack'), R('adam', 'cmy888'), R('adam', 'lab1266'), R('adam', 'lab1266_clean'))
    add(T(f"<b>調色盤把漸層切成色帶，也吃掉了手指。</b>膚色與天空的連續明暗只能用 K 個平塗色近似；淺色皮膚與灰泥牆面色差很小，k-means 依像素數量分配顏色，大面積牆面佔走了名額，兩根食指受光的一側被分到和牆面同一色。剪貼式還在色塊之間露出底色（{cut['background_exposed']:,} 像素），堆疊後降為 {stack['background_exposed']:,}（{pct(cut['background_exposed'], stack['background_exposed']):.0f}%），SSIM 由 {cut['vector_ssim']:.3f} 升到 {stack['vector_ssim']:.3f}。",
          f"<b>The palette turns gradients into bands and swallows the fingers.</b> The continuous shading of skin and sky can only be approximated by K flat colors; pale skin and the plaster wall differ very little, and k-means allots colors by pixel count, so the large wall takes the palette slots and the lit side of both index fingers gets the wall's color. Cut-out also shows the background between regions ({cut['background_exposed']:,} pixels); stacked, {stack['background_exposed']:,} ({pct(cut['background_exposed'], stack['background_exposed']):.0f}%), and SSIM rises from {cut['vector_ssim']:.3f} to {stack['vector_ssim']:.3f}."))
    add(T(f"<b>Lab 救回了手指，也最接近原畫。</b>Lab 12+6+6 的平均 ΔE 只有 {lab12['vector_mean_delta_e']:.2f}，是所有做法中最低（堆疊調色盤 {stack['vector_mean_delta_e']:.2f}、CMY {cmy['vector_mean_delta_e']:.2f}），SSIM {lab12['vector_ssim']:.3f}。膚色比灰泥偏紅，這點差異由 a* 色版獨立分出一階，不必和整面牆搶調色盤名額，兩根食指的形狀因此保留下來。CMY 的 SSIM 略高（{cmy['vector_ssim']:.3f}），但近中性的灰泥被三個色版分別量化、分別擬合，疊回來出現粉紅、淡綠的雜色，手指周圍也有彩色鑲邊；Lab 的灰泥只落在亮度版，沒有這個問題。",
          f"<b>Lab brings the fingers back and comes closest to the painting.</b> Lab 12+6+6 has a mean ΔE of only {lab12['vector_mean_delta_e']:.2f}, the lowest of all (stacked palette {stack['vector_mean_delta_e']:.2f}, CMY {cmy['vector_mean_delta_e']:.2f}), with SSIM {lab12['vector_ssim']:.3f}. Skin is redder than plaster; the a* plate gives that difference its own level instead of competing with the whole wall for palette slots, so both index fingers keep their shape. CMY scores a slightly higher SSIM ({cmy['vector_ssim']:.3f}), but the near-neutral plaster, quantized and fitted separately in three plates, comes back with pink and pale-green specks, and the fingers get colored fringes; in Lab the plaster sits in the lightness plate only, so this does not happen."))
    add(T(f"<b>清理是取捨。</b>Lab 12+6+6 清理後輪廓由 {lab12['loops']:,} 降到 {cleaned['loops']:,}（{pct(lab12['loops'], cleaned['loops']):.0f}%），SVG 由 {lab12['svg_bytes'] / 1000:.0f} 降到 {cleaned['svg_bytes'] / 1000:.0f} KB，但平均 ΔE 由 {lab12['vector_mean_delta_e']:.2f} 增為 {cleaned['vector_mean_delta_e']:.2f}、SSIM 由 {lab12['vector_ssim']:.3f} 降為 {cleaned['vector_ssim']:.3f}：裂紋和斑駁少了，細節也少了。",
          f"<b>Cleanup is a trade-off.</b> Cleaning Lab 12+6+6 cuts outlines from {lab12['loops']:,} to {cleaned['loops']:,} ({pct(lab12['loops'], cleaned['loops']):.0f}%) and the SVG from {lab12['svg_bytes'] / 1000:.0f} to {cleaned['svg_bytes'] / 1000:.0f} KB, but mean ΔE rises from {lab12['vector_mean_delta_e']:.2f} to {cleaned['vector_mean_delta_e']:.2f} and SSIM falls from {lab12['vector_ssim']:.3f} to {cleaned['vector_ssim']:.3f}: fewer cracks and specks, and less detail."))

    # ---- Comparison. The per-painting tables above hold every number; this page only compares.
    title(T('兩幅畫的比較', 'Comparing the two paintings'), T('Part 4 延伸 · 名畫比較', 'Part 4 extended · Painting comparison'))
    d_cut, d_stack, a_cut, a_stack = (R('gleizes', 'k24'), R('gleizes', 'k24_stack'), R('adam', 'k24'), R('adam', 'k24_stack'))
    add(T(f"<b>堆疊在兩幅畫都比剪貼好。</b>背景露出分別減少 {-pct(d_cut['background_exposed'], d_stack['background_exposed']):.0f}% 與 {-pct(a_cut['background_exposed'], a_stack['background_exposed']):.0f}%，ΔE 與 SSIM 也同時改善；只要接受較大的 SVG，就沒有理由用剪貼式。",
          f"<b>Stacking beats cut-out on both paintings.</b> Background exposure falls by {-pct(d_cut['background_exposed'], d_stack['background_exposed']):.0f}% and {-pct(a_cut['background_exposed'], a_stack['background_exposed']):.0f}%, and ΔE and SSIM improve together; unless the larger SVG matters, there is no reason to cut out."))
    g_lab, g_cmy, a_lab, a_cmy = (R('gleizes', 'lab1266'), R('gleizes', 'cmy888'), R('adam', 'lab1266'), R('adam', 'cmy888'))
    add(T(f"<b>同樣 24 層，Lab 的色差在兩幅畫都比 CMY 低，在漸層為主的濕壁畫上差得最多。</b>《女人與孩子》{g_lab['vector_mean_delta_e']:.2f} 對 {g_cmy['vector_mean_delta_e']:.2f}，《創造亞當》{a_lab['vector_mean_delta_e']:.2f} 對 {a_cmy['vector_mean_delta_e']:.2f}，也沒有 CMY 的套色雜色與鑲邊；CMY 的 SSIM 則較高（{g_cmy['vector_ssim']:.3f} 與 {a_cmy['vector_ssim']:.3f}）。Lab 是依人眼設計的色彩空間：灰色只落在亮度版，色差版則專門處理顏色，三版各自分階、各自擬合的誤差不會在中性色上疊出雜色。",
          f"<b>With the same 24 layers, Lab's color error is below CMY's on both paintings, by far the most on the gradient-heavy fresco.</b> Woman and Child: {g_lab['vector_mean_delta_e']:.2f} against {g_cmy['vector_mean_delta_e']:.2f}; The Creation of Adam: {a_lab['vector_mean_delta_e']:.2f} against {a_cmy['vector_mean_delta_e']:.2f}, without CMY's misregistration specks and fringes, though CMY's SSIM is higher ({g_cmy['vector_ssim']:.3f} and {a_cmy['vector_ssim']:.3f}). Lab is built around human vision: grays fall in the lightness plate only and the difference plates carry just the color, so errors from levelling and fitting each plate separately do not pile up as color noise on neutral areas."))
    more_light = all(R(n, 'lab1266')['vector_ssim'] > R(n, 'lab888')['vector_ssim'] for n in ('gleizes', 'adam'))
    add(T('<b>最適合的做法取決於畫面。</b>平塗色塊、帶細線的《女人與孩子》，堆疊調色盤保住了細線，色差也與最好的做法相當；漸層為主的《創造亞當》，分通道做法明顯勝出，Lab 12+6+6 同時拿到最低色差並救回手指。'
          + ('兩幅畫的共同點是：把較多階數分給亮度（12+6+6）比平均分配（8+8+8）的 SSIM 更高，保留更多明暗結構。' if more_light else ''),
          '<b>The best method depends on the picture.</b> For Woman and Child, with its flat color planes and thin lines, the stacked palette keeps the lines with a color error close to the best; for the gradient-heavy Creation of Adam, the channel methods clearly win, and Lab 12+6+6 both has the lowest color error and brings the fingers back.'
          + (' Common to both: giving more levels to lightness (12+6+6) gives a higher SSIM than an even split (8+8+8), keeping more of the light-and-shade structure.' if more_light else '')))
    add(T('<b>沒有全面最好的做法。</b>ΔE 衡量每個像素的顏色，SSIM 衡量明暗結構，SVG 大小衡量檔案成本；三者沒有一個做法同時最好。清理在兩幅畫上都以些許細節換取更少的輪廓與更小的檔案。',
          '<b>No method wins on every count.</b> ΔE measures the color of each pixel, SSIM the light-and-shade structure, SVG size the file cost; no single method is best on all three. Cleanup, on both paintings, trades some detail for fewer outlines and smaller files.'))
    dl, al = (R('gleizes', 'lab1266'), R('adam', 'lab1266'))
    story.append(p(T('Lab 在瀏覽器中', 'Lab in a browser'), subheading))
    add(T(f"另一份 *_browser.svg 把三個版分別畫進紅、綠、藍通道（L*/100、(a*+128)/255、(b*+128)/255），以 screen 合成，再用一條 SVG 濾鏡重組：feColorMatrix 算出 f(X)、f(Y)、f(Z)，feComponentTransfer 取立方得到 XYZ，第二個 feColorMatrix 轉成線性 RGB，最後以 gamma 曲線轉回 sRGB。檔案只比原版多約 1 KB。濾鏡沒有分段函數，Lab 與 sRGB 在接近黑色處的短線性段被省略；逐像素模擬這條濾鏡鏈，與精確轉換相比，Lab 12+6+6 的平均 ΔE 在《女人與孩子》為 {dl['filter_mean_delta_e']:.3f}（最大 {dl['filter_max_delta_e']:.1f}）、《創造亞當》為 {al['filter_mean_delta_e']:.3f}（最大 {al['filter_max_delta_e']:.1f}）。若瀏覽器在濾鏡步驟之間只保留 8 位元，模擬的平均 ΔE 升為 {dl['filter_8bit_mean_delta_e']:.2f} 與 {al['filter_8bit_mean_delta_e']:.2f}。若改用 JPEG 的 YCbCr，它與 RGB 是線性關係，只要一個 feColorMatrix 就能精確重組。在實際瀏覽器中的人工檢查見附錄。",
          f"A second file, *_browser.svg, draws the three plates into the red, green and blue channels (L*/100, (a*+128)/255, (b*+128)/255), combines them with screen, and rebuilds RGB with one SVG filter: feColorMatrix computes f(X), f(Y), f(Z), feComponentTransfer cubes them into XYZ, a second feColorMatrix converts to linear RGB, and a gamma curve returns to sRGB. The file is only about 1 KB larger. Filters have no piecewise functions, so the short linear pieces of the Lab and sRGB curves near black are left out; simulating the chain pixel by pixel against the exact conversion, Lab 12+6+6 has mean ΔE {dl['filter_mean_delta_e']:.3f} (max {dl['filter_max_delta_e']:.1f}) on Woman and Child and {al['filter_mean_delta_e']:.3f} (max {al['filter_max_delta_e']:.1f}) on The Creation of Adam. If a browser keeps only 8 bits between filter steps, the simulated mean ΔE rises to {dl['filter_8bit_mean_delta_e']:.2f} and {al['filter_8bit_mean_delta_e']:.2f}. With the YCbCr of JPEG, which is linear in RGB, a single feColorMatrix would rebuild it exactly. A manual check in a real browser is in the appendix."))
    add(T('圖片來源：米開朗基羅《創造亞當》（約 1512，公有領域）；Albert Gleizes《女人與孩子》（Femme et enfant，1920；作者 1953 年逝世，在保護期為死後 70 年的國家自 2024 年起屬公有領域，在美國亦因發表於 1929 年以前而屬公有領域）。',
          "Image sources: Michelangelo, The Creation of Adam (c. 1512, public domain); Albert Gleizes, Woman and Child (Femme et enfant, 1920; the artist died in 1953, so the work has been in the public domain since 2024 where copyright lasts 70 years after the author's death, and in the United States as a work from before 1929)."), small)


def supp_hifi():
    """High-fidelity run (--hifi): outline sources and fits compared, drawn from the series."""
    require(exists('hifi_summary.json'), 'The extended Part 4 needs the high-fidelity run: the program with hifi --paintings')
    hs = data('hifi_summary.json')
    d, a = (hs['paintings']['gleizes'], hs['paintings']['adam'])
    lv = hs['levels']
    M = lambda info, key: info['measures'][key]
    title(T('高精度：以梯度下降直接畫出名畫', 'High fidelity: painting directly with gradient descent'), T('Part 4 延伸 · 高精度', 'Part 4 extended · High fidelity'))
    add(T(f"<b>問題：</b>梯度下降擬合的傅立葉輪廓能否重建整幅畫，精度能到多高？上一節的名畫以 DFT 取得係數再轉成圖元，最多 24 層；這裡另以 --hifi 執行：《女人與孩子》用原始尺寸 {d['size'][0]}×{d['size'][1]}，《創造亞當》只縮小一半為 {a['size'][0]}×{a['size'][1]}；Lab 三版分為 {lv[0]}+{lv[1]}+{lv[2]} 階（共 {sum(lv)} 層）；每條輪廓的諧波數約為輪廓點數的 1/3，上限 {hs['max_harmonics']}。結果不轉成圖元，直接由傅立葉級數取樣後以 Part 2 填色器（k={hs['fill_k']}）填回。比較兩種輪廓來源與三種取得係數的方式：",
          f"<b>Question:</b> can Fourier outlines fitted by gradient descent rebuild a whole painting, and how precisely? The painting runs in the previous section took their coefficients from a DFT, turned them into primitives and used at most 24 layers. A separate run (--hifi) uses Woman and Child at its original {d['size'][0]}×{d['size'][1]} and The Creation of Adam shrunk only by half to {a['size'][0]}×{a['size'][1]}; the Lab plates get {lv[0]}+{lv[1]}+{lv[2]} levels ({sum(lv)} layers), and each outline gets about one harmonic per three outline points, up to {hs['max_harmonics']}. Nothing is turned into primitives: the Fourier series itself is sampled and refilled with the Part 2 rasterizer (k={hs['fill_k']}). Two outline sources and three ways of getting the coefficients are compared:"))
    for text in [
        T('<b>像素邊界輪廓</b>：與上一節的名畫相同，逐像素分階、併入小區塊後沿像素邊界追蹤。',
          '<b>Crack outlines</b>: as for the paintings in the previous section, each pixel takes a level, small regions are merged, and the boundaries are traced along pixel cracks.'),
        T(f"<b>平滑輪廓</b>：先以雙邊濾波平滑，再以 marching squares 取平滑後通道在相鄰兩階中點處的等值線，交點在格邊上線性內插，因此是次像素精度、沒有階梯；面積小於 {hs['min_area']} 像素的輪廓捨去。",
          f"<b>Smooth outlines</b>: the picture is first smoothed with the bilateral filter; each level boundary is then the iso-line (marching squares) of the smoothed channel at the midpoint between two levels, its crossings interpolated along the cell edges, so it is sub-pixel and has no staircase; outlines enclosing under {hs['min_area']} pixels are dropped."),
        T('<b>DFT</b>：有序輪廓依長度等距重新取樣後做 FFT，保留低頻係數。',
          '<b>DFT</b>: the ordered outline, resampled evenly by length and transformed with an FFT, keeping the low harmonics.'),
        T(f"<b>DFT＋梯度下降</b>：以 DFT 結果為起點，用與測試星形相同的對稱 chamfer 損失再下降 {hs['refine_steps']} 步（學習率 {hs['rate']}）。",
          f"<b>DFT + gradient descent</b>: starting from the DFT result, {hs['refine_steps']} more steps of descent (rate {hs['rate']}) on the same symmetric chamfer loss as the test star."),
        T(f"<b>僅梯度下降</b>：與測試星形相同，從質心上的圓開始、由粗到細（4、16、64… 個諧波，每階段 {hs['stage_steps']} 步），只使用輪廓上的點，不使用它們的順序。",
          f"<b>Gradient descent alone</b>: as for the test star, from a circle at the centroid, coarse to fine (4, 16, 64, ... harmonics, {hs['stage_steps']} steps per stage), using the outline's points but not their order."),
    ]:
        bullet(text)

    def reduced(name):
        """The bitmap averaged over 2x2 blocks: printed 2-4 inches wide, the
        full-size hi-fi bitmaps would make the PDF four times larger."""
        with PILImage.open(art(name)) as im:
            half = im.convert('RGB').reduce(2)
        buffer = BytesIO()
        half.save(buffer, 'PNG')
        buffer.seek(0)
        return buffer

    def exact_crop(name, box, width):
        """Only the pixels inside `box`, unchanged, enlarged on the page."""
        with PILImage.open(art(name)) as im:
            part = im.convert('RGB').crop(box)
        buffer = BytesIO()
        part.save(buffer, 'PNG')
        buffer.seek(0)
        return Image(buffer, width=width, height=width * (box[3] - box[1]) / (box[2] - box[0]))

    labels = [('input', T('輸入', 'Input')), ('crack_dft', T('像素邊界＋DFT', 'Cracks + DFT')), ('refined', T('平滑＋DFT＋梯度下降', 'Smooth + DFT + descent')), ('descent', T('平滑＋僅梯度下降', 'Smooth + descent alone'))]
    for name, info in (('adam', a), ('gleizes', d)):
        columns = 2 if name == 'adam' else 4
        cell = (W - 6 * columns) / columns
        height = cell * info['size'][1] / info['size'][0]
        cells = []
        for start in range(0, 4, columns):
            chunk = labels[start:start + columns]
            cells.append([Image(reduced(f'hifi_{name}_{v}.bmp'), width=cell, height=height) for v, _ in chunk])
            cells.append([p(c, caption) for _, c in chunk])
        grid(cells, columns)
    width = (W - 24) / 4
    for name, box, zh, en in (('adam', (570, 300, 790, 430), '手指', 'Fingers'), ('gleizes', (420, 180, 680, 440), '臉部與細線', 'Face and thin lines')):
        crops = [('input', T(f'{zh}：輸入', f'{en}: input')), ('crack_dft', T('像素邊界＋DFT', 'Cracks + DFT')), ('dft', T('平滑＋DFT', 'Smooth + DFT')), ('refined', T('平滑＋DFT＋梯度下降', 'Smooth + DFT + descent'))]
        grid([[exact_crop(f'hifi_{name}_{v}.bmp', box, width) for v, _ in crops], [p(c, caption) for _, c in crops]], 4)
    story.append(p(T('全圖以 2×2 區塊平均縮小後放入頁面（僅為控制 PDF 大小）；放大圖為原始像素。',
                     'Whole pictures are averaged over 2×2 blocks for the page (only to keep the PDF small); the close-ups are the original pixels.'), caption))
    header = [T('畫作與做法', 'Painting and method'), T('平均 ΔE', 'Mean ΔE'), 'SSIM', T('ΔE > 10 像素', 'Pixels ΔE > 10'), T('邊緣 RMS（px）', 'Edge RMS (px)'), T('CPU 秒', 'CPU s')]
    body_rows = []
    for pname, info in ((T('《女人與孩子》', 'Gleizes'), d), (T('《亞當》', 'Adam'), a)):
        sep = T('：', ': ')
        for key, label, family, variant in (('quant_raw', T('像素分階（未向量化）', 'raw levels, not vectorized'), None, None),
                                            ('crack_dft', T('像素邊界＋DFT', 'cracks + DFT'), 'crack', 'dft'),
                                            ('crack_refined', T('像素邊界＋DFT＋下降', 'cracks + DFT + descent'), 'crack', 'refined'),
                                            ('quant', T('平滑分階（未向量化）', 'smoothed levels, not vectorized'), None, None),
                                            ('dft', T('平滑＋DFT', 'smooth + DFT'), 'smooth', 'dft'),
                                            ('refined', T('平滑＋DFT＋下降', 'smooth + DFT + descent'), 'smooth', 'refined'),
                                            ('descent', T('平滑＋僅下降', 'smooth + descent alone'), 'smooth', 'descent')):
            m = M(info, key)
            rms = f"{info[family]['edge_rms'][variant]:.3f}" if family else '-'
            cpu = f"{info[family]['compute_seconds'][variant]:,.0f}" if family else '-'
            body_rows.append([pname + sep + label, f"{m['mean_delta_e']:.2f}", f"{m['ssim']:.3f}", f"{m['visible']:,}", rms, cpu])
    table([header] + body_rows, [190, 50, 46, 72, 66, W - 424], keep=True)
    story.append(p(T('ΔE 與 SSIM 均與原圖（縮小後、未平滑）比較。邊緣 RMS 為各輪廓 chamfer 損失開根號、以輪廓點數加權的平均。CPU 秒為所有輪廓擬合時間的總和。',
                     "ΔE and SSIM are against the (shrunk, unsmoothed) original. Edge RMS is the square root of each outline's chamfer loss, averaged with the outline's point count as weight. CPU seconds sum the fitting time of all outlines."), caption))
    loops_rows = [r for name in ('gleizes', 'adam') for r in rows(f'hifi_{name}_loops.csv') if r['outline'] == 'smooth']
    stuck = lambda r: float(r['loss_descent']) > hs['stuck_factor'] * float(r['loss_dft']) + hs['stuck_allowance']
    size_rows = []
    for lo, hi in [(0, 100), (100, 1000), (1000, 5000), (5000, None)]:
        g = [r for r in loops_rows if int(r['n']) >= lo and (hi is None or int(r['n']) < hi)]
        med = lambda key: statistics.median(float(r[key]) for r in g)
        better = sum(float(r['loss_refined']) < float(r['loss_dft']) for r in g)
        size_rows.append([f'{lo:,}-{hi - 1:,}' if hi else f'{lo:,}+', f'{len(g):,}', f'{better:,}', f'{med("loss_dft"):.3f}', f'{med("loss_refined"):.3f}', f'{med("loss_descent"):.3f}', f'{sum(map(stuck, g)):,}'])
    table([[T('輪廓點數', 'Outline points'), T('輪廓', 'Outlines'), T('DFT＋下降優於 DFT', 'DFT + descent beats DFT'), T('損失中位數：DFT', 'Median loss: DFT'), T('DFT＋下降', 'DFT + descent'), T('僅下降', 'Descent alone'), T('僅下降卡住', 'Descent stuck')]] + size_rows, [66, 52, 80, 82, 70, 64, W - 414], keep=True)
    all_refined = sum(float(r['loss_refined']) < float(r['loss_dft']) for r in loops_rows)
    stuck_points = sum(int(r['n']) for r in loops_rows if stuck(r)) / sum(int(r['n']) for r in loops_rows) * 100
    big = [r for r in loops_rows if int(r['n']) >= 5000]
    big_stuck = sum(map(stuck, big)) / len(big) * 100
    story.append(p(T(f"兩幅畫的平滑輪廓合計。損失單位為 px²；輪廓點彼此相距約 1 px，即使曲線完美，到最近點的距離也不為 0，因此損失有一個大於 0 的下限。「卡住」指損失大於 DFT 的 {hs['stuck_factor']:.0f} 倍再加 {hs['stuck_allowance']}。",
                     f"Smooth outlines of both paintings together. Loss is in px²; outline points are about 1 px apart, so even a perfect curve keeps a nonzero distance to the nearest point, and the loss has a floor above zero. \"Stuck\" means a loss above {hs['stuck_factor']:.0f} times the DFT's plus {hs['stuck_allowance']}."), caption))
    dc, ac = (d['crack'], a['crack'])
    ds, as_ = (d['smooth'], a['smooth'])
    add(T(f"<b>1. 破碎的邊緣來自分階，不是 DFT。</b>像素邊界輪廓上，邊緣參差不齊，臉部的白色細線斷開、手指變形；放大比較可見這些破碎在分階的圖上就已存在，DFT 只是忠實地畫出來。原因有三：JPEG 雜訊與筆觸讓閾值附近的像素在相鄰兩階之間來回跳動；小區塊被併入鄰居成為斑塊；以及像素邊界本身的階梯。改用平滑輪廓後，《女人與孩子》的平均 ΔE 由 {M(d, 'crack_dft')['mean_delta_e']:.2f} 降到 {M(d, 'dft')['mean_delta_e']:.2f}，SSIM 由 {M(d, 'crack_dft')['ssim']:.3f} 升到 {M(d, 'dft')['ssim']:.3f}，明顯色差的像素由 {M(d, 'crack_dft')['visible']:,} 降到 {M(d, 'dft')['visible']:,}；《創造亞當》由 {M(a, 'crack_dft')['mean_delta_e']:.2f} 降到 {M(a, 'dft')['mean_delta_e']:.2f}。雜訊不再產生輪廓，輪廓由 {dc['loops'] + ac['loops']:,} 條減為 {ds['loops'] + as_['loops']:,} 條，係數由 {dc['coefficients'] + ac['coefficients']:,} 個減為 {ds['coefficients'] + as_['coefficients']:,} 個：更準確，也更精簡。",
          f"<b>1. The broken edges come from the levelling, not from the DFT.</b> On crack outlines the edges are ragged, the thin white line of the face breaks and the fingers lose their shape; the close-ups show that the breaks are already in the levelled picture and the DFT only draws them faithfully. There are three causes: JPEG noise and brushwork make pixels near a threshold jump between two levels; small regions merged into a neighbor become blotches; and the pixel cracks are a staircase. With smooth outlines, Woman and Child's mean ΔE goes from {M(d, 'crack_dft')['mean_delta_e']:.2f} to {M(d, 'dft')['mean_delta_e']:.2f}, SSIM from {M(d, 'crack_dft')['ssim']:.3f} to {M(d, 'dft')['ssim']:.3f}, and the visibly wrong pixels from {M(d, 'crack_dft')['visible']:,} to {M(d, 'dft')['visible']:,}; The Creation of Adam goes from {M(a, 'crack_dft')['mean_delta_e']:.2f} to {M(a, 'dft')['mean_delta_e']:.2f}. Noise no longer makes outlines: they fall from {dc['loops'] + ac['loops']:,} to {ds['loops'] + as_['loops']:,} and the coefficients from {dc['coefficients'] + ac['coefficients']:,} to {ds['coefficients'] + as_['coefficients']:,}: more accurate and leaner."))
    # The same refinement on the crack outlines: closer edges, and does the color follow?
    crack_worse = [M(i, 'crack_refined')['mean_delta_e'] > M(i, 'crack_dft')['mean_delta_e'] for i in (d, a)]
    crack_sentence = T(
        f"同樣的精修用在像素邊界輪廓上，邊緣 RMS 也由 {dc['edge_rms']['dft']:.3f} 與 {ac['edge_rms']['dft']:.3f} 降到 {dc['edge_rms']['refined']:.3f} 與 {ac['edge_rms']['refined']:.3f} px，平均 ΔE 卻由 {M(d, 'crack_dft')['mean_delta_e']:.2f} 與 {M(a, 'crack_dft')['mean_delta_e']:.2f} 變為 {M(d, 'crack_refined')['mean_delta_e']:.2f} 與 {M(a, 'crack_refined')['mean_delta_e']:.2f}"
        + ("：曲線貼近的是像素階梯，而不是畫上原本的邊緣；改用平滑輪廓後，邊緣與顏色才一致。" if all(crack_worse) else "。"),
        f"The same refinement on crack outlines also brings edge RMS from {dc['edge_rms']['dft']:.3f} and {ac['edge_rms']['dft']:.3f} down to {dc['edge_rms']['refined']:.3f} and {ac['edge_rms']['refined']:.3f} px, while mean ΔE goes from {M(d, 'crack_dft')['mean_delta_e']:.2f} and {M(a, 'crack_dft')['mean_delta_e']:.2f} to {M(d, 'crack_refined')['mean_delta_e']:.2f} and {M(a, 'crack_refined')['mean_delta_e']:.2f}"
        + (": what the curve gets closer to there is the pixel staircase, not the painting's own edge; only with smooth outlines do edges and colors agree." if all(crack_worse) else "."))
    add(T(f"<b>2. 從 DFT 出發再做梯度下降，邊緣與顏色都更準。</b>{len(loops_rows):,} 條平滑輪廓有 {all_refined:,} 條改善；邊緣 RMS 在《女人與孩子》由 {ds['edge_rms']['dft']:.3f} 降到 {ds['edge_rms']['refined']:.3f} px，《創造亞當》由 {as_['edge_rms']['dft']:.3f} 降到 {as_['edge_rms']['refined']:.3f} px，平均 ΔE 也分別降為 {M(d, 'refined')['mean_delta_e']:.2f} 與 {M(a, 'refined')['mean_delta_e']:.2f}，是所有向量化結果中最好的。所以梯度下降可以畫出整幅畫，而且比 DFT 更精確：DFT 固定以沿輪廓的長度作為參數，梯度下降則讓曲線自己決定參數沿輪廓怎麼走。" + crack_sentence,
          f"<b>2. Starting from the DFT, gradient descent makes both edges and colors more precise.</b> {all_refined:,} of {len(loops_rows):,} smooth outlines improve; edge RMS falls from {ds['edge_rms']['dft']:.3f} to {ds['edge_rms']['refined']:.3f} px on Woman and Child and from {as_['edge_rms']['dft']:.3f} to {as_['edge_rms']['refined']:.3f} px on The Creation of Adam, and mean ΔE falls to {M(d, 'refined')['mean_delta_e']:.2f} and {M(a, 'refined')['mean_delta_e']:.2f}, the best of all vectorized results. Gradient descent can therefore paint the whole picture, more precisely than the DFT: the DFT ties the parameter to length along the outline, while descent lets the curve choose how its parameter runs along it. " + crack_sentence))
    add(T(f"<b>3. 失敗模式：走訪順序。</b>只用梯度下降、從圓開始時，小於 100 點的輪廓與 DFT 相當，超過 5,000 點的卻有 {big_stuck:.0f}% 卡住。卡住的 {d['descent_stuck'] + a['descent_stuck']} 條輪廓佔平滑輪廓點數的 {stuck_points:.0f}%，畫面大片出錯，平均 ΔE 為 {M(d, 'descent')['mean_delta_e']:.2f} 與 {M(a, 'descent')['mean_delta_e']:.2f}。卡住的曲線行進方向正確、幾乎全程貼著真正的邊緣，但走訪的順序錯了，在輪廓相隔很遠的兩處之間直線抄捷徑；以非零繞數填色時，捷徑圍出的整片區域內外顛倒。L* 版的輪廓較單純、錯得少，a*、b* 版的輪廓曲折、錯得多，所以明暗大致正確而顏色錯誤。同一套梯度、取樣與填色程式從 DFT 起點出發時讓{'每條' if all_refined == len(loops_rows) else '幾乎每條'}輪廓都改善，可見問題不在計算：chamfer 損失只問曲線與輪廓點是否接近、不問順序，是非凸函數，從圓出發落進了局部最佳解。",
          f"<b>3. Failure mode: the order of the visit.</b> Descent alone, from a circle, matches the DFT on outlines below 100 points, but {big_stuck:.0f}% of those above 5,000 points get stuck. The {d['descent_stuck'] + a['descent_stuck']} stuck outlines hold {stuck_points:.0f}% of the smooth outline points, so large areas go wrong and mean ΔE is {M(d, 'descent')['mean_delta_e']:.2f} and {M(a, 'descent')['mean_delta_e']:.2f}. A stuck curve turns the right way and runs along the real edges almost everywhere, but visits them in the wrong order and cuts straight across between distant parts of the outline; filled with the non-zero winding rule, each area such a shortcut encloses comes out inside out. L* outlines are simpler and go wrong less, a* and b* outlines are intricate and go wrong more, which is why light and shade come out roughly right and the colors wrong. The same gradient, sampling and fill code improves {'every' if all_refined == len(loops_rows) else 'almost every'} outline when started from the DFT, so the computation is sound: the chamfer loss asks only whether curve and points are close, not in which order, is not convex, and from a circle the descent lands in a local optimum."))
    add(T(f"<b>4. 好的起點比更多的迭代重要。</b>順序資訊才是關鍵，而 DFT 起點正好提供了它：DFT 本身只花 {ds['compute_seconds']['dft'] + as_['compute_seconds']['dft']:.0f} CPU 秒，下降精修約 {(ds['compute_seconds']['refined'] + as_['compute_seconds']['refined']) / 60:.0f} CPU 分鐘，僅下降約 {(ds['compute_seconds']['descent'] + as_['compute_seconds']['descent']) / 60:.0f} CPU 分鐘卻得到最差的結果。在損失中加入面積項並沒有解決順序問題（見附錄）；若要不依賴順序，可行的方向是 LIVE（2022）的做法：逐步加入許多簡單的路徑，每條放在目前誤差最大的地方。以 {hs['threads']} 個執行緒，兩幅畫實際各需 {d['wall_seconds'] / 60:.1f} 與 {a['wall_seconds'] / 60:.1f} 分鐘。精度也有儲存上的代價：《女人與孩子》用了 {ds['coefficients']:,} 個複數係數，點陣圖只有 {d['bitmap_bytes']:,} 位元組；高精度級數的價值在於與解析度無關、可任意放大，而不是壓縮。",
          f"<b>4. A good starting point matters more than more iterations.</b> The order is what matters, and the DFT start supplies it: the DFT itself takes {ds['compute_seconds']['dft'] + as_['compute_seconds']['dft']:.0f} CPU seconds, refining by descent about {(ds['compute_seconds']['refined'] + as_['compute_seconds']['refined']) / 60:.0f} CPU minutes, and descent alone about {(ds['compute_seconds']['descent'] + as_['compute_seconds']['descent']) / 60:.0f} CPU minutes for the worst result. Adding a region term to the loss did not fix the order (see the appendix); to do without the order, a workable direction is that of LIVE (2022): add many simple paths one at a time, each where the current error is largest. With {hs['threads']} threads the two paintings took {d['wall_seconds'] / 60:.1f} and {a['wall_seconds'] / 60:.1f} minutes of real time. Precision also costs storage: Woman and Child uses {ds['coefficients']:,} complex coefficients against a bitmap of {d['bitmap_bytes']:,} bytes; the value of the high-precision series is that it is resolution independent and can be enlarged at will, not compression."))


def supp_transform():
    tf = data('transform_summary.json')
    tf_curve = rows('transform_curve.csv')
    gray_at = {int(r['steps']): int(r['partial_pixels']) for r in tf_curve}

    def loglog_slope(key):
        xs = [math.log(int(r['steps'])) for r in tf_curve]
        ys = [math.log(float(r[key])) for r in tf_curve]
        mx, my = (sum(xs) / len(xs), sum(ys) / len(ys))
        return sum(((x - mx) * (y - my) for x, y in zip(xs, ys))) / sum(((x - mx) ** 2 for x in xs))
    gray_slope = loglog_slope('partial_pixels')
    title(T('先旋轉再點陣化，或先點陣化再旋轉', 'Rotate then rasterize, or the reverse'), 'Transformations')
    add(T('本文件不繳交：它回答 Part 4 清單上的另一個選項（Transformations），與報告選定的向量化題目分開。',
          'Not submitted: this experiment answers another option on the Part 4 list (Transformations) and is kept apart from the vectorizing topic the report chose.'), small)
    add(T('<b>Transformations（第 4 章）：繞任意點旋轉。</b>以 Point2::rotated_about 實作：先平移使旋轉中心到原點、乘上旋轉矩陣、再平移回去。點陣圖旋轉 Image::rotated_bilinear 為自寫的重取樣：每個輸出像素中心反向旋轉回原圖，再以雙線性插值讀值，圖外視為空白。第 3 章預測向量會撐過反覆變換、點陣不會；這裡量測差距。',
          '<b>Transformations (Chapter 4): rotation about an arbitrary point.</b> It is implemented in Point2::rotated_about: translate so that the center of rotation is at the origin, multiply by the rotation matrix, and translate back. Bitmap rotation, Image::rotated_bilinear, is our own resampling: each output pixel center is rotated back into the source image and read with bilinear interpolation, with everything outside the image treated as blank. Chapter 3 predicts that the vector survives repeated transformation and the bitmap does not; this experiment measures the gap.'))
    add(T(f"對象為 Part 2 的星形，繞畫布中心 ({tf['center'][0]},{tf['center'][1]}) 每次轉 {tf['step_degrees']}°，共 {tf['steps']} 次。<b>向量路線</b>：每次只旋轉五個頂點，最後點陣化一次。<b>點陣路線</b>：先點陣化，再將點陣圖旋轉並重取樣 {tf['steps']} 次。兩條路線及參考圖都使用同一個 k={tf['k']} supersampling 填色器，參考圖為原頂點一次轉 100° 後點陣化。",
          f"The subject is the Part 2 star, rotated about the canvas center ({tf['center'][0]},{tf['center'][1]}) by {tf['step_degrees']}° at a time, {tf['steps']} times. <b>Vector route</b>: each step rotates only the five vertices, and the result is rasterized once at the end. <b>Bitmap route</b>: rasterize first, then rotate and resample the bitmap {tf['steps']} times. Both routes and the reference use the same k={tf['k']} supersampling fill; the reference rasterizes the original vertices after a single 100° rotation."))
    cells = [[p(T('參考：頂點一次轉 100°', 'Reference: vertices rotated 100° at once'), small), p(T('向量路線：頂點轉 100 次', 'Vector route: vertices rotated 100 times'), small)], [Image(art('transform_reference.bmp'), width=160, height=160), Image(art('transform_vector.bmp'), width=160, height=160)], [p(T('點陣路線：重取樣 100 次', 'Bitmap route: resampled 100 times'), small), p(T('|點陣路線 - 參考|，越黑誤差越大', '|bitmap route - reference|, darker = larger error'), small)], [Image(art('transform_bitmap.bmp'), width=160, height=160), Image(art('transform_error.bmp'), width=160, height=160)]]
    t = Table(cells, colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    story.append(t)
    table([[T('路線', 'Route'), T('與參考的 RMS 差', 'RMS difference from reference'), T('最大差', 'Max difference'), T('灰色像素數', 'Gray pixels')], [T('參考（一次轉 100°）', 'Reference (100° at once)'), '0', '0', f"{tf['partial_reference']:,}"], [T('向量：頂點轉 100 次', 'Vector: vertices rotated 100 times'), f"{tf['vector_rms']:.6f}", f"{tf['vector_max_abs']:.6f}", f"{tf['partial_vector']:,}"], [T('點陣：重取樣 100 次', 'Bitmap: resampled 100 times'), f"{tf['bitmap_rms']:.6f}", f"{tf['bitmap_max_abs']:.6f}", f"{tf['partial_bitmap']:,}"], [T('對照：點陣一次轉 100°', 'Control: bitmap rotated 100° at once'), f"{tf['once_rms']:.6f}", f"{tf['once_max_abs']:.6f}", f"{tf['partial_once']:,}"]], [150, 105, 90, W - 345], keep=True)
    story.append(p(T('差值為每像素覆蓋率（0–1）之差。灰色像素指 8-bit 輸出中既非全白也非全黑的像素，用來衡量邊緣過渡帶有多寬。',
                     'Differences are per-pixel differences in coverage (0 to 1). Gray pixels are pixels of the 8-bit output that are neither fully white nor fully black; they measure how wide the edge transition band is.'), small))
    add(T(f"<b>向量路線完好無損。</b>100 次旋轉後頂點最大偏移僅 {tf['vertex_max_drift_px']:.1e} px，只是浮點捨入；點陣化結果與參考圖逐像素相同，RMS 差為 0。每一步都只在連續座標上運算，從未落到像素網格，因此沒有東西可以累積。",
          f"<b>The vector route comes through intact.</b> After 100 rotations the largest vertex drift is only {tf['vertex_max_drift_px']:.1e} px, mere floating-point rounding; the rasterized result is identical to the reference pixel for pixel, with an RMS difference of 0. Every step operates only on continuous coordinates and never lands on the pixel grid, so there is nothing to accumulate."))
    add(T(f"<b>點陣路線明顯劣化。</b>RMS 差 {tf['bitmap_rms']:.4f}、最大差 {tf['bitmap_max_abs']:.2f}，誤差圖顯示錯誤全部集中在邊緣與五個尖角：尖角被磨圓，邊緣糊開。灰色像素由 {tf['partial_reference']:,} 增為 {tf['partial_bitmap']:,}（{tf['partial_bitmap'] / tf['partial_reference']:.1f} 倍）。總覆蓋面積只差 {abs(tf['area_bitmap'] / tf['area_reference'] - 1) * 100:.2f}%，所以星形並沒有變小，而是被「抹開」：墨水量守恆，銳利度流失。",
          f"<b>The bitmap route degrades visibly.</b> The RMS difference is {tf['bitmap_rms']:.4f} and the maximum difference {tf['bitmap_max_abs']:.2f}; the error image shows that all of the error is concentrated on the edges and the five tips: the tips are rounded off and the edges blur. Gray pixels increase from {tf['partial_reference']:,} to {tf['partial_bitmap']:,} ({tf['partial_bitmap'] / tf['partial_reference']:.1f} times). The total covered area differs by only {abs(tf['area_bitmap'] / tf['area_reference'] - 1) * 100:.2f}%, so the star has not shrunk but has been smeared out: the amount of ink is conserved while sharpness is lost."))
    add(T(f"<b>對照組</b>只重取樣一次、直接轉 100°，RMS 差僅 {tf['once_rms']:.4f}，與重取樣一次 1° 的 {float(tf_curve[0]['rms']):.4f} 相當。可見損壞來自重取樣的<b>次數</b>，而不是旋轉角度的大小。",
          f"<b>The control</b> resamples only once, rotating directly by 100°; its RMS difference is only {tf['once_rms']:.4f}, comparable to the {float(tf_curve[0]['rms']):.4f} of a single 1° resampling. The damage therefore comes from the <b>number</b> of resamplings, not from the size of the rotation angle."))
    story.append(Chart([(T('點陣路線灰色像素（千）', 'Bitmap-route gray pixels (thousands)'), ORANGE, [(int(r['steps']), int(r['partial_pixels']) / 1000) for r in tf_curve]), (T('G(1)×√n（千）', 'G(1)×√n (thousands)'), BLUE, [(int(r['steps']), gray_at[1] * math.sqrt(int(r['steps'])) / 1000) for r in tf_curve])], height=165, xlabel=T('重取樣次數 n', 'Number of resamplings n'), ylabel=T('灰色像素（千）', 'Gray pixels (thousands)')))
    story.append(Spacer(1, 6))
    add(T(f"<b>發現：邊緣寬度隨 √n 成長。</b>對 100 個點做 log-log 迴歸，灰色像素數的指數為 {gray_slope:.2f}，與 1/2 吻合：次數每 4 倍，過渡帶約寬 2 倍（n=1、4、16 時為 {gray_at[1]:,}、{gray_at[4]:,}、{gray_at[16]:,}）。雙線性插值在小數位移 f 處等於與 [1-f, f] 做卷積，相當於一次變異數 f(1-f) 的小模糊；每次旋轉的位移不同，這些模糊彼此獨立，變異數相加，就像隨機漫步，標準差與過渡寬度因此正比於 √n。前幾次重取樣的單次傷害最大：第 25 次時灰色像素已達最終的 {gray_at[25] / gray_at[100] * 100:.0f}%。",
          f"<b>Finding: the edge width grows as √n.</b> A log-log regression over the 100 points gives an exponent of {gray_slope:.2f} for the gray-pixel count, consistent with 1/2: every fourfold increase in the number of resamplings makes the transition band about twice as wide ({gray_at[1]:,}, {gray_at[4]:,} and {gray_at[16]:,} at n=1, 4 and 16). Bilinear interpolation at a fractional offset f equals a convolution with [1-f, f], i.e. a small blur with variance f(1-f); each rotation has different offsets, so these blurs are independent and their variances add, as in a random walk, which makes the standard deviation, and with it the transition width, proportional to √n. The first few resamplings do the most damage each: by the 25th, the gray pixels already reach {gray_at[25] / gray_at[100] * 100:.0f}% of the final count."))
    add(T('<b>與第 3 章的預測一致。</b>向量圖以數學描述形狀，變換只改座標，可以反覆進行而不失真；點陣圖的每次變換都必須重新取樣，資訊會逐次流失。向量路線誤差為 0，點陣路線誤差隨次數持續累積。實務上應保留向量（或原始高解析）來源，從原圖一次變換到目標角度，而不是對已變換過的點陣圖再變換。',
          "<b>This agrees with Chapter 3's prediction.</b> A vector image describes shapes mathematically, so a transformation only changes coordinates and can be repeated without loss; every transformation of a bitmap must resample it, and information is lost each time. The vector route has zero error, while the error of the bitmap route keeps accumulating with the number of steps. In practice one should keep the vector (or original high-resolution) source and transform it to the target angle in one step, rather than transforming an already transformed bitmap again."))


def supp_related():
    title(T('與已知方法的關係', 'Relation to known methods'), T('Part 4 延伸 · 相關做法', 'Part 4 extended · Related work'))
    add(T('流程中的每個元件都是已知方法；這次的價值在於把它們串成同一個流程並逐步量測，例如「Gibbs 現象只影響角點附近，以相鄰圖元的交點即可補回尖角」，以及分色法的套色誤差。',
          'Every component of the pipeline is a known method; the value of this work lies in chaining them into one pipeline and measuring each step, for example "the Gibbs phenomenon only affects the neighborhood of the corners, and intersecting adjacent primitives restores the sharp corners", and the misregistration of the ink-plate approach.'))
    table([[T('本流程的步驟', 'Step of this pipeline'), T('已知做法', 'Known method')], [T('用傅立葉級數描述輪廓', 'Describing an outline with a Fourier series'), T('Fourier descriptors（Zahn & Roskies 1972）、橢圓傅立葉描述子（Kuhl & Giardina 1982）', 'Fourier descriptors (Zahn & Roskies 1972), elliptic Fourier descriptors (Kuhl & Giardina 1982)')], [T('以最佳化讓曲線貼合邊緣', 'Fitting a curve to edges by optimization'), T('Active contours（Kass, Witkin & Terzopoulos 1988）；傅立葉參數化可變形輪廓（Staib & Duncan 1992）', 'Active contours (Kass, Witkin & Terzopoulos 1988); Fourier-parameterized deformable contours (Staib & Duncan 1992)')], [T('以梯度下降調整曲線參數', 'Adjusting curve parameters by gradient descent'), T('可微分向量圖形 DiffVG（Li 等 2020）、LIVE（2022）', 'Differentiable vector graphics DiffVG (Li et al. 2020), LIVE (2022)')], [T('分出形狀與洞', 'Separating shapes and holes'), T('輪廓追蹤與階層（Suzuki & Abe 1985）；DBSCAN（Ester 等 1996）', 'Contour tracing and hierarchy (Suzuki & Abe 1985); DBSCAN (Ester et al. 1996)')], [T('找角點、切段', 'Finding corners and cutting pieces'), T('Douglas-Peucker、曲率尺度空間（Mokhtarian & Mackworth）；直線與圓弧切段（Rosin & West 1995）', 'Douglas-Peucker, curvature scale space (Mokhtarian & Mackworth); segmentation into lines and arcs (Rosin & West 1995)')], [T('直線、圓、Bézier 擬合', 'Fitting lines, circles and Béziers'), T('總體最小平方；Kåsa（1976）圓擬合；Schneider（1990）Bézier 擬合', 'Total least squares; Kåsa (1976) circle fitting; Schneider (1990) Bézier fitting')], [T('點陣轉向量的完整工具', 'Complete raster-to-vector tools'), T('Potrace（Selinger 2003）、AutoTrace、VTracer', 'Potrace (Selinger 2003), AutoTrace, VTracer')], [T('A：沿空間填充曲線掃描', 'A: scanning along a space-filling curve'), T('以希爾伯特曲線排序像素的影像壓縮與掃描', 'Image compression and scanning that order pixels along a Hilbert curve')]], [150, W - 150])
    story.append(p(T('設計選擇與其來源', 'Design choices and where they come from'), subheading))
    for text in [
        T('<b>圖元擬合原始邊緣點，而非傅立葉曲線。</b>內凹銳角附近的 Gibbs 波紋會超出容差，使圓弧被誤判為 Bézier；傅立葉曲線因此只用來排序與找角點。',
          '<b>Primitives are fitted to the original edge points, not to the Fourier curve.</b> Near sharp inner corners the Gibbs ripples exceed the tolerance, so arcs would be taken for Béziers; the Fourier curve is used only for ordering and finding corners.'),
        T('<b>切分再擬合（split-and-fit）。</b>參考 Rosin &amp; West 與 Potrace：一段既不是直線也不是圓弧時，在段內轉角最大處再切一刀。淺角因此被找回，而圓弧段本來就擬合得上，不會被多切。',
          '<b>Split-and-fit.</b> Following Rosin &amp; West and Potrace: when a piece is neither a line nor an arc, it is cut again where its turning angle is largest. Shallow corners are found this way, while arc pieces, which fit in the first place, are not cut any further.'),
        T('<b>洞的階層。</b>參考 OpenCV findContours 的外框與洞階層，以包含深度決定外框與洞，並讓洞反向。',
          '<b>Hole hierarchy.</b> Following the outer-outline and hole hierarchy of OpenCV findContours, the containment depth decides which outlines are outer outlines and which are holes, and the holes are reversed.'),
        T('<b>彩色依顏色分群、堆疊式排列。</b>參考 VTracer 先依顏色分群再擬合輪廓，以及它的 stacked 模式，讓每層延伸到上層底下；另加混色反解處理抗鋸齒邊緣。',
          '<b>Clustering by color, and stacked layers.</b> Following VTracer, which clusters by color before fitting outlines, and its stacked mode, in which each layer runs under the layers above; unmixing is added to handle anti-aliased edges.'),
        T('<b>以 Lab 分版。</b>參考 JPEG 把亮度與色差分開（YCbCr）的概念，改用依人眼設計的 Lab：亮度與兩個色差各自分階，避免 CMY 色版在近中性色上疊出雜色。',
          '<b>Lab plates.</b> Following the idea of JPEG, which separates lightness from color difference (YCbCr), the perceptual Lab space is used: lightness and two color differences, each levelled on its own, which avoids the color noise CMY plates pile up on near-neutral areas.'),
    ]:
        bullet(text)


def supp_devlog():
    title(T('附錄：開發紀錄', 'Appendix: development notes'), T('Part 4 延伸 · 附錄', 'Part 4 extended · Appendix'))
    add(T('以下記錄流程如何演變成現在的樣子。這些是工程歷程，不是實驗結果；標示為人工觀察的數字不是程式輸出，也未經稽核。',
          'These notes record how the pipeline reached its present form. They are engineering history, not results; numbers marked as manual observations were not produced by the program and were not checked by the audit.'))
    # The region-term sweep, summarized from its table (shown below).
    sweep = rows('hifi_region_sweep.csv')
    hs = data('hifi_summary.json')
    by_outline = {}
    for r in sweep:
        by_outline.setdefault((r['painting'], r['plate']), {})[(r['fit'], float(r['weight']))] = r
    cuts, ratios = ([], [])
    for fits in by_outline.values():
        plain, weak, dft_row = (fits[('descent', 0.0)], fits.get(('descent', 1.0)), fits[('dft', 0.0)])
        if weak and weak['diverged'] == '0' and plain['diverged'] == '0':
            cuts.append((1 - int(weak['wrong_pixels']) / int(plain['wrong_pixels'])) * 100)
            ratios.append(int(weak['wrong_pixels']) / int(dft_row['wrong_pixels']))
    strongest = max(hs['region_weights'])
    all_diverge = all(fits[('descent', strongest)]['diverged'] == '1' for fits in by_outline.values())
    region_zh = (f"修正後，權重 1 讓填錯的像素依輪廓不同減少 {min(cuts):.0f}% 至 {max(cuts):.0f}%，但仍是 DFT 的 {min(ratios):.0f} 至 {max(ratios):.0f} 倍；"
                 + (f"權重 {strongest:g} 時每一條輪廓都發散。" if all_diverge else f"權重 {strongest:g} 時有輪廓發散。")
                 + "面積項只能把每一點往內或往外推，無法改變曲線走訪輪廓的順序。數字見下表。")
    region_en = (f"With both fixes, weight 1 cut the wrongly filled pixels by {min(cuts):.0f}% to {max(cuts):.0f}% depending on the outline, yet still left {min(ratios):.0f} to {max(ratios):.0f} times as many as the DFT; "
                 + (f"at weight {strongest:g} every outline diverged. " if all_diverge else f"at weight {strongest:g} some outlines diverged. ")
                 + "The region term can only push each point in or out, while the problem is the order in which the curve visits the outline. The numbers are in the table below.")
    for text in [
        T('<b>邊緣點的內外判定。</b>最初以「大於 0.5」判為內部；三形狀場景中，覆蓋率剛好 0.5 的像素讓邊緣點漏掉，輪廓斷成許多碎片，DBSCAN 也分出過多的群。改為「大於或等於 0.5 為內」後恢復正常。',
          '<b>The inside test for edge points.</b> At first "more than 0.5" counted as inside; in the three-shape scene, pixels with coverage of exactly 0.5 made edge points go missing, the outlines broke into many fragments, and DBSCAN found too many clusters. With "0.5 or more counts as inside" they were whole again.'),
        T('<b>圖元的擬合對象。</b>最初直接對傅立葉曲線擬合圖元；彩色場景中紅色「圓盤減星形」區域的圓弧因內凹角附近的 Gibbs 波紋被判成 Bézier。改為擬合原始邊緣點後，恢復為直線加圓弧。',
          '<b>What the primitives are fitted to.</b> At first the primitives were fitted directly to the Fourier curve; in the color scene, the arcs of the red "disc minus star" region were taken for Béziers because of the Gibbs ripples near the inner corners. Fitted to the original edge points, the region came back as lines plus arcs.'),
        T('<b>讓高精度執行跑得完。</b>第一版用格子找最近點；曲線離輪廓點很遠時，格子要搜尋上萬格，第一次執行到《創造亞當》100 分鐘仍未完成，而且各圖層要等最大的輪廓算完才換下一層，20 個核心平均只用到約 2 個。改進：曲線取樣與梯度各是一次 FFT（O(S log S)）；最近點改用自寫的 k-d 樹，與暴力搜尋結果完全相同，但成本與距離無關；所有圖層的輪廓排進同一個佇列、由大到小分給各執行緒；填色器改為只檢查會經過該列的邊（其他輸出逐位元組不變）。（時間為人工觀察。）',
          '<b>Making the high-fidelity run finish.</b> The first version found nearest points with a grid; when the curve is far from the outline points, the grid searches tens of thousands of cells, and the first run was still on The Creation of Adam after 100 minutes; each layer also waited for its largest outline before the next began, so of 20 cores only about 2 were busy on average. Fixes: sampling the curve and its gradient are one FFT each (O(S log S)); nearest points come from a self-written k-d tree, identical to a brute-force search but with a cost independent of distance; the outlines of all layers share one queue, largest first, across the threads; and the rasterizer only tests the edges that can reach each row (all other outputs unchanged byte for byte). (Times are manual observations.)'),
        T('<b>人工觀察（非稽核數字）：Lab 版在瀏覽器中。</b>開發期間在先前的畫作組合（《創造亞當》與一幅後來被替換的畫）上，以 Chrome 152 開啟 *_browser.svg，用 scripts/svg_check.html 與精確轉換逐像素比較，平均 ΔE 約 1.5 至 2.1，與 8 位元模擬的量級一致，其餘來自瀏覽器與本程式不同的反鋸齒。這是一次性的外部檢查，不屬於 Rust 程式輸出。',
          '<b>Manual observation (not an audited number): the Lab version in a browser.</b> During development, on the earlier pair of paintings (The Creation of Adam and a painting since replaced), opening *_browser.svg in Chrome 152 and comparing it pixel by pixel with the exact conversion using scripts/svg_check.html gave a mean ΔE of about 1.5 to 2.1, the same order as the 8-bit simulation; the rest comes from the browser\'s anti-aliasing, which differs from ours. This was a one-off outside check, not output of the Rust program.'),
        T('<b>嘗試以面積項修正走訪順序。</b>在損失中加入區域型主動輪廓（Chan–Vese）的面積項：比較曲線圍住的面積與輪廓圍住的面積，曲線填多了就往內推、填少了就往外推。兩個問題先後出現：曲線圍住的面積等於 π Σ k|c_k|²，在目標內部面積項會獎勵高次諧波無限成長（曲線繞好幾圈來賺面積），必須限制它只作用在低次諧波；面積項在邊緣半像素內由 +1 翻成 -1，使曲線在邊界來回振盪，必須把目標區域模糊 3 px。' + region_zh,
          '<b>An attempt to fix the order with a region term.</b> The region term of a region-based active contour (Chan-Vese) was added to the loss: it compares the area the curve encloses with the area the outline encloses, pushing the curve in where it encloses too much and out where too little. Two problems came up in turn. The area a curve encloses is pi times the sum of k|c_k|², so inside the target the term rewards high harmonics for growing without bound (winding round several times to collect area), and it has to be limited to low harmonics. And the term flips from +1 to -1 within half a pixel of the edge, making the curve oscillate across the boundary, so the target has to be blurred by 3 px. ' + region_en),
    ]:
        bullet(text)
    titles = {'gleizes': T('《女人與孩子》', 'Woman and Child'), 'adam': T('《創造亞當》', 'Creation of Adam')}
    header = [T('畫作與色版', 'Painting and plate'), T('輪廓點數', 'Points'), T('內部像素', 'Inside px'), 'DFT'] + [T(f'僅下降，權重 {w:g}', f'Descent, weight {w:g}') for w in hs['region_weights']]
    table_rows = []
    for (name, plate), group in itertools.groupby(sweep, key=lambda r: (r['painting'], r['plate'])):
        group = list(group)
        cell = lambda r: T('發散', 'diverged') if r['diverged'] == '1' else f"{int(r['wrong_pixels']):,}"
        table_rows.append([f"{titles.get(name, name)} {plate}*", f"{int(group[0]['points']):,}", f"{int(group[0]['inside_pixels']):,}"] + [cell(r) for r in group])
    table([header] + table_rows, [118, 52, 64, 56] + [(W - 290) / len(hs['region_weights'])] * len(hs['region_weights']))
    story.append(p(T(f"每幅畫 a*、b* 版中最大的平滑輪廓，各種擬合填色後與輪廓本身的填色相比，內外判錯的像素數。權重 0 即只用 chamfer 損失；面積項的目標模糊 {hs['region_blur']} px。損失超過 {hs['diverged_loss']:g} px²（平均偏離 10 px）或非有限值時記為發散。",
                     f"The largest smooth outline of each painting's a* and b* plates: pixels whose inside/outside differs between each fit's fill and the outline's own fill. Weight 0 is the chamfer loss alone; the region target is blurred by {hs['region_blur']} px. A fit whose loss exceeds {hs['diverged_loss']:g} px² (10 px RMS) or is not finite counts as diverged."), caption))


TRACE_EXTENDED = [
    (T('Part 4 延伸：洞與彩色', 'Part 4 extended: holes and color'), T('holes_*、color_*、layers_summary.json（輪廓、圖元、色版、SVG、填回）', 'holes_*, color_*, layers_summary.json (outlines, primitives, ink plates, SVG, fill-back)'), ['holes_*', 'color_*', 'layers_summary.json']),
    (T('Part 4 延伸：名畫', 'Part 4 extended: paintings'), T('paint_*（輸入、標籤、量化、輪廓、圖元、SVG、填回）、paintings_summary.json；以 all --paintings 產生', 'paint_* (inputs, labels, quantized, outlines, primitives, SVG, refills), paintings_summary.json; produced with all --paintings'), ['paint_*', 'paintings_summary.json']),
    (T('Part 4 延伸：高精度', 'Part 4 extended: high fidelity'), T('hifi_*（輸入、平滑、兩種分階、兩種輪廓與三種係數的填回、逐輪廓損失與時間、面積項掃描）、hifi_summary.json；以 hifi --paintings 產生', 'hifi_* (input, smoothed, both levellings, refills of both outline sources and the three fits, per-outline losses and times, region-term sweep), hifi_summary.json; produced with hifi --paintings'), ['hifi_*']),
]

TRACE_TRANSFORMATIONS = [
    ('Transformations', T('transform_*.bmp 與 *_coverage.csv、transform_*_vertices.csv、transform_curve.csv/.svg、transform_summary.json', 'transform_*.bmp and *_coverage.csv, transform_*_vertices.csv, transform_curve.csv/.svg, transform_summary.json'), ['transform_*']),
    (T('整體', 'Overall'), T('run.json、evidence_manifest.json、verification.md、本 PDF', 'run.json, evidence_manifest.json, verification.md, this PDF'), ['run.json', 'evidence_manifest.json']),
]


def audit_transformations():
    title(T('追溯', 'Traceability'), T('Transformations · 稽核', 'Transformations · Audit'))
    table([[T('實驗', 'Experiment'), T('原始資料與成果（output/）', 'Raw data and results (output/)')]] + [[a, b] for a, b, _ in TRACE_TRANSFORMATIONS], [110, W - 110])
    add(T('與報告相同，產生器先核對 SHA-256 清單，並確認本文件讀取的每個檔案都列在上表。獨立稽核重新計算旋轉後的頂點、一步雙線性重取樣與各項誤差。',
          'As for the report, the generator checks the SHA-256 manifest first and checks that every file this document reads is listed above. The independent audit recomputes the rotated vertices, one bilinear resampling step and every error measure.'))


# ===================================================================== build ==

if args.part == 'main':
    SECTION[1] = ''
    for section in (cover, part1_curves, part1_joins, part2_fill, part2_ring, part3_supersampling, part3_row, part3_wheel,
                    part4_overview, part4_results, part4_recovery,
                    part4_extended, supp_fourier, supp_shapes, supp_holes, supp_color, supp_paintings, supp_hifi,
                    supp_related, supp_devlog, audit_main):
        section()
    check_traced(TRACE_MAIN + TRACE_EXTENDED)
    destination = OUT / 'pdf' / f'assignment_1151_parts_1_4_{args.lang}.pdf'
    build(destination, 'MULTIMEDIA SYSTEMS / ASSIGNMENT 1151', f"Parts 1-4 · source {run['source_fingerprint']}", 'Assignment 1151: Parts 1-4')
else:
    SECTION[1] = 'T'
    for section in (supp_transform, audit_transformations):
        section()
    check_traced(TRACE_TRANSFORMATIONS)
    destination = OUT / 'pdf' / f'assignment_1151_transformations_{args.lang}.pdf'
    build(destination, 'MULTIMEDIA SYSTEMS / ASSIGNMENT 1151 / TRANSFORMATIONS', f"Transformations · not submitted · source {run['source_fingerprint']}", 'Assignment 1151: Transformations')
