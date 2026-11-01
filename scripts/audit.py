"""Independent artifact checks; never used to compute the Rust render outputs.

Run: python scripts/audit.py [output-directory]
Requires Pillow only. Validation failures stop before writing a success manifest,
including under Python optimization (-O).
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from PIL import Image

def require(condition, message):
    if not condition:
        raise ValueError(message)
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'output').resolve()

def rows(name):
    with (OUT / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def data(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))

def points(name):
    return [(float(r['x']), float(r['y'])) for r in rows(name)]

def near(actual, expected, tolerance=1e-08):
    require(math.isfinite(actual) and abs(actual - expected) <= tolerance, (actual, expected))

def bernstein(p, t):
    weights = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3)
    return tuple((sum((w * v[axis] for w, v in zip(weights, p))) for axis in (0, 1)))

def segment_distance(p, a, b):
    dx, dy = (b[0] - a[0], b[1] - a[1])
    length = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length)) if length else 0.0
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)

def winding(vertices, p):
    result = 0
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])
        if a[1] <= p[1] < b[1] and cross > 0:
            result += 1
        elif b[1] <= p[1] < a[1] and cross < 0:
            result -= 1
    return result

def fingerprint():
    names = sorted(['Cargo.toml', 'Cargo.lock', 'build.rs'] + [p.relative_to(ROOT).as_posix() for p in (ROOT / 'src').glob('*.rs')])
    value = 14695981039346656037
    for name in names:
        for byte in name.encode() + b'\x00' + (ROOT / name).read_bytes() + b'\x00':
            value = (value ^ byte) * 1099511628211 & (1 << 64) - 1
    return (f'{value:016x}', names)

def coverage(name, k, size):
    values = rows(name)
    require(len(values) == size * size, 'Validation failed at line 71')
    result = []
    for i, row in enumerate(values):
        require((int(row['x']), int(row['y'])) == (i % size, i // size), 'Validation failed at line 74')
        v = float(row['coverage'])
        require(math.isfinite(v) and 0 <= v <= 1, 'Validation failed at line 76')
        near(v * k * k, round(v * k * k), 1e-07)
        result.append(v)
    return result

def continuous(name, size):
    """Coverage that came from resampling, so it is not a multiple of 1/k^2."""
    values = rows(name)
    require(len(values) == size * size, f'{name}: wrong pixel count')
    result = []
    for i, row in enumerate(values):
        require((int(row['x']), int(row['y'])) == (i % size, i // size), f'{name}: pixel order')
        v = float(row['coverage'])
        require(math.isfinite(v) and -1e-09 <= v <= 1 + 1e-09, f'{name}: coverage range')
        result.append(v)
    return result

def rotate_bilinear(values, size, center, angle):
    """Independent inverse-mapped bilinear rotation of a size x size image."""
    c, s = (math.cos(angle), math.sin(angle))
    at = lambda x, y: values[y * size + x] if 0 <= x < size and 0 <= y < size else 0.0
    result = []
    for y in range(size):
        for x in range(size):
            dx, dy = (x + 0.5 - center, y + 0.5 - center)
            u = center + c * dx + s * dy - 0.5
            v = center - s * dx + c * dy - 0.5
            x0, y0 = (math.floor(u), math.floor(v))
            fx, fy = (u - x0, v - y0)
            top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
            bottom = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
            result.append(top * (1 - fy) + bottom * fy)
    return result

def hilbert_d2xy(n, d):
    """Independent Hilbert-curve position of step d on an n x n grid."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & d // 2
        ry = 1 & (d ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = (s - 1 - x, s - 1 - y)
            x, y = (y, x)
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return (x, y)

def fft(values, inverse=False):
    """Independent recursive radix-2 FFT on Python complex numbers."""
    n = len(values)
    if n == 1:
        return list(values)
    even = fft(values[0::2], inverse)
    odd = fft(values[1::2], inverse)
    sign = 1 if inverse else -1
    result = [0j] * n
    for j in range(n // 2):
        t = complex(math.cos(sign * 2 * math.pi * j / n), math.sin(sign * 2 * math.pi * j / n)) * odd[j]
        result[j] = even[j] + t
        result[j + n // 2] = even[j] - t
    return result

def edge_points(values, size):
    """Where coverage crosses 50% between neighbouring pixel centers."""
    edges = []
    for y in range(size):
        for x in range(size):
            here = values[y * size + x]
            for nx, ny, horizontal in ((x + 1, y, True), (x, y + 1, False)):
                if nx < size and ny < size:
                    there = values[ny * size + nx]
                    if (here >= 0.5) != (there >= 0.5):
                        t = (0.5 - here) / (there - here)
                        edges.append(complex(x + 0.5 + t, y + 0.5) if horizontal else complex(x + 0.5, y + 0.5 + t))
    return edges

def dbscan(pts, eps, min_points):
    """Independent DBSCAN with brute-force neighbour search."""
    near_of = [[j for j, q in enumerate(pts) if abs(q - p) <= eps] for p in pts]
    labels = [None] * len(pts)
    visited = [False] * len(pts)
    clusters = 0
    for start in range(len(pts)):
        if visited[start]:
            continue
        visited[start] = True
        if len(near_of[start]) < min_points:
            continue
        labels[start] = clusters
        queue = list(near_of[start])
        while queue:
            j = queue.pop()
            if labels[j] is None:
                labels[j] = clusters
            if visited[j]:
                continue
            visited[j] = True
            if len(near_of[j]) >= min_points:
                queue.extend(near_of[j])
        clusters += 1
    return labels

def chamfer_loss(terms, targets, samples):
    """Symmetric chamfer loss of a Fourier curve sampled at `samples` points."""
    curve = [sum((c * complex(math.cos(k * 2 * math.pi * j / samples), math.sin(k * 2 * math.pi * j / samples)) for k, c in terms)) for j in range(samples)]
    return sum((min((abs(z - p) ** 2 for p in targets)) for z in curve)) / len(curve) + sum((min((abs(z - p) ** 2 for z in curve)) for p in targets)) / len(targets)

def primitive_samples(row):
    """Dense points along one recovered primitive, as the Rust side samples them."""
    f = lambda key: float(row[key])
    a, b = (complex(f('x0'), f('y0')), complex(f('x1'), f('y1')))
    if row['kind'] == 'line':
        n = max(1, math.ceil(abs(b - a) * 4))
        return [a + (b - a) * (i / n) for i in range(n + 1)]
    if row['kind'] == 'cubic':
        q = [a, complex(f('q1x'), f('q1y')), complex(f('q2x'), f('q2y')), b]

        def at(t):
            s = 1 - t
            return q[0] * s ** 3 + q[1] * 3 * s * s * t + q[2] * 3 * s * t * t + q[3] * t ** 3
        return [at(i / 400) for i in range(401)]
    center, radius, sweep = (complex(f('cx'), f('cy')), f('radius'), f('sweep'))
    start = math.atan2((a - center).imag, (a - center).real)
    n = max(8, math.ceil(abs(sweep) * radius * 4))
    return [center + radius * complex(math.cos(start + sweep * i / n), math.sin(start + sweep * i / n)) for i in range(n + 1)]

def check_bitmap(name, values, size, tolerance=0):
    """tolerance > 0 allows 9-decimal CSV rounding to flip an 8-bit level."""
    with Image.open(OUT / name) as image:
        require(image.format == 'BMP' and image.size == (size, size) and (image.mode == 'RGB'), 'Validation failed at line 84')
        for pixel, v in zip(image.get_flattened_data(), values):
            expected = math.floor((1 - v) * 255 + 0.5)
            require(pixel[0] == pixel[1] == pixel[2] and abs(pixel[0] - expected) <= tolerance, (name, pixel, expected))

def check_recovery(samples):
    """Shape recovery: clusters, primitives and errors against the scene."""
    summary = data('recover_summary.json')
    size = summary['canvas']
    scene = coverage('recover_scene_coverage.csv', 8, size)
    check_bitmap('recover_scene.bmp', scene, size)
    edges = edge_points(scene, size)
    saved = rows('recover_clusters.csv')
    require(len(saved) == len(edges) == summary['edge_points'], 'Recovery edge count')
    labels = dbscan(edges, summary['eps'], summary['min_points'])
    for row, e, label in zip(saved, edges, labels):
        near(float(row['x']), e.real)
        near(float(row['y']), e.imag)
        require(row['cluster'] == ('noise' if label is None else str(label)), 'DBSCAN labels')
    require(summary['clusters'] == max((l for l in labels if l is not None)) + 1 == len(summary['shapes']), 'Cluster count')
    require(summary['noise'] == labels.count(None), 'Noise count')
    # The scene as built: star, circle and D, with their true geometry.
    spin = math.radians(5)
    star_c, outer = (complex(78, 82), 56)
    inner = outer * math.cos(2 * math.pi / 5) / math.cos(math.pi / 5)
    star_corners = []
    for i in range(5):
        a = -math.pi / 2 + i * 2 * math.pi / 5 + spin
        star_corners += [star_c + outer * complex(math.cos(a), math.sin(a)), star_c + inner * complex(math.cos(a + math.pi / 5), math.sin(a + math.pi / 5))]
    truths = {'star': (star_c, star_corners, None), 'circle': (complex(192, 70), [], (complex(192, 70), 44)), 'd': (complex(110 + 16.8, 195), [complex(110, 153), complex(110, 237)], (complex(110, 195), 42))}
    primitives = rows('recover_primitives.csv')
    corners = rows('recover_corners.csv')
    coefficient_rows = rows('recover_coefficients.csv')
    losses = rows('recover_loss.csv')
    for shape in summary['shapes']:
        cid = shape['cluster']
        targets = [e for e, l in zip(edges, labels) if l == cid]
        centroid = sum(targets) / len(targets)
        name = min(truths, key=lambda n: abs(truths[n][0] - centroid))
        require(name == shape['truth'] and len(targets) == shape['edge_points'], 'Cluster matched to its shape')
        terms = [(int(r['k']), complex(float(r['re']), float(r['im']))) for r in coefficient_rows if int(r['cluster']) == cid]
        loss = chamfer_loss(terms, targets, samples)
        near(shape['fourier_loss'], loss, 1e-06)
        near(float([r for r in losses if int(r['cluster']) == cid][-1]['loss']), loss, 1e-06)
        own = [r for r in primitives if int(r['cluster']) == cid]
        require([int(r['index']) for r in own] == list(range(len(own))), 'Primitive order')
        for kind, key in (('line', 'lines'), ('arc', 'arcs'), ('cubic', 'cubics')):
            require(sum((r['kind'] == kind for r in own)) == shape[key], 'Primitive counts')
        ends = [(complex(float(r['x0']), float(r['y0'])), complex(float(r['x1']), float(r['y1']))) for r in own]
        for (_, end), (start, _) in zip(ends, ends[1:] + ends[:1]):
            require(abs(end - start) < 1e-05, 'Primitives chain into a closed outline')
        outline = [q for r in own for q in primitive_samples(r)]
        distances = [min((abs(q - p) for q in outline)) for p in targets]
        near(shape['vector_mean_edge_px'], sum(distances) / len(distances), 1e-04)
        near(shape['vector_max_edge_px'], max(distances), 1e-04)
        _, true_corners, true_circle = truths[name]
        sharp = [complex(float(r['x']), float(r['y'])) for r in corners if int(r['cluster']) == cid and r['sharp'] == 'true']
        require(len(sharp) == shape['sharp_corners'] and len(true_corners) == shape['true_corners'], 'Corner counts')
        if true_corners:
            near(shape['corner_error_px'], max((min((abs(q - t) for q in sharp)) for t in true_corners)), 1e-06)
        if true_circle:
            arcs = [r for r in own if r['kind'] == 'arc']
            best = min(arcs, key=lambda r: abs(float(r['radius']) - true_circle[1]))
            near(shape['arc_center_error_px'], abs(complex(float(best['cx']), float(best['cy'])) - true_circle[0]), 1e-05)
            near(shape['arc_radius_error_px'], abs(float(best['radius']) - true_circle[1]), 1e-05)
    rebuilt = coverage('recover_rebuilt_coverage.csv', 8, size)
    check_bitmap('recover_rebuilt.bmp', rebuilt, size)
    near(summary['rebuilt_rms'], math.sqrt(sum(((a - b) ** 2 for a, b in zip(rebuilt, scene))) / len(scene)))
    require(summary['rebuilt_misclassified'] == sum(((a >= 0.5) != (b >= 0.5) for a, b in zip(rebuilt, scene))), 'Rebuilt misclassified')
    svg = (OUT / 'recover_vector.svg').read_text(encoding='utf-8')
    require(svg.count('<path ') == summary['clusters'], 'One SVG path per shape')
    for name in ('recover_clusters.bmp', 'recover_fourier.bmp', 'recover_primitives.bmp'):
        with Image.open(OUT / name) as image:
            require(image.format == 'BMP' and image.size == (size, size), name)

def rgb_rows(name, size, height=None):
    values = rows(name)
    require(len(values) == size * (height or size), f'{name}: pixel count')
    result = []
    for i, row in enumerate(values):
        require((int(row['x']), int(row['y'])) == (i % size, i // size), f'{name}: pixel order')
        result.append(tuple((float(row[k]) for k in 'rgb')))
    return result

def check_rgb_bitmap(name, values, size, height=None):
    """8-bit BMP must be the rounded colours (one level of CSV rounding slack)."""
    with Image.open(OUT / name) as image:
        require(image.format == 'BMP' and image.size == (size, height or size) and image.mode == 'RGB', name)
        for pixel, v in zip(image.get_flattened_data(), values):
            require(all((abs(p - math.floor(max(0.0, min(1.0, c)) * 255 + 0.5)) <= 1 for p, c in zip(pixel, v))), (name, pixel, v))

def lab(c):
    """Independent sRGB -> CIE Lab (D65)."""
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    x = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return (116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))

def color_error(a, b, visible):
    square = sum(((p - q) ** 2 for u, v in zip(a, b) for p, q in zip(u, v)))
    count = sum((any((abs(p - q) > visible for p, q in zip(u, v))) for u, v in zip(a, b)))
    return (math.sqrt(square / (3 * len(a))), count)

def inside_even_odd(point, polygon):
    """Independent ray-crossing (even-odd) test, unlike the Rust winding test."""
    x, y = (point.real, point.imag)
    result = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a.imag > y) != (b.imag > y):
            if x < a.real + (y - a.imag) * (b.real - a.real) / (b.imag - a.imag):
                result = not result
    return result

def check_nesting(contour_rows, primitive_rows):
    """Depth, parent and final orientation of every traced contour."""
    layers = {}
    for row in contour_rows:
        layers.setdefault(row['layer'], []).append(row)
    for layer, contours in layers.items():
        outlines = []
        for row in contours:
            own = [r for r in primitive_rows if r['layer'] == layer and r['contour'] == row['contour']]
            require([int(r['index']) for r in own] == list(range(len(own))), 'Primitive order')
            outline = [q for r in own for q in primitive_samples(r)]
            area = sum(((a.real * b.imag - a.imag * b.real) for a, b in zip(outline, outline[1:] + outline[:1]))) / 2
            near(float(row['area']), area, 0.01)
            for kind, key in (('line', 'lines'), ('arc', 'arcs'), ('cubic', 'cubics')):
                require(sum((r['kind'] == kind for r in own)) == int(row[key]), 'Primitive counts')
            outlines.append((outline, area))
        for i, row in enumerate(contours):
            containers = [j for j, (outline, _) in enumerate(outlines) if j != i and inside_even_odd(outlines[i][0][0], outline)]
            require(int(row['depth']) == len(containers), f'Depth {layer}/{i}')
            parent = min(containers, key=lambda j: abs(outlines[j][1])) if containers else None
            require(row['parent'] == ('' if parent is None else str(parent)), f'Parent {layer}/{i}')
            final = outlines[i][1] * (-1 if row['reversed'] == 'true' else 1)
            require((final > 0) == (len(containers) % 2 == 0), f'Orientation {layer}/{i}')
    return layers

def check_layers():
    summary = data('layers_summary.json')
    size = summary['canvas']
    # Holes.
    holes = summary['holes']
    scene = coverage('holes_scene_coverage.csv', 8, size)
    check_bitmap('holes_scene.bmp', scene, size)
    contour_rows = rows('holes_contours.csv')
    layers = check_nesting(contour_rows, rows('holes_primitives.csv'))
    require(len(contour_rows) == holes['contours'], 'Hole contour count')
    depth = {int(r['contour']): int(r['depth']) for r in contour_rows}
    parent = {int(r['contour']): r['parent'] for r in contour_rows}
    outers = [s['outer'] for s in holes['shapes']]
    require(sorted(outers) == sorted((c for c, d in depth.items() if d % 2 == 0)), 'Shapes are the even-depth contours')
    for s in holes['shapes']:
        require(sorted(s['holes']) == sorted((c for c, d in depth.items() if d % 2 == 1 and parent[c] == str(s['outer']))), 'Holes belong to their outer contour')
    svg = (OUT / 'holes_vector.svg').read_text(encoding='utf-8')
    require(svg.count('<path ') == len(holes['shapes']), 'One SVG path per shape')
    for name, key in (('holes_rebuilt', 'rebuilt_misclassified'), ('holes_unoriented', 'unoriented_misclassified')):
        values = coverage(f'{name}_coverage.csv', 8, size)
        check_bitmap(f'{name}.bmp', values, size)
        require(holes[key] == sum(((a >= 0.5) != (b >= 0.5) for a, b in zip(values, scene))), key)
    # Colour scene.
    color = summary['color']
    original = rgb_rows('color_scene.csv', size)
    check_rgb_bitmap('color_scene.bmp', original, size)
    q = lambda v: math.floor(max(0.0, min(1.0, v)) * 255 + 0.5)
    counts = {}
    for c in original:
        key = tuple((q(v) for v in c))
        counts[key] = counts.get(key, 0) + 1
    kept = []
    for key, n in sorted(((k, n) for k, n in counts.items() if n >= color['min_pixels']), key=lambda item: (-item[1], item[0])):
        c = tuple((v / 255 for v in key))
        for entry in kept:
            if math.dist(lab(entry[0]), lab(c)) < color['merge_delta_e']:
                w = n / (entry[1] + n)
                entry[0] = tuple((a * (1 - w) + b * w for a, b in zip(entry[0], c)))
                entry[1] += n
                break
        else:
            kept.append([c, n])
    require(len(kept) == len(color['palette']), 'Palette size')
    for (c, n), saved in zip(kept, color['palette']):
        require(n == saved['pixels'], 'Palette counts')
        for a, b in zip(c, saved['rgb']):
            near(a, b, 1e-06)
    colors = [tuple(p['rgb']) for p in color['palette']]
    background = color['background']
    # Unmixing: every pixel split between its best pair of palette colours.
    masks = [[0.0] * (size * size) for _ in colors]
    for i, p in enumerate(original):
        best = None
        for a in range(len(colors)):
            for b in range(a + 1, len(colors)):
                d = [y - x for x, y in zip(colors[a], colors[b])]
                e = [v - x for x, v in zip(colors[a], p)]
                length = sum((v * v for v in d))
                t = max(0.0, min(1.0, sum((x * y for x, y in zip(d, e))) / length)) if length else 0.0
                err = sum(((y - x * t) ** 2 for x, y in zip(d, e)))
                if best is None or err < best[3]:
                    best = (a, b, t, err)
        masks[best[0]][i] += 1 - best[2]
        masks[best[1]][i] += best[2]
    unmixed = [colors[background]] * (size * size)
    for index, mask in enumerate(masks):
        if index != background:
            unmixed = [tuple((u * (1 - m) + c * m for u, c in zip(px, colors[index]))) for px, m in zip(unmixed, mask)]
    rms, visible = color_error(unmixed, original, color['visible'])
    near(color['palette_method']['masks_rms'], rms, 1e-06)
    require(color['palette_method']['masks_visible'] == visible, 'Unmixing visible errors')
    # Channel decomposition: CMY ink, nested level masks, exact before fitting.
    levels = color['channel_method']['levels']
    inks = []
    for channel, name in enumerate(('cyan', 'magenta', 'yellow')):
        values = {}
        for c in original:
            key = q(1 - c[channel])
            values[key] = values.get(key, 0) + 1
        merged = []
        for v, n in sorted(((v, n) for v, n in values.items() if n >= color['min_pixels'])):
            v = v / 255
            if merged and v - merged[-1][0] < color['merge_level']:
                merged[-1] = [(merged[-1][0] * merged[-1][1] + v * n) / (merged[-1][1] + n), merged[-1][1] + n]
            else:
                merged.append([v, n])
        require(len(merged) == len(levels[name]), f'{name} level count')
        for (v, _), saved in zip(merged, levels[name]):
            near(v, saved)
        ink = [0.0] * (size * size)
        for low, high in zip(levels[name], levels[name][1:]):
            for i, c in enumerate(original):
                m = max(0.0, min(1.0, (1 - c[channel] - low) / (high - low)))
                ink[i] = ink[i] * (1 - m) + high * m
        inks.append(ink)
    exact = [tuple((1 - inks[k][i] for k in range(3))) for i in range(size * size)]
    rms, visible = color_error(exact, original, color['visible'])
    near(color['channel_method']['masks_rms'], rms, 1e-06)
    require(color['channel_method']['masks_visible'] == visible, 'Channel decomposition visible errors')
    # Vectorized results.
    for method, stem, svg_name in (('palette_method', 'color_palette', 'color_palette.svg'), ('channel_method', 'color_channels', 'color_channels.svg')):
        rebuilt = rgb_rows(f'{stem}_rebuilt.csv', size)
        check_rgb_bitmap(f'{stem}_rebuilt.bmp', rebuilt, size)
        rms, visible = color_error(rebuilt, original, color['visible'])
        near(color[method]['rms'], rms, 1e-06)
        require(color[method]['visible'] == visible, f'{method} visible errors')
        svg = (OUT / svg_name).read_text(encoding='utf-8')
        require(svg.count('<path ') == color[method]['paths'] and len(svg) == color[method]['svg_bytes'], f'{svg_name} structure')
    require((OUT / 'color_channels.svg').read_text(encoding='utf-8').count('mix-blend-mode:multiply') == 3, 'Three multiply plates')
    color_rows = rows('color_contours.csv')
    check_nesting(color_rows, rows('color_primitives.csv'))
    require(sum((r['layer'].startswith('palette') for r in color_rows)) == color['palette_method']['contours'], 'Palette contours')
    require(sum((not r['layer'].startswith('palette') for r in color_rows)) == color['channel_method']['contours'], 'Channel contours')

def check_paintings():
    """Optional painting showcase: only when the run used --paintings."""
    summary = data('paintings_summary.json')
    folder = ROOT / 'assets' / 'paintings'
    pictures = {}
    radius, sigma_s, sigma_r = (summary['bilateral'][k] for k in ('radius', 'sigma_space', 'sigma_delta_e'))
    for name, info in summary['inputs'].items():
        f = info['factor']
        w, h = info['size']
        values = rgb_rows(f'paint_{name}_input.csv', w, h)
        check_rgb_bitmap(f'paint_{name}_input.bmp', values, w, h)
        with Image.open(folder / f'{name}.bmp') as source:
            require(list(source.size) == info['source'], f'{name} source size')
            x0, y0 = ((source.size[0] - w * f) // 2, (source.size[1] - h * f) // 2)
            pixels = source.load()
            # Box averages re-derived for every 97th output pixel.
            for i in range(0, w * h, 97):
                x, y = (i % w, i // w)
                block = [pixels[x0 + x * f + sx, y0 + y * f + sy] for sy in range(f) for sx in range(f)]
                for c in range(3):
                    near(values[i][c], sum((b[c] for b in block)) / (255 * f * f), 1e-06)
        labs = [lab(c) for c in values]
        smooth = rgb_rows(f'paint_{name}_smooth.csv', w, h)
        check_rgb_bitmap(f'paint_{name}_smooth.bmp', smooth, w, h)
        # Bilateral smoothing re-derived for every 211th pixel.
        for i in range(0, w * h, 211):
            x, y = (i % w, i // w)
            total, acc = (0.0, [0.0, 0.0, 0.0])
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = (x + dx, y + dy)
                    if 0 <= nx < w and 0 <= ny < h:
                        j = ny * w + nx
                        weight = math.exp(-(dx * dx + dy * dy) / (2 * sigma_s * sigma_s)) * math.exp(-math.dist(labs[i], labs[j]) ** 2 / (2 * sigma_r * sigma_r))
                        total += weight
                        for c in range(3):
                            acc[c] += weight * values[j][c]
            for c in range(3):
                near(smooth[i][c], acc[c] / total, 1e-06)
        pictures[name] = (w, h, values, labs, smooth)

    def ssim(lights, reference):
        w, h, la = reference
        total, blocks = (0.0, 0)
        for by in range(h // 8):
            for bx in range(w // 8):
                idx = [(by * 8 + k // 8) * w + bx * 8 + k % 8 for k in range(64)]
                a = [la[i] for i in idx]
                b = [lights[i] for i in idx]
                ma, mb = (sum(a) / 64, sum(b) / 64)
                va = sum(((v - ma) ** 2 for v in a)) / 64
                vb = sum(((v - mb) ** 2 for v in b)) / 64
                cov = sum(((u - ma) * (v - mb) for u, v in zip(a, b))) / 64
                total += (2 * ma * mb + 1) * (2 * cov + 9) / ((ma * ma + mb * mb + 1) * (va + vb + 9))
                blocks += 1
        return total / blocks

    def cracks(inside, w, h):
        count = 0
        for i, v in enumerate(inside):
            if not v:
                continue
            x, y = (i % w, i // w)
            for j, ok in ((i - 1, x > 0), (i + 1, x + 1 < w), (i - w, y > 0), (i + w, y + 1 < h)):
                count += 1 if not ok or not inside[j] else 0
        return count

    def regions_at_least(labels, w, h, min_area, what):
        seen = [False] * (w * h)
        for start in range(w * h):
            if seen[start]:
                continue
            seen[start] = True
            stack, size = ([start], 0)
            while stack:
                i = stack.pop()
                size += 1
                x, y = (i % w, i // w)
                for j, ok in ((i - 1, x > 0), (i + 1, x + 1 < w), (i - w, y > 0), (i + w, y + 1 < h)):
                    if ok and not seen[j] and labels[j] == labels[start]:
                        seen[j] = True
                        stack.append(j)
            require(size >= min_area, f'{what}: speck of {size} px')

    def lab_to_rgb(l, a, b):
        fy = (l + 16) / 116
        fx, fz = (fy + a / 500, fy - b / 200)
        finv = lambda v: v ** 3 if v ** 3 > 0.008856 else (v - 16 / 116) / 7.787
        x, y, z = (0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz))
        def gamma(v):
            v = 12.92 * v if v <= 0.0031308 else 1.055 * max(v, 0.0) ** (1 / 2.4) - 0.055
            return min(1.0, max(0.0, v))
        return (gamma(3.2406 * x - 1.5372 * y - 0.4986 * z), gamma(-0.9689 * x + 1.8758 * y + 0.0415 * z), gamma(0.0557 * x - 0.204 * y + 1.057 * z))
    for run in summary['runs']:
        name, stem = (run['painting'], run['stem'])
        w, h, original, original_labs, smooth = pictures[name]
        source = smooth if run['clean'] else original
        label_rows = rows(f'{stem}_labels.csv')
        require(len(label_rows) == w * h, f'{stem} labels')
        contour_rows = rows(f'{stem}_contours.csv')
        traced = {}
        for r in contour_rows:
            traced[r['layer']] = traced.get(r['layer'], 0) + int(r['edges'])
        if run['method'] == 'palette':
            labels = [int(r['label']) for r in label_rows]
            palette = run['palette']
            # Palette = mean colour of each label in the run's source picture.
            counts = [0] * len(palette)
            sums = [[0.0, 0.0, 0.0] for _ in palette]
            for l, c in zip(labels, source):
                counts[l] += 1
                for j in range(3):
                    sums[l][j] += c[j]
            for l, entry in enumerate(palette):
                require(counts[l] == entry['pixels'], f'{stem} palette counts')
                for j in range(3):
                    near(sums[l][j] / counts[l], entry['rgb'][j], 1e-08)
            if run['order'] == 'area':
                require(counts == sorted(counts, reverse=True), f'{stem} palette ordered by area')
            else:
                lights = [lab(tuple(e['rgb']))[0] for e in palette]
                require(lights == sorted(lights, reverse=True), f'{stem} palette ordered light to dark')
            regions_at_least(labels, w, h, summary['min_area'], stem)
            quantized = [tuple(palette[l]['rgb']) for l in labels]
            for colour in range(1, len(palette)):
                # Stacked layers cover their colour and every later one.
                inside = [l >= colour if run['stacked'] else l == colour for l in labels]
                require(cracks(inside, w, h) == traced.get(f'colour{colour}', 0), f'{stem} colour {colour} cracks')
            # Gap measure: nearest palette colour of each refilled pixel.
            palette_labs = [lab(tuple(e['rgb'])) for e in palette]
            rebuilt_rows = rgb_rows(f'{stem}_vector.csv', w, h)
            wrong = exposed = 0
            for c, l in zip(rebuilt_rows, labels):
                cl = lab(c)
                n = min(range(len(palette_labs)), key=lambda k: (math.dist(cl, palette_labs[k]), k))
                wrong += n != l
                exposed += n == 0 and l != 0
            require(abs(wrong - run['misassigned']) <= 2 and abs(exposed - run['background_exposed']) <= 2, f'{stem} gap counts')
        else:
            levels = run['levels']
            plates = ('cyan', 'magenta', 'yellow') if run['space'] == 'cmy' else ('lightness', 'a', 'b')
            source_labs = [lab(c) for c in source] if run['space'] == 'lab' else None
            values_q = [[0.0, 0.0, 0.0] for _ in range(w * h)]
            for k, plate in enumerate(plates):
                labels = [int(r[plate]) for r in label_rows]
                values = [1 - c[k] for c in source] if run['space'] == 'cmy' else [v[k] for v in source_labs]
                level = levels[plate]
                counts = [0] * len(level)
                sums = [0.0] * len(level)
                for l, v in zip(labels, values):
                    counts[l] += 1
                    sums[l] += v
                require(all((n > 0 for n in counts)), f'{stem} {plate} empty level')
                for l, value in enumerate(level):
                    near(sums[l] / counts[l], value, 1e-08)
                require(level == sorted(level), f'{stem} {plate} levels ordered')
                regions_at_least(labels, w, h, summary['min_area'], f'{stem} {plate}')
                for i, l in enumerate(labels):
                    values_q[i][k] = level[l]
                # Nested masks: level j and above.
                for j in range(1, len(level)):
                    inside = [l >= j for l in labels]
                    require(cracks(inside, w, h) == traced.get(f'{plate}{j}', 0), f'{stem} {plate}{j} cracks')
            if run['space'] == 'cmy':
                quantized = [tuple((1 - v for v in q)) for q in values_q]
            else:
                quantized = [lab_to_rgb(*q) for q in values_q]
            require(run['layers'] == sum((len(v) for v in levels.values())), f'{stem} layer count')
        saved_quant = rgb_rows(f'{stem}_quant.csv', w, h)
        for a, b in zip(saved_quant, quantized):
            for c in range(3):
                near(a[c], b[c], 1e-06)
        check_rgb_bitmap(f'{stem}_quant.bmp', quantized, w, h)
        require(len(contour_rows) == run['loops'], f'{stem} loop count')
        primitive_rows = rows(f'{stem}_primitives.csv')
        groups = {}
        for r in primitive_rows:
            groups.setdefault((r['layer'], r['contour']), []).append(r)
        require(len(groups) == run['loops'], f'{stem} primitives per loop')
        for kind, key in (('line', 'lines'), ('arc', 'arcs'), ('cubic', 'cubics')):
            require(sum((r['kind'] == kind for r in primitive_rows)) == run[key], f'{stem} {key}')
        for own in groups.values():
            ends = [(complex(float(r['x0']), float(r['y0'])), complex(float(r['x1']), float(r['y1']))) for r in own]
            for (_, end), (begin, _) in zip(ends, ends[1:] + ends[:1]):
                require(abs(end - begin) < 1e-05, f'{stem} open outline')
        rebuilt = rgb_rows(f'{stem}_vector.csv', w, h)
        check_rgb_bitmap(f'{stem}_vector.bmp', rebuilt, w, h)
        reference = (w, h, [v[0] for v in original_labs])
        for image, prefix in ((quantized, 'quant'), (rebuilt, 'vector')):
            image_labs = [lab(c) for c in image]
            d = [math.dist(a, b) for a, b in zip(image_labs, original_labs)]
            near(sum(d) / len(d), run[f'{prefix}_mean_delta_e'], 1e-05)
            # Inputs come back from 9-decimal CSV: a pixel sitting exactly on
            # the threshold may flip, so allow two of them.
            require(abs(sum((v > summary['visible_delta_e'] for v in d)) - run[f'{prefix}_visible']) <= 2, f'{stem} {prefix}_visible')
            near(ssim([v[0] for v in image_labs], reference), run[f'{prefix}_ssim'], 1e-06)
        svg = (OUT / f'{stem}.svg').read_text(encoding='utf-8')
        require(svg.count('<path ') == run['shapes'] and len(svg) == run['svg_bytes'] and run['bitmap_bytes'] == 3 * w * h, f'{stem} SVG')
        if run['method'] == 'channels' and run['space'] == 'cmy':
            require(svg.count('mix-blend-mode:multiply') == 3, f'{stem} three multiply plates')
        if run['method'] == 'channels' and run['space'] == 'lab':
            require(all((f'<g id="{p}">' in svg for p in ('lightness', 'a', 'b'))), f'{stem} three Lab plates')
            # Browser SVG: plates lit into R, G, B and rebuilt by a filter.
            browser = (OUT / f'{stem}_browser.svg').read_text(encoding='utf-8')
            require(len(browser) == run['browser_svg_bytes'] and browser.count('<path ') == run['shapes'], f'{stem} browser SVG paths')
            require('<filter id="lab"' in browser and browser.count('mix-blend-mode:screen') == 3 and browser.count('<feFunc') == 6, f'{stem} browser SVG filter')
            plate_rows = rows(f'{stem}_plates.csv')
            require(len(plate_rows) == w * h, f'{stem} plates')
            M1 = [[100 / 116, 255 / 500, 0, 16 / 116 - 128 / 500], [100 / 116, 0, 0, 16 / 116], [100 / 116, 0, -255 / 200, 16 / 116 + 128 / 200]]
            M2 = [[3.2406 * 0.95047, -1.5372, -0.4986 * 1.08883], [-0.9689 * 0.95047, 1.8758, 0.0415 * 1.08883], [0.0557 * 0.95047, -0.204, 1.057 * 1.08883]]

            def chain(l, a, b, eight):
                def clamp(v):
                    v = min(1.0, max(0.0, v))
                    return round(v * 255) / 255 if eight else v
                e = [clamp(l / 100), clamp((a + 128) / 255), clamp((b + 128) / 255)]
                f = [clamp(m[0] * e[0] + m[1] * e[1] + m[2] * e[2] + m[3]) for m in M1]
                xyz = [clamp(v ** 3) for v in f]
                return tuple((clamp(1.055 * clamp(m[0] * xyz[0] + m[1] * xyz[1] + m[2] * xyz[2]) ** (1 / 2.4) - 0.055) for m in M2))
            total = worst = total8 = 0.0
            for i, (row, c) in enumerate(zip(plate_rows, rebuilt)):
                l, a, b = (float(row['l']), float(row['a']), float(row['b']))
                cl = lab(c)
                if i % 13 == 0:
                    exact = lab_to_rgb(l, a, b)
                    require(all((abs(x - y) < 1e-06 for x, y in zip(exact, c))), f'{stem} plates rebuild')
                d = math.dist(lab(chain(l, a, b, False)), cl)
                total += d
                worst = max(worst, d)
                total8 += math.dist(lab(chain(l, a, b, True)), cl)
            near(total / (w * h), run['filter_mean_delta_e'], 1e-04)
            near(worst, run['filter_max_delta_e'], 1e-03)
            near(total8 / (w * h), run['filter_8bit_mean_delta_e'], 1e-03)

def check_hifi():
    """Optional high-fidelity run (--hifi): sizes, smoothing, levels, loop table and measures."""
    summary = data('hifi_summary.json')
    require(summary['source_fingerprint'] == fingerprint()[0], 'hi-fi run is from other sources')
    folder = ROOT / 'assets' / 'paintings'
    images = ['quant_raw', 'crack_dft', 'crack_refined', 'quant', 'dft', 'refined', 'descent']
    variants = ['dft', 'refined', 'descent']
    for name, info in summary['paintings'].items():
        f = info['factor']
        w, h = info['size']
        decoded = {}
        for variant in ['input', 'smooth'] + images:
            with Image.open(OUT / f'hifi_{name}_{variant}.bmp') as image:
                require(image.format == 'BMP' and image.size == (w, h), f'hifi {name} {variant} size')
                raw = image.convert('RGB').tobytes()
                decoded[variant] = [(raw[k] / 255, raw[k + 1] / 255, raw[k + 2] / 255) for k in range(0, len(raw), 3)]
        with Image.open(folder / f'{name}.bmp') as source:
            require(list(source.size) == info['source'], f'hifi {name} source size')
            x0, y0 = ((source.size[0] - w * f) // 2, (source.size[1] - h * f) // 2)
            pixels = source.load()

            def exact(x, y):
                block = [pixels[x0 + x * f + sx, y0 + y * f + sy] for sy in range(f) for sx in range(f)]
                return tuple(sum(b[c] for b in block) / (255 * f * f) for c in range(3))
            # Box averages, rounded to 8 bits, for every 89th pixel.
            for i in range(0, w * h, 89):
                value = exact(i % w, i // w)
                for c in range(3):
                    require(abs(decoded['input'][i][c] - value[c]) <= 0.5 / 255 + 1e-09, f'hifi {name} box average')
            # Bilateral smoothing re-derived from the exact averages for every 997th pixel.
            radius, sigma_s, sigma_r = (2, 2.0, 8.0)
            for i in range(0, w * h, 997):
                x, y = (i % w, i // w)
                centre = exact(x, y)
                lc = lab(centre)
                total, acc = (0.0, [0.0, 0.0, 0.0])
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = (x + dx, y + dy)
                        if 0 <= nx < w and 0 <= ny < h:
                            v = exact(nx, ny)
                            weight = math.exp(-(dx * dx + dy * dy) / (2 * sigma_s * sigma_s)) * math.exp(-math.dist(lc, lab(v)) ** 2 / (2 * sigma_r * sigma_r))
                            total += weight
                            for c in range(3):
                                acc[c] += weight * v[c]
                for c in range(3):
                    require(abs(decoded['smooth'][i][c] - acc[c] / total) <= 0.5 / 255 + 1e-06, f'hifi {name} bilateral')
        for plate, count in zip(('lightness', 'a', 'b'), summary['levels']):
            raw_levels = info['raw_levels'][plate]
            require(len(raw_levels) <= count and raw_levels == sorted(raw_levels), f'hifi {name} {plate} raw levels')
            levels, thresholds = (info['smooth_levels'][plate]['levels'], info['smooth_levels'][plate]['thresholds'])
            require(len(levels) <= count and levels == sorted(levels) and len(thresholds) == len(levels) - 1, f'hifi {name} {plate} levels')
            for t, a, b in zip(thresholds, levels, levels[1:]):
                near(t, (a + b) / 2, 1e-08)
        # Measures recomputed from the decoded bitmaps.
        reference = [lab(c) for c in decoded['input']]
        lights = [v[0] for v in reference]
        for variant in images:
            labs = [lab(c) for c in decoded[variant]]
            d = [math.dist(u, v) for u, v in zip(labs, reference)]
            measure = info['measures'][variant]
            near(measure['mean_delta_e'], sum(d) / len(d), 1e-04)
            visible = sum(v > summary['visible_delta_e'] for v in d)
            require(abs(measure['visible'] - visible) <= 2, f'hifi {name} {variant} visible')
            total, blocks = (0.0, 0)
            light = [v[0] for v in labs]
            for by in range(h // 8):
                for bx in range(w // 8):
                    idx = [(by * 8 + k // 8) * w + bx * 8 + k % 8 for k in range(64)]
                    a = [light[i] for i in idx]
                    b = [lights[i] for i in idx]
                    ma, mb = (sum(a) / 64, sum(b) / 64)
                    va = sum((v - ma) ** 2 for v in a) / 64
                    vb = sum((v - mb) ** 2 for v in b) / 64
                    cov = sum((u - ma) * (v - mb) for u, v in zip(a, b)) / 64
                    total += (2 * ma * mb + 1) * (2 * cov + 9) / ((ma * ma + mb * mb + 1) * (va + vb + 9))
                    blocks += 1
            near(measure['ssim'], total / blocks, 1e-04)
        # Loop table: harmonic and sample counts follow the rule, and the
        # summary totals are its sums.
        table = rows(f'hifi_{name}_loops.csv')
        for family, used in (('crack', variants[:2]), ('smooth', variants)):
            part = [r for r in table if r['outline'] == family]
            stats = info[family]
            require(len(part) == stats['loops'], f'hifi {name} {family} loop count')
            require(len({(r['plate'], r['level']) for r in part}) == stats['masks'], f'hifi {name} {family} masks')
            n_total, coefficients = (0, 0)
            weighted = [0.0] * len(used)
            stuck = 0
            for row in part:
                n, m, samples = (int(row['n']), int(row['harmonics']), int(row['samples']))
                require(m == min(max(n // 3, 2), summary['max_harmonics']), f'hifi {name} harmonics')
                size = max(2 * n, 4 * m + 2)
                require(samples == max(64, 1 << (size - 1).bit_length()), f'hifi {name} samples')
                losses = [float(row['loss_' + v]) for v in used]
                require(all(math.isfinite(v) and v >= 0 for v in losses), f'hifi {name} losses')
                require(all(row['loss_' + v] == '' for v in variants[len(used):]), f'hifi {name} empty cells')
                n_total += n
                coefficients += 2 * m + 1
                for v in range(len(used)):
                    weighted[v] += n * math.sqrt(losses[v])
                if family == 'smooth':
                    stuck += losses[2] > summary['stuck_factor'] * losses[0] + summary['stuck_allowance']
            require(n_total == stats['edge_points'] and coefficients == stats['coefficients'], f'hifi {name} {family} totals')
            for v, key in enumerate(used):
                near(stats['edge_rms'][key], weighted[v] / n_total, 1e-05)
            if family == 'smooth':
                require(stuck == info['descent_stuck'], f'hifi {name} stuck count')
        require(info['bitmap_bytes'] == 3 * w * h, f'hifi {name} bitmap bytes')
    # Region sweep: per painting, the largest smooth a* and b* outlines, the
    # DFT and descent at every weight; the diverged flag follows the stated
    # rule and only diverged fits lack a pixel count (counts not re-derived).
    sweep = rows('hifi_region_sweep.csv')
    for name in summary['paintings']:
        for plate in ('a', 'b'):
            part = [r for r in sweep if r['painting'] == name and r['plate'] == plate]
            require([(r['fit'], float(r['weight'])) for r in part] == [('dft', 0.0)] + [('descent', w) for w in summary['region_weights']], f'hifi sweep {name} {plate} rows')
            require(len({(r['level'], r['points'], r['harmonics'], r['inside_pixels']) for r in part}) == 1, f'hifi sweep {name} {plate} outline')
            require(int(part[0]['harmonics']) == min(max(int(part[0]['points']) // 3, 2), summary['max_harmonics']), f'hifi sweep {name} {plate} harmonics')
            for r in part:
                loss = float(r['loss'])
                diverged = not math.isfinite(loss) or loss > summary['diverged_loss']
                require(r['diverged'] == str(int(diverged)) and (r['wrong_pixels'] == '') == diverged, f'hifi sweep {name} {plate} diverged flag')
                size = summary['paintings'][name]['size']
                require(diverged or 0 <= int(r['wrong_pixels']) <= size[0] * size[1], f'hifi sweep {name} {plate} count')

def main():
    checks = []
    run = data('run.json')
    require(run['status'] == 'complete', 'Validation failed at line 93')
    identity, inputs = fingerprint()
    require(identity == run['source_fingerprint'], 'Rebuild and rerun: artifacts do not match current Rust sources')
    size = run['canvas']
    checks.append('Build-input fingerprint matches current sources and complete run manifest')
    with Image.open(OUT / 'diagnostic.bmp') as image:
        require(image.size == (3, 2) and image.mode == 'RGB', 'Validation failed at line 99')
        require(list(image.get_flattened_data()) == [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)], 'Validation failed at line 100')
    checks.append('Independent Pillow BMP decode: padded 3x2 diagnostic, six exact colors and row orientation')
    controls = points('curve_controls.csv')
    samples = points('curve_formula.csv')
    summary = data('curve_summary.json')
    require(len(samples) == summary['formula_samples'], 'Validation failed at line 105')
    for i, point in enumerate(samples):
        expected = bernstein(controls, i / (len(samples) - 1))
        near(point[0], expected[0])
        near(point[1], expected[1])
    spacing = max((math.dist(a, b) for a, b in zip(samples, samples[1:])))
    require(spacing < 0.1, 'Validation failed at line 110')
    near(spacing, summary['max_spacing_px'])
    convergence = rows('convergence.csv')
    for row in convergence:
        depth = int(row['depth'])
        n = 2 ** depth
        require(int(row['segments']) == n, 'Validation failed at line 116')
        vertices = [bernstein(controls, i / n) for i in range(n + 1)]
        gap = max((min((segment_distance(p, a, b) for a, b in zip(vertices, vertices[1:]))) for p in samples))
        near(gap, float(row['max_gap_px']), 2e-08)
    shallow = points('curve_depth2.csv')
    require(len(shallow) == 5, 'Validation failed at line 121')
    for i, p in enumerate(shallow):
        require(math.dist(p, bernstein(controls, i / 4)) < 1e-08, 'Validation failed at line 123')
    threshold = next((int(r['depth']) for r in convergence if float(r['max_gap_px']) < 0.5))
    require(threshold == summary['first_depth_below_half_pixel'], 'Validation failed at line 125')
    checks.append('All formula samples and all convergence errors independently recomputed with Bernstein and projected segment distance (2e-8 px tolerance)')
    metrics = rows('joins_metrics.csv')
    smooth = [r for r in metrics if r['join'] == '0']
    corner = [r for r in metrics if r['join'] == '1']
    for row in smooth:
        require(abs(float(row['tangent_cross'])) < 1e-08, 'Validation failed at line 131')
    near(float(smooth[0]['incoming_length']), float(smooth[0]['outgoing_length']))
    near(float(smooth[1]['outgoing_length']) / float(smooth[1]['incoming_length']), 3.0)
    require(all((float(r['tangent_angle_deg']) > 45 for r in corner)), 'Validation failed at line 134')
    checks.append('Join metrics: equal smooth handles; unequal ratio 3; visible non-collinear corner')
    star = points('star_vertices.csv')
    require(len(star) == 5 and winding(star, (size / 2, size / 2)) == data('star_summary.json')['winding'] == 2, 'Validation failed at line 137')
    with Image.open(OUT / 'star.bmp') as image:
        require(image.size == (size, size) and image.getpixel((size // 2, size // 2)) == (0, 0, 0), 'Validation failed at line 139')
    ring = data('ring_summary.json')
    ring_points = rows('ring_controls.csv')
    for subpath in ('0', '1'):
        variants = [sorted(((r['x'], r['y']) for r in ring_points if r['variant'] == variant and r['subpath'] == subpath)) for variant in ('same', 'opposite')]
        require(variants[0] == variants[1], 'Validation failed at line 144')
    require(ring['positions_unchanged'] and ring['same_winding'] == 2 and (ring['opposite_winding'] == 0), 'Validation failed at line 145')
    for name, color in (('ring_same.bmp', (0, 0, 0)), ('ring_opposite.bmp', (255, 255, 255))):
        with Image.open(OUT / name) as image:
            require(image.size == (size, size) and image.getpixel((size // 2, size // 2)) == color, 'Validation failed at line 148')
    checks.append('Five-vertex star independently has winding +2; ring coordinate multisets unchanged and decoded center colors match direction')
    aa = data('aa_summary.json')
    rotated = points('rotated_star_vertices.csv')
    edges = list(zip(rotated, rotated[1:] + rotated[:1]))
    require(all((abs(a[0] - b[0]) > 1e-09 and abs(a[1] - b[1]) > 1e-09 for a, b in edges)), 'Validation failed at line 153')
    images = {}
    for k in (1, 2, 4, 8):
        values = coverage(f'star_k{k}_coverage.csv', k, size)
        check_bitmap(f'star_k{k}.bmp', values, size)
        images[k] = values
    for y in range(3, size, 17):
        for x in range(3, size, 17):
            require(images[1][y * size + x] == float(winding(rotated, (x + 0.5, y + 0.5)) != 0), 'Validation failed at line 162')
    checks.append('All four coverage arrays checked for range, discrete sample fractions, BMP quantization and dimensions; k=1 checked against independent ray test grid')
    selection = data('row_selection.json')
    chosen = min(range(5), key=lambda i: abs((edges[i][1][1] - edges[i][0][1]) / (edges[i][1][0] - edges[i][0][0])))
    require(chosen == selection['edge_index'], 'Validation failed at line 166')
    require(min(selection['fraction_from_start'], 1 - selection['fraction_from_start']) <= 0.25, 'Validation failed at line 167')
    crop = rows('row_samples.csv')
    require(len(crop) == selection['x_end_exclusive'] - selection['x_start'], 'Validation failed at line 169')
    for row in crop:
        x, y = (int(row['x']), int(row['y']))
        require(y == selection['row'], 'Validation failed at line 172')
        for k in (1, 8):
            near(float(row[f'coverage_k{k}']), images[k][y * size + x])
    require(sum((0 < float(r['coverage_k8']) < 1 for r in crop)) == selection['partial_pixels_k8'] >= 3, 'Validation failed at line 174')
    require({float(r['coverage_k8']) for r in crop}.issuperset({0.0, 1.0}), 'Validation failed at line 175')
    row_y = selection['row'] + 0.5
    for i, (a, b) in enumerate(edges):
        if i != chosen and min(a[1], b[1]) <= row_y < max(a[1], b[1]):
            cross = a[0] + (b[0] - a[0]) * (row_y - a[1]) / (b[1] - a[1])
            require(not selection['x_start'] - 0.5 <= cross <= selection['x_end_exclusive'] + 0.5, 'Validation failed at line 180')
    checks.append('Row is near an endpoint of the shallowest edge, isolates its crossing, and exactly matches unquantized source samples')
    trials = rows('timing_trials.csv')
    for row in rows('timing.csv'):
        values = [float(r['render_ms']) for r in trials if r['k'] == row['k']]
        require(len(values) == aa['timing_repeats'] == run['timing_repeats'] == 3, 'Validation failed at line 185')
        near(statistics.median(values), float(row['median_render_ms']))
        require(int(row['width']) == int(row['height']) == size and all((v > 0 for v in values)), 'Validation failed at line 187')
    checks.append('Timing medians recomputed from all three trials per factor, same canvas')
    wheel = data('wheel_summary.json')
    wedges = wheel['wedges']
    require(wedges == run['wedges'] == wheel['subpaths'] and wedges >= 40, 'Validation failed at line 191')
    near(wheel['wedge_angle'], math.pi / wedges)
    near(wheel['gap_angle'], math.pi / wedges)
    triangles = rows('wheel_triangles.csv')
    require(len(triangles) == wedges * 3, 'Validation failed at line 194')
    for i in range(wedges):
        triangle = triangles[3 * i:3 * i + 3]
        ps = [(float(r['x']), float(r['y'])) for r in triangle]
        require(ps[0] == (size / 2, size / 2), 'Validation failed at line 198')
        angles = [wheel['phase_radians'] + 2 * i * math.pi / wedges, wheel['phase_radians'] + (2 * i + 1) * math.pi / wedges]
        for p, angle in zip(ps[1:], angles):
            near(p[0], size / 2 + wheel['radius'] * math.cos(angle))
            near(p[1], size / 2 + wheel['radius'] * math.sin(angle))
    for k in (1, 8):
        values = coverage(f'wheel_k{k}_coverage.csv', k, size)
        check_bitmap(f'wheel_k{k}.bmp', values, size)
        half = wheel['center_roi_half_width']
        roi = [values[y * size + x] for y in range(size // 2 - half, size // 2 + half) for x in range(size // 2 - half, size // 2 + half)]
        near(statistics.mean(roi), wheel[f'center_mean_k{k}'])
        near(math.sqrt(statistics.mean(((v - 0.5) ** 2 for v in roi))), wheel[f'center_rms_from_half_k{k}'])
    checks.append('Wheel triangles independently checked against equal angular wedges/gaps; both coverage arrays, decoded BMPs and center statistics checked')
    transform = data('transform_summary.json')
    require(transform['canvas'] == size and transform['k'] == 8 and transform['steps'] == 100 and transform['step_degrees'] == 1, 'Part 4 parameters')
    center = size / 2
    angle = math.radians(100)
    ring_angles = [-math.pi / 2 + i * 2 * math.pi / 5 for i in (0, 2, 4, 1, 3)]
    expected = []
    for a in ring_angles:
        px, py = 0.42 * size * math.cos(a), 0.42 * size * math.sin(a)
        expected.append((center + math.cos(angle) * px - math.sin(angle) * py, center + math.sin(angle) * px + math.cos(angle) * py))
    direct = points('transform_direct_vertices.csv')
    chained = points('transform_vector_vertices.csv')
    require(len(direct) == len(chained) == 5, 'Part 4 vertex count')
    for d, c, e in zip(direct, chained, expected):
        require(math.dist(d, e) < 1e-08 and math.dist(c, d) < 1e-09, 'Part 4 vertices')
    require(transform['vertex_max_drift_px'] < 1e-09, 'Part 4 vertex drift')
    maps = {}
    for name in ('base', 'reference', 'vector'):
        maps[name] = coverage(f'transform_{name}_coverage.csv', 8, size)
        check_bitmap(f'transform_{name}.bmp', maps[name], size)
    for name in ('bitmap', 'once', 'step1'):
        maps[name] = continuous(f'transform_{name}_coverage.csv', size)
    for name in ('bitmap', 'once'):
        check_bitmap(f'transform_{name}.bmp', maps[name], size, 1)
    check_bitmap('transform_error.bmp', [abs(b - r) for b, r in zip(maps['bitmap'], maps['reference'])], size, 1)
    require(maps['vector'] == maps['reference'] and transform['vector_rms'] == transform['vector_max_abs'] == 0, 'Part 4 vector route')
    step = rotate_bilinear(maps['base'], size, center, math.radians(1))
    require(max((abs(a - b) for a, b in zip(step, maps['step1']))) < 1e-08, 'Part 4 independent bilinear step')
    half_level = 0.5 / 255
    gray = lambda values: sum((half_level <= v < 1 - half_level for v in values))
    for name in ('bitmap', 'once'):
        diffs = [a - b for a, b in zip(maps[name], maps['reference'])]
        near(math.sqrt(sum((d * d for d in diffs)) / len(diffs)), transform[f'{name}_rms'])
        near(max((abs(d) for d in diffs)), transform[f'{name}_max_abs'])
    for name in ('reference', 'vector', 'bitmap', 'once'):
        require(gray(maps[name]) == transform[f'partial_{name}'], 'Part 4 gray pixel count')
    near(sum(maps['bitmap']), transform['area_bitmap'], 1e-05)
    near(sum(maps['reference']), transform['area_reference'], 1e-05)
    curve = rows('transform_curve.csv')
    require([int(r['steps']) for r in curve] == list(range(1, 101)), 'Part 4 curve steps')
    near(float(curve[-1]['rms']), transform['bitmap_rms'])
    require(int(curve[-1]['partial_pixels']) == transform['partial_bitmap'] and int(curve[-1]['reference_partial_pixels']) == transform['partial_reference'], 'Part 4 curve end')
    checks.append('Part 4 supplement (Transformations): rotated vertices recomputed independently; vector route equals direct rendering; one bilinear resampling step independently recomputed; RMS, maximum error, gray-pixel counts, area and curve end recomputed')
    vec = data('vectorize_summary.json')
    vsize = vec['canvas']
    require(vsize == 256 and vec['k'] == 8 and vec['signal_length'] == vsize * vsize, 'Vectorize parameters')
    original = coverage('vectorize_original_coverage.csv', 8, vsize)
    check_bitmap('vectorize_original.bmp', original, vsize)
    vmetrics = rows('vectorize_metrics.csv')
    rms_of = lambda values: math.sqrt(sum(((a - b) ** 2 for a, b in zip(values, original))) / len(original))
    wrong_of = lambda values: sum(((a >= 0.5) != (b >= 0.5) for a, b in zip(values, original)))
    # A: Hilbert order, signal, and every reconstruction from an independent FFT.
    signal_rows = rows('vectorize_signal.csv')
    order = [hilbert_d2xy(vsize, d) for d in range(vsize * vsize)]
    require(len(set(order)) == vsize * vsize and all((abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(order, order[1:]))), 'Hilbert curve')
    signal = []
    for d, row in enumerate(signal_rows):
        x, y = order[d]
        require((int(row['index']), int(row['x']), int(row['y'])) == (d, x, y), 'Hilbert order')
        near(float(row['value']), original[y * vsize + x])
        signal.append(original[y * vsize + x])
    spectrum = fft([complex(v) for v in signal])
    length = len(spectrum)
    energy = lambda f: abs(spectrum[f]) ** 2 * (1 if f in (0, length // 2) else 2)
    ranked = sorted(range(length // 2 + 1), key=lambda f: (-energy(f), f))
    for budget in vec['a_budgets']:
        kept = [0j] * length
        for f in ranked[:budget]:
            kept[f] = spectrum[f]
            if f not in (0, length // 2):
                kept[length - f] = spectrum[length - f]
        back = fft(kept, True)
        expected = [0.0] * (vsize * vsize)
        for (x, y), value in zip(order, back):
            expected[y * vsize + x] = min(1.0, max(0.0, value.real / length))
        saved = continuous(f'vectorize_a_n{budget}_coverage.csv', vsize)
        require(max((abs(a - b) for a, b in zip(saved, expected))) < 1e-06, f'Hilbert reconstruction N={budget}')
        check_bitmap(f'vectorize_a_n{budget}.bmp', saved, vsize, 1)
        row = next((r for r in vmetrics if r['method'] == 'hilbert' and int(r['budget']) == budget))
        near(float(row['rms']), rms_of(saved))
        require(int(row['misclassified']) == wrong_of(saved), f'Hilbert misclassified N={budget}')
    # B: edge points, then curve metrics recomputed from the saved coefficients.
    edges = edge_points(original, vsize)
    saved_edges = points('vectorize_edge_points.csv')
    require(len(saved_edges) == len(edges) == vec['edge_points'], 'Edge point count')
    for p, e in zip(saved_edges, edges):
        near(p[0], e.real)
        near(p[1], e.imag)
    center, outer = (vsize / 2, 0.42 * vsize)
    inner = outer * math.cos(2 * math.pi / 5) / math.cos(math.pi / 5)
    near(inner / outer, vec['inner_radius_ratio'])
    spin = math.radians(5)
    tips = [complex(center, center) + outer * complex(math.cos(a), math.sin(a)) for a in (-math.pi / 2 + i * 2 * math.pi / 5 + spin for i in range(5))]
    corners = [complex(center, center) + inner * complex(math.cos(a), math.sin(a)) for a in (-math.pi / 2 + i * 2 * math.pi / 5 + math.pi / 5 + spin for i in range(5))]
    coefficient_rows = rows('vectorize_b_coefficients.csv')
    losses = rows('vectorize_b_loss.csv')
    for m in vec['b_harmonics']:
        terms = [(int(r['k']), complex(float(r['re']), float(r['im']))) for r in coefficient_rows if int(r['harmonics']) == m]
        require([k for k, _ in terms] == list(range(-m, m + 1)), f'Coefficients M={m}')
        curve = lambda count: [sum((c * complex(math.cos(k * 2 * math.pi * j / count), math.sin(k * 2 * math.pi * j / count)) for k, c in terms)) for j in range(count)]
        samples = curve(vec['b_samples'])
        loss = sum((min((abs(z - p) ** 2 for p in edges)) for z in samples)) / len(samples) + sum((min((abs(z - p) ** 2 for z in samples)) for p in edges)) / len(edges)
        row = next((r for r in vmetrics if r['method'] == 'fourier_fit' and int(r['harmonics']) == m))
        near(float(row['final_loss']), loss, 1e-06)
        stage = [r for r in losses if int(r['harmonics']) == m]
        require(len(stage) == vec['b_iterations_per_stage'] + 1, f'Loss rows M={m}')
        near(float(stage[-1]['loss']), loss, 1e-06)
        dense = curve(4096)
        near(float(row['tip_px']), max((min((abs(z - p) for z in dense)) for p in tips)), 1e-06)
        near(float(row['corner_px']), max((min((abs(z - p) for z in dense)) for p in corners)), 1e-06)
        values = coverage(f'vectorize_b_m{m}_coverage.csv', 8, vsize)
        check_bitmap(f'vectorize_b_m{m}.bmp', values, vsize)
        near(float(row['rms']), rms_of(values))
        require(int(row['misclassified']) == wrong_of(values), f'Fit misclassified M={m}')
    for name in ['vectorize_hilbert_order.bmp', 'vectorize_epicycles_m8.bmp'] + [f'vectorize_b_overlay_m{m}.bmp' for m in vec['b_harmonics']]:
        with Image.open(OUT / name) as image:
            require(image.format == 'BMP' and image.size == (vsize, vsize), name)
    check_recovery(vec['b_samples'])
    check_layers()
    if (OUT / 'paintings_summary.json').exists():
        check_paintings()
        checks.append('Part 4 supplement (paintings): box averages and bilateral smoothing sampled against the decoded BMPs; palette means and order, CMY and Lab channel levels and minimum region size re-derived from the labels; quantized colours re-derived (Lab back to RGB independently); every cut-out, stacked or channel-level boundary crack found on a traced loop; gap counts (background exposed) recomputed; browser Lab SVG structure checked and its filter chain (exact and 8-bit) recomputed from the plate values; outlines closed; colour differences, lightness SSIM, quantized and vector bitmaps and SVG structure recomputed (DBSCAN cluster counts not re-derived)')
    if (OUT / 'hifi_summary.json').exists():
        check_hifi()
        checks.append('Part 4 supplement (high fidelity, --hifi): run fingerprint matches the sources; box averages and bilateral smoothing sampled against the source BMPs; level counts and order, thresholds at the midpoints; every mean colour difference, visible count and lightness SSIM recomputed from the decoded bitmaps of all seven variants; per-outline harmonic and sample counts follow the stated rule, and the totals, edge RMS and stuck counts of both outline families are the loop table sums; region-sweep table complete, with its diverged flags following the stated rule (pixel counts not re-derived)')
    checks.append('Part 4 supplement (holes and colour): contour depth, parent and orientation re-derived with an independent even-odd test; palette, unmixing and CMY level decomposition recomputed from the scene; every refill, error count and SVG structure recomputed')
    checks.append('Part 4 (corner recovery): edge points and an independent brute-force DBSCAN reproduce the clusters; primitives chain into closed outlines; Fourier losses, corner, arc and edge-distance errors against the known shapes, and the refilled bitmap recomputed')
    checks.append('Part 4 (vectorizing the star): Hilbert curve bijective with unit steps; scan, independent FFT and every strongest-N reconstruction recomputed; edge points, chamfer losses, tip/corner errors, RMS and misclassified counts recomputed from saved coefficients and coverage')
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in inputs}
    artifact_hashes = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('evidence_manifest.json', 'verification.md')}
    manifest = {'status': 'passed', 'run': run, 'checks': checks, 'source_sha256': source_hashes, 'artifact_sha256': artifact_hashes, 'decoder': 'Pillow ' + Image.__version__, 'tolerances': {'geometry_px': 2e-08, 'coverage': 1e-08}}
    (OUT / 'evidence_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'verification.md').write_text('# Independent artifact audit\n\nSource fingerprint: `' + identity + '`\n\n' + '\n'.join(('- PASS: ' + v for v in checks)) + '\n', encoding='utf-8')
    print(f'PASS: {len(checks)} independent checks; {len(artifact_hashes)} artifact hashes; source {identity}')
if __name__ == '__main__':
    main()
