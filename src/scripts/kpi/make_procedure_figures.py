"""의뢰시험 절차서 첨부 이미지 생성기 (공인시험용 패키지 snu-ai-control-kpi 기준).

시험용 PC에서 절차서의 명령을 실제로 실행하고, 그 출력을 터미널 화면 형태의 PNG로 그린다.
시험 프로그램 실행 화면(IMG-05~07)은 한 번 실행한 출력(<패키지>/reference_results/console.txt)을 나눠 담는다.
구성도(IMG-00)는 흐름도로 그리고, 그래프(IMG-09, IMG-10)는 패키지에 저장된 PNG를 복사한다.
확인할 측정값은 빨간 상자로 표시한다. 이미지는 내부 문서용이므로 저장소의 reports/figures/에 저장한다.

사용 예 (저장소 루트에서):
    python src/scripts/kpi/make_procedure_figures.py            # 전체
    python src/scripts/kpi/make_procedure_figures.py --only 00 01
"""

import argparse
import re
import shutil
import socket
import subprocess
import unicodedata
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
PKG = REPO.parent / 'snu-ai-control-kpi'      # 공인시험용 패키지 (시험기관에 보여 주는 폴더)
RESULT_DIR = PKG / 'reference_results'        # 사전 측정 결과 (bash run_test.sh 출력 폴더를 옮겨 둔 것)
RESULT_SHOWN = '../snu-ai-control-kpi-results'  # 그림에는 실행 직후의 결과 폴더(패키지 밖)로 표시
TEST_CMD = 'bash run_test.sh'                 # 공인시험용 실행 스크립트 (IMG-05에 표시)
OUT_DIR = REPO / 'reports' / 'figures'

TTF = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
KOREAN_FONT = PKG / 'fonts' / 'NanumGothic-Regular.ttf'
MONO_FONT = TTF / 'DejaVuSansMono.ttf'
SANS_FONT = TTF / 'DejaVuSans.ttf'
MONO_BOLD = TTF / 'DejaVuSansMono-Bold.ttf'
# 서버의 어떤 글꼴에도 없는 호환용 단위 문자는 같은 뜻의 두 글자로 그린다 (폭 2칸 유지).
GLYPH_SUBSTITUTES = {'㎍': 'μg', '㎥': 'm³'}
PKG_CMD = ('python -c "import torch, numpy, pandas, sklearn, yaml, matplotlib; '
           "[print(f'{n:<14}{v}') for n, v in [('torch', torch.__version__), ('numpy', numpy.__version__), "
           "('pandas', pandas.__version__), ('scikit-learn', sklearn.__version__), ('PyYAML', yaml.__version__), "
           "('matplotlib', matplotlib.__version__)]]\"")

FONT_PX, CELL_W, LINE_H, PAD, MAX_COLS = 15, 9, 21, 18, 118
BG, FG, PROMPT, TITLE_BG, TITLE_FG, MARK = '#1e2226', '#e3e6e8', '#8fd694', '#2d3339', '#aab3b9', '#ff4d4f'


def cell_width(ch: str) -> int:
    if unicodedata.category(ch) == 'Cf':  # BOM 등 폭 0 문자
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1


def wrap(line: str, max_cols: int = MAX_COLS):
    out, cur, cols = [], '', 0
    for ch in line:
        w = cell_width(ch)
        if cols + w > max_cols:
            out.append(cur)
            cur, cols = '', 0
        cur += ch
        cols += w
    out.append(cur)
    return out


_CMAPS = {}


def has_glyph(font_path: Path, ch: str) -> bool:
    if font_path not in _CMAPS:
        from fontTools.ttLib import TTFont
        _CMAPS[font_path] = set(TTFont(str(font_path), fontNumber=0).getBestCmap())
    return ord(ch) in _CMAPS[font_path]


def run(cmd: str, cwd: Path) -> str:
    proc = subprocess.run(['bash', '-lc', cmd], cwd=cwd, capture_output=True, text=True)
    return (proc.stdout + proc.stderr).rstrip('\n')


def pick_font(ch, base, symbol, kfont):
    if has_glyph(Path(base.path), ch):
        return base
    if has_glyph(SANS_FONT, ch):
        return symbol
    return kfont


def render_terminal(blocks, path: Path, title: str, marks=()):
    """blocks: [(cwd, command, output)]. marks: regex patterns to box in red."""
    host = socket.gethostname()
    lines = []  # (text, kind, prompt_len)
    for cwd, command, output in blocks:
        prompt = f'root@{host}:{cwd}# '
        for i, seg in enumerate(wrap(prompt + command)):
            lines.append((seg, 'cmd', len(prompt) if i == 0 else 0))
        for raw in output.split('\n'):
            for seg in wrap(raw.expandtabs(8)):
                lines.append((seg, 'out', 0))
    cols = max(sum(cell_width(c) for c in text) for text, _, _ in lines)
    width = PAD * 2 + max(cols, 60) * CELL_W
    title_h = 30
    height = title_h + PAD * 2 + LINE_H * len(lines)
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, title_h], fill=TITLE_BG)
    for i, color in enumerate(('#ff5f57', '#febc2e', '#28c840')):
        draw.ellipse([12 + i * 20, 9, 24 + i * 20, 21], fill=color)
    kfont_title = ImageFont.truetype(str(KOREAN_FONT), 13)
    draw.text((80, 8), f'{title}  —  root@{host}', font=kfont_title, fill=TITLE_FG)

    mono = ImageFont.truetype(str(MONO_FONT), FONT_PX)
    bold = ImageFont.truetype(str(MONO_BOLD), FONT_PX)
    kfont = ImageFont.truetype(str(KOREAN_FONT), FONT_PX + 2)
    symbol = ImageFont.truetype(str(SANS_FONT), FONT_PX)
    y = title_h + PAD
    for text, kind, prompt_len in lines:
        x_cells = 0
        for idx, ch in enumerate(text):
            if cell_width(ch) == 0:
                continue
            is_prompt = kind == 'cmd' and idx < prompt_len
            color = PROMPT if is_prompt else FG
            x = PAD + x_cells * CELL_W
            if ch in GLYPH_SUBSTITUTES:
                for j, sub in enumerate(GLYPH_SUBSTITUTES[ch]):
                    draw.text((x + j * CELL_W, y + 1), sub, font=pick_font(sub, mono, symbol, kfont), fill=color)
            elif cell_width(ch) == 2:
                draw.text((x + 1, y), ch, font=kfont, fill=color)
            else:
                base = bold if kind == 'cmd' and not is_prompt else mono
                font = pick_font(ch, base, symbol, kfont)
                if font is symbol and symbol.getlength(ch) > CELL_W:
                    font = ImageFont.truetype(str(SANS_FONT), int(FONT_PX * CELL_W / symbol.getlength(ch)))
                draw.text((x, y + 1), ch, font=font, fill=color)
            x_cells += cell_width(ch)
        for pattern in marks:
            for m in re.finditer(pattern, text):
                x0 = PAD + sum(cell_width(c) for c in text[:m.start()]) * CELL_W - 3
                x1 = PAD + sum(cell_width(c) for c in text[:m.end()]) * CELL_W + 3
                draw.rectangle([x0, y - 1, x1, y + LINE_H - 1], outline=MARK, width=2)
        y += LINE_H
    img.save(path)
    return path


def fig00_diagram(path: Path):
    """보고서 페이지의 흐름도(ENV-1)와 같은 배치를 흰 배경으로 그린다."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcdefaults()
    font_manager.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(KOREAN_FONT)).get_name()
    fig, ax = plt.subplots(figsize=(18.8, 2.5), dpi=150)
    ax.set_xlim(0, 18.8)
    ax.set_ylim(0, 2.5)
    ax.axis('off')

    def node(x0, x1, text):
        ax.add_patch(FancyBboxPatch((x0, 0.62), x1 - x0, 0.96, boxstyle='round,pad=0,rounding_size=0.05',
                                    facecolor='white', edgecolor='#333333', linewidth=1.1))
        ax.text((x0 + x1) / 2, 1.1, text, ha='center', va='center', fontsize=11, linespacing=1.5)

    def arrow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, 1.1), (x1, 1.1), arrowstyle='-|>', mutation_scale=12,
                                     color='#333333', lw=1.1, shrinkA=0, shrinkB=0))

    ax.add_patch(FancyBboxPatch((3.25, 0.2), 11.45, 2.1, boxstyle='round,pad=0,rounding_size=0.03',
                                facecolor='#f7f7f7', edgecolor='#9a9a9a', linewidth=1.0))
    ax.text(8.975, 2.03, 'TL-1 시험용 PC (Ubuntu 22.04, CPU 추론)', ha='center', va='center', fontsize=11)

    node(0.15, 2.65, 'TL-2 시험 데이터\nCSV 3개 (거실·방1·방2)')
    node(3.6, 6.0, 'TL-3 시험 프로그램\n1. 파일 확인 (SHA-256)')
    node(6.6, 8.6, '2. 데이터 분할\n(8:1:1, 표준화)')
    node(9.2, 11.6, '3. 추론 (가중치 15개)\n120분 입력 → 60분 예측')
    node(12.2, 14.35, '4. MSE 계산\n(공간별·3개 공간 평균)')
    node(15.25, 18.65, '결과 출력\n화면 · kpi_result.json · 그래프 PNG')
    for x0, x1 in ((2.65, 3.6), (6.0, 6.6), (8.6, 9.2), (11.6, 12.2), (14.35, 15.25)):
        arrow(x0, x1)
    fig.savefig(path, facecolor='white', bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return path


def fig11_result_table(path: Path):
    """측정결과 표 (kpi_result.json 값을 그대로 사용, IMG-09·10과 같은 파스텔톤). 첫 행이 대표값(공간 평균), 나머지는 세부 측정값.
    판정 칸은 두지 않는다."""
    import json

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    report = json.loads((RESULT_DIR / 'kpi_result.json').read_text(encoding='utf-8'))
    seeds = [str(s) for s in report['setup']['seeds']]
    labels = {'livingroom': '거실', 'room1': '방1', 'room2': '방2'}
    ov = report['overall']
    header = ['구분', '예측 구간 수', '목표 MSE'] + [f'시드 {s}' for s in seeds] + ['평균 (표준편차)']
    rows = [[f"{len(ov['spaces'])}개 공간 평균", f"{sum(report['spaces'][e]['windows'] for e in ov['spaces']):,}", '0.233 이하']
            + [f"{ov['by_seed'][s]:.4f}" for s in seeds] + [f"{ov['mse_norm_mean']:.4f} ({ov['mse_norm_std']:.4f})"]]
    for env, d in report['spaces'].items():
        rows.append([f"{labels.get(env, env)} (세부)", f"{d['windows']:,}", '–']
                    + [f"{d['seeds'][s]['mse_norm']:.4f}" for s in seeds]
                    + [f"{d['mse_norm_mean']:.4f} ({d['mse_norm_std']:.4f})"])
    widths = [2.4, 1.9, 1.7] + [1.45] * len(seeds) + [2.6]
    accent_col = len(header) - 1
    row_h, head_h = 0.62, 0.62
    total_w, total_h = sum(widths), head_h + row_h * len(rows)

    plt.rcdefaults()
    font_manager.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(KOREAN_FONT)).get_name()
    fig, ax = plt.subplots(figsize=(total_w * 0.85, total_h * 0.85 + 0.1), dpi=150)
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis('off')
    ink, head_fill, line, accent = '#3A3A3A', '#EAF2FA', '#CFCFCF', '#3F7FBF'

    ax.add_patch(Rectangle((0, 0), total_w, head_h, facecolor=head_fill, edgecolor='none'))
    x = 0
    for w, text in zip(widths, header):
        ax.text(x + w / 2, head_h / 2, text, ha='center', va='center', fontsize=11, color=ink)
        x += w
    for i, row in enumerate(rows):
        y = head_h + i * row_h
        x = 0
        for j, (w, text) in enumerate(zip(widths, row)):
            color = accent if (i == 0 and j == accent_col) else ink
            ax.text(x + w / 2, y + row_h / 2, text, ha='center', va='center', fontsize=11, color=color)
            x += w
    for k in range(len(rows) + 2):  # 가로선
        y = 0 if k == 0 else head_h + (k - 1) * row_h
        ax.plot([0, total_w], [y, y], color=line, lw=1.0 if k in (0, 1, len(rows) + 1) else 0.8)
    x = 0
    for w in widths[:-1]:  # 세로선
        x += w
        ax.plot([x, x], [0, total_h], color=line, lw=0.8)
    ax.plot([0, 0], [0, total_h], color=line, lw=1.0)
    ax.plot([total_w, total_w], [0, total_h], color=line, lw=1.0)
    fig.savefig(path, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*', help='생성할 번호만 (예: 00 01 02)')
    args = parser.parse_args()
    want = set(args.only) if args.only else None
    ok = lambda n: want is None or n in want
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkg = str(PKG)
    made = []

    if ok('00'):
        made.append(fig00_diagram(OUT_DIR / 'IMG-00_env1_diagram.png'))
    if ok('01'):
        cmds = ['lscpu | grep "Model name"', 'free -h | head -2']
        made.append(render_terminal([(pkg, c, run(c, PKG)) for c in cmds],
                                    OUT_DIR / 'IMG-01_hardware.png', 'IMG-01 하드웨어 확인 (CPU·RAM)',
                                    marks=[r'Intel\(R\) Core\(TM\) i7-7700 CPU @ 3\.60GHz', r'(?<=Mem:)\s+\S+']))
    if ok('02'):
        cmds = ['head -n 4 /etc/os-release', 'python -V', PKG_CMD]
        made.append(render_terminal([(pkg, c, run(c, PKG)) for c in cmds],
                                    OUT_DIR / 'IMG-02_os_python_pkgs.png', 'IMG-02 OS·패키지 버전 확인',
                                    marks=[r'Ubuntu 22\.04\.3 LTS', r'Python 3\.12\.13', r'2\.5\.1\+cu121', r'1\.26\.4',
                                           r'2\.2\.3', r'1\.8\.0', r'6\.0\.3', r'3\.9\.3']))
    if ok('03'):
        cmds = ['ls -l data/*.csv DLinear/weights/seed*/*.pt | awk \'{print $5, $9}\'', 'wc -l data/*.csv']
        made.append(render_terminal([(pkg, c, run(c, PKG)) for c in cmds],
                                    OUT_DIR / 'IMG-03_data_files.png', 'IMG-03 시험 데이터·가중치 파일 확인',
                                    marks=[r'^\s*\d+ data/(livingroom|room1|room2)\.csv']))
    if ok('04'):
        c = 'head -n 3 data/livingroom.csv'
        made.append(render_terminal([(pkg, c, run(c, PKG))], OUT_DIR / 'IMG-04_data_columns.png', 'IMG-04 시험 데이터 열 구성',
                                    marks=[r'데이터 시간', r'환기유효전력\(W\)']))
    if ok('05') or ok('06') or ok('07'):
        output = (RESULT_DIR / 'console.txt').read_text(encoding='utf-8').rstrip('\n')
        head, rest = output.split('\n[2/3 데이터 분할]', 1)
        middle, result = rest.split('\n[측정 결과]', 1)
        if ok('05'):
            made.append(render_terminal([(pkg, TEST_CMD, head)], OUT_DIR / 'IMG-05_run_check.png',
                                        'IMG-05 시험 프로그램 실행·파일 확인',
                                        marks=[r'\d+개 파일 모두 일치 \(시험대상 \d+개, 시험도구 \d+개\)', r'사전 학습 가중치로 시험 구간 추론 \(학습하지 않음\)',
                                               r'시험대상\(개발 SW\): 프로그램·설정·가중치 \d+개', r'시험도구\(TL-2\): 시험 데이터 \d+개']))
        if ok('06'):
            made.append(render_terminal([(pkg, '(이어서) 데이터 분할·추론', '[2/3 데이터 분할]' + middle)],
                                        OUT_DIR / 'IMG-06_split_inference.png', 'IMG-06 데이터 분할·추론',
                                        marks=[r'\d{4}-\d\d-\d\d \d\d:\d\d ~ \d{4}-\d\d-\d\d \d\d:\d\d', r'\d,\d{3}$',
                                               r'환기누적전력량\(Wh\), 냉난방누적전력량\(Wh\)']))
        if ok('07'):
            made.append(render_terminal([(pkg, '(이어서) 측정 결과', '[측정 결과]' + result)],
                                        OUT_DIR / 'IMG-07_evaluation_result.png', 'IMG-07 측정 결과',
                                        marks=[r'0\.\d{4}(?=\s+0\.\d{4}\s+0\.\d{4}\s+0\.\d{4}\s*$)', r'(?<=MSE\(표준화\): )0\.\d{4}']))
    if ok('08'):
        c = f'cat {RESULT_SHOWN}/kpi_result.json'
        made.append(render_terminal([(pkg, c, (RESULT_DIR / 'kpi_result.json').read_text(encoding='utf-8').rstrip('\n'))],
                                    OUT_DIR / 'IMG-08_result_json.png', 'IMG-08 결과 파일 내용',
                                    marks=[r'"mse_norm_mean": [\d.]+', r'"seeds": \[[\d, ]+\]', r'"all_match": true',
                                           r'"test_target_files"', r'"test_tool_data_files"']))
    if ok('11'):
        made.append(fig11_result_table(OUT_DIR / 'IMG-11_result_table.png'))
    if ok('09'):
        made.append(Path(shutil.copy(RESULT_DIR / 'pred_vs_actual.png', OUT_DIR / 'IMG-09_pred_vs_actual.png')))
    if ok('10'):
        made.append(Path(shutil.copy(PKG / 'DLinear' / 'weights' / 'learning_curve_seed42.png', OUT_DIR / 'IMG-10_learning_curve.png')))
    for p in made:
        print(p)


if __name__ == '__main__':
    main()
