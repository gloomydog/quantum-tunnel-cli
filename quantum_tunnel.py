#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_tunnel.py
==================
1次元シュレーディンガー方程式を split-step Fourier 法(分裂演算子法)で
数値的に解き、左から右へ進む波束がポテンシャル障壁/井戸/調和振動子ポテンシャル
にぶつかったときの振る舞い(トンネル効果・反射)をターミナル上にリアルタイム
アニメーション表示するツール。

物理単位: hbar = 1, m = 1 の自然単位系を使用。

境界には複素吸収ポテンシャル (CAP: Complex Absorbing Potential) を導入し、
波束が計算領域の端(=画面外)に到達すると振幅が減衰して消えるようにしている。
これによりFFTの周期境界条件による折り返し(エイリアシング)ノイズも防げる。

操作:
    q : 終了
    p : 一時停止 / 再開
    r : 波束を初期状態にリセット

使用例:
    python3 quantum_tunnel.py
    python3 quantum_tunnel.py --potential well --height 8 --width 4
    python3 quantum_tunnel.py --potential harmonic --height 20
    python3 quantum_tunnel.py --potential rectangular --num-barriers 3 --spacing 6 --height 12
    python3 quantum_tunnel.py --display all --k0 6 --height 10
"""

import argparse
import curses
import time
import sys

import numpy as np


def trapz(y, x):
    """numpyのバージョン差 (trapz/trapezoid) に依存しない台形積分。"""
    y = np.asarray(y)
    x = np.asarray(x)
    if y.size < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


# ----------------------------------------------------------------------
# 引数処理
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="1次元量子力学: 波束のトンネル効果ターミナル可視化 (hbar=m=1 の自然単位系)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--potential", choices=["rectangular", "well", "harmonic", "coulomb"],
                    default="rectangular", help="ポテンシャルの種類")
    p.add_argument("--height", type=float, default=15.0,
                    help="障壁の高さ / 井戸の深さ / harmonicの曲率目安値 (エネルギー単位)")
    p.add_argument("--width", type=float, default=2.0,
                    help="障壁(井戸)1つあたりの幅")
    p.add_argument("--num-barriers", type=int, default=1,
                    help="障壁/井戸の数 (harmonicでは無視される)")
    p.add_argument("--spacing", type=float, default=None,
                    help="複数障壁の中心間隔 (デフォルト: width*3)")
    p.add_argument("--k0", type=float, default=5.0,
                    help="波束の初期運動量 (正=右向き)。 E0 = k0^2/2 が目安の運動エネルギー")
    p.add_argument("--sigma", type=float, default=3.0,
                    help="波束の初期空間幅 (標準偏差)")
    p.add_argument("--x0", type=float, default=None,
                    help="波束の初期中心位置 (デフォルト: -xmax*0.6)")
    p.add_argument("--xmax", type=float, default=60.0,
                    help="空間領域の半幅。領域は [-xmax, xmax]")
    p.add_argument("--grid-points", type=int, default=3000,
                    help="空間グリッド点数")
    p.add_argument("--dt", type=float, default=0.005,
                    help="シミュレーションの時間刻み")
    p.add_argument("--steps-per-frame", type=int, default=8,
                    help="1描画フレームあたりの時間発展ステップ数")
    p.add_argument("--fps", type=float, default=24.0,
                    help="描画フレームレート")
    p.add_argument("--display", choices=["prob", "real", "imag", "all"],
                    default="prob",
                    help="表示モード: prob=|psi|^2, real=実部, imag=虚部, all=すべて重ね描き")
    p.add_argument("--absorb-strength", type=float, default=6.0,
                    help="境界吸収ポテンシャルの強さ")
    p.add_argument("--absorb-width", type=float, default=0.15,
                    help="境界吸収領域の割合 (領域全長に対する比率, 片側)")
    p.add_argument("--no-loop", action="store_true",
                    help="波束が消えても自動リスタートしない(1回のみ実行)")
    return p.parse_args()


# ----------------------------------------------------------------------
# ポテンシャル構築
# ----------------------------------------------------------------------
def build_potential(x, args):
    """物理的なポテンシャル V(x) (吸収ポテンシャルは含まない、表示用にも使う)"""
    if args.potential == "harmonic":
        half_w = max(args.width, 1e-6)
        omega2 = 2.0 * args.height / (half_w ** 2)
        V = 0.5 * omega2 * x ** 2
        return V

    if args.potential == "coulomb":
        # 反発クーロン障壁: V(x) = height / sqrt(x^2 + eps^2)
        # (原子核のアルファ崩壊/核融合で有名な「クーロン障壁のトンネル効果」を模したもの)
        # width をソフトニング長として使い、x=0 での発散を防ぐ。
        eps = max(args.width, 1e-6)
        V = args.height / np.sqrt(x ** 2 + eps ** 2)
        return V

    V = np.zeros_like(x)
    height = args.height if args.potential == "rectangular" else -abs(args.height)
    n = max(1, args.num_barriers)
    spacing = args.spacing if args.spacing is not None else args.width * 3.0
    centers = [(i - (n - 1) / 2.0) * spacing for i in range(n)]
    half = args.width / 2.0
    for c in centers:
        mask = (x >= c - half) & (x < c + half)
        V[mask] += height
    return V


def build_absorber(x, args):
    """複素吸収ポテンシャル CAP(x) >= 0。 領域端で滑らかに立ち上がる。"""
    L = 2.0 * args.xmax
    edge_w = max(args.absorb_width * L, 1e-6)
    x_edge = args.xmax - edge_w
    d = np.abs(x) - x_edge
    d = np.clip(d, 0.0, edge_w)
    frac = d / edge_w
    CAP = args.absorb_strength * frac ** 2
    return CAP


# ----------------------------------------------------------------------
# 初期波束
# ----------------------------------------------------------------------
def initial_psi(x, args):
    x0 = args.x0 if args.x0 is not None else -args.xmax * 0.6
    sigma = args.sigma
    norm = (2.0 * np.pi * sigma ** 2) ** (-0.25)
    psi = norm * np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * args.k0 * x)
    return psi


# ----------------------------------------------------------------------
# 時間発展 (split-step Fourier法)
# ----------------------------------------------------------------------
class Evolver:
    def __init__(self, x, V, CAP, dt):
        N = x.size
        dx = x[1] - x[0]
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        V_eff = V - 1j * CAP
        self.expV_half = np.exp(-1j * V_eff * dt / 2.0)
        self.expK = np.exp(-1j * (k ** 2 / 2.0) * dt)

    def step(self, psi):
        psi = self.expV_half * psi
        psi = np.fft.ifft(self.expK * np.fft.fft(psi))
        psi = self.expV_half * psi
        return psi


# ----------------------------------------------------------------------
# ターミナル描画
# ----------------------------------------------------------------------
class Renderer:
    COLOR_PROB = 1
    COLOR_REAL = 2
    COLOR_IMAG = 3
    COLOR_BARRIER_POS = 4
    COLOR_BARRIER_NEG = 5
    COLOR_TEXT = 6
    COLOR_DIM = 7

    def __init__(self, stdscr, args, x, V):
        self.stdscr = stdscr
        self.args = args
        self.x = x
        self.V = V
        curses.curs_set(0)
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        curses.init_pair(self.COLOR_PROB, curses.COLOR_YELLOW, bg)
        curses.init_pair(self.COLOR_REAL, curses.COLOR_CYAN, bg)
        curses.init_pair(self.COLOR_IMAG, curses.COLOR_MAGENTA, bg)
        curses.init_pair(self.COLOR_BARRIER_POS, curses.COLOR_RED, bg)
        curses.init_pair(self.COLOR_BARRIER_NEG, curses.COLOR_BLUE, bg)
        curses.init_pair(self.COLOR_TEXT, curses.COLOR_WHITE, bg)
        curses.init_pair(self.COLOR_DIM, curses.COLOR_GREEN, bg)

        # 表示スケール基準は初期波束のピークで固定する
        prob0 = np.abs(initial_psi(x, args)) ** 2
        self.prob_scale = max(prob0.max(), 1e-9) * 1.05
        self.amp_scale = max(np.abs(initial_psi(x, args)).max(), 1e-9) * 1.15
        v_abs = np.abs(V)
        self.v_scale = max(v_abs.max(), 1e-9)

    def safe_addch(self, row, col, ch, attr):
        try:
            self.stdscr.addch(row, col, ch, attr)
        except curses.error:
            pass

    def safe_addstr(self, row, col, s, attr=0):
        try:
            self.stdscr.addstr(row, col, s, attr)
        except curses.error:
            pass

    def render(self, psi, t, paused, restarting):
        stdscr = self.stdscr
        args = self.args
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()

        header_rows = 2
        footer_rows = 3
        plot_rows = rows - header_rows - footer_rows
        plot_cols = cols - 2

        if plot_rows < 6 or plot_cols < 20:
            self.safe_addstr(0, 0, "端末が小さすぎます。ウィンドウを広げてください。",
                              curses.color_pair(self.COLOR_TEXT))
            stdscr.refresh()
            return

        # --- ヘッダー ---
        title = " Quantum Wave Packet Tunneling  |  potential={}  ".format(args.potential)
        self.safe_addstr(0, 0, title[:cols - 1],
                          curses.color_pair(self.COLOR_TEXT) | curses.A_BOLD)
        status = "PAUSED" if paused else ("RESTARTING" if restarting else "RUNNING")
        info = " t={:6.2f}  V0={:.2f}  width={:.2f}  n={}  disp={}  [{}]".format(
            t, args.height, args.width, args.num_barriers, args.display, status)
        self.safe_addstr(1, 0, info[:cols - 1], curses.color_pair(self.COLOR_TEXT))

        # --- ビン分割 (物理グリッド -> 画面列) ---
        x = self.x
        edges = np.linspace(-args.xmax, args.xmax, plot_cols + 1)
        idx_edges = np.searchsorted(x, edges)

        prob = np.abs(psi) ** 2
        re = psi.real
        im = psi.imag
        V = self.V

        bar_bottom_row = header_rows + plot_rows - 1
        baseline_row = header_rows + plot_rows // 2
        max_barrier_rows = max(2, plot_rows // 3)

        for c in range(plot_cols):
            i0, i1 = idx_edges[c], idx_edges[c + 1]
            if i1 <= i0:
                i1 = min(i0 + 1, x.size)
            if i1 <= i0:
                continue
            col_screen = c + 1
            if col_screen >= cols - 1:
                break

            seg_prob = prob[i0:i1]
            seg_re = re[i0:i1]
            seg_im = im[i0:i1]
            seg_V = V[i0:i1]

            pval = float(seg_prob.max()) if seg_prob.size else 0.0
            mid = seg_re.size // 2
            rval = float(seg_re[mid]) if seg_re.size else 0.0
            ival = float(seg_im[mid]) if seg_im.size else 0.0
            vval = float(seg_V[np.argmax(np.abs(seg_V))]) if seg_V.size else 0.0

            # --- ポテンシャル(障壁/井戸)を背景として描画 ---
            if vval != 0.0:
                frac = min(abs(vval) / self.v_scale, 1.0)
                n_rows = int(round(frac * max_barrier_rows))
                color = (self.COLOR_BARRIER_POS if vval > 0 else self.COLOR_BARRIER_NEG)
                ch = "▓" if vval > 0 else "▒"
                for r in range(n_rows):
                    row = bar_bottom_row - r
                    if row < header_rows:
                        break
                    self.safe_addch(row, col_screen, ch, curses.color_pair(color))

            mode = args.display
            # --- 確率密度 (棒グラフ、下から積み上げ) ---
            if mode in ("prob", "all"):
                frac = min(max(pval / self.prob_scale, 0.0), 1.0)
                n_rows = int(round(frac * plot_rows))
                attr = curses.color_pair(self.COLOR_PROB)
                if mode == "all":
                    attr |= curses.A_DIM
                for r in range(n_rows):
                    row = bar_bottom_row - r
                    if row < header_rows:
                        break
                    self.safe_addch(row, col_screen, "█", attr)

            # --- 実部 (符号付き、中心線基準の折れ線) ---
            if mode in ("real", "all"):
                frac = min(max(rval / self.amp_scale, -1.0), 1.0)
                row = baseline_row - int(round(frac * (plot_rows // 2)))
                row = max(header_rows, min(bar_bottom_row, row))
                self.safe_addch(row, col_screen, "●", curses.color_pair(self.COLOR_REAL) | curses.A_BOLD)

            # --- 虚部 ---
            if mode in ("imag", "all"):
                frac = min(max(ival / self.amp_scale, -1.0), 1.0)
                row = baseline_row - int(round(frac * (plot_rows // 2)))
                row = max(header_rows, min(bar_bottom_row, row))
                self.safe_addch(row, col_screen, "●", curses.color_pair(self.COLOR_IMAG) | curses.A_BOLD)

        # --- 中心基準線 (real/imag/all のとき) ---
        if args.display in ("real", "imag", "all"):
            for c in range(1, min(plot_cols + 1, cols - 1)):
                self.safe_addch(baseline_row, c, "-", curses.color_pair(self.COLOR_DIM) | curses.A_DIM)

        # --- フッター: 凡例 + 統計値 ---
        norm_total = float(trapz(prob, x))
        left_mask = x < 0
        right_mask = ~left_mask
        norm_left = float(trapz(prob[left_mask], x[left_mask]))
        norm_right = float(trapz(prob[right_mask], x[right_mask]))

        legend = (" prob=\u2588(黄)  real=\u25cf(水)  imag=\u25cf(紫)  "
                  "barrier=\u2593(赤)  well=\u2592(青) | [q]終了 [p]一時停止 [r]リセット")
        self.safe_addstr(rows - 3, 0, legend[:cols - 1], curses.color_pair(self.COLOR_TEXT))

        stats = " norm={:.3f}  left={:.3f}  right(透過側)={:.3f}".format(
            norm_total, norm_left, norm_right)
        self.safe_addstr(rows - 2, 0, stats[:cols - 1], curses.color_pair(self.COLOR_TEXT))

        stdscr.refresh()


# ----------------------------------------------------------------------
# メインループ
# ----------------------------------------------------------------------
def run(stdscr, args):
    x = np.linspace(-args.xmax, args.xmax, args.grid_points, endpoint=False)
    V = build_potential(x, args)
    CAP = build_absorber(x, args)
    evolver = Evolver(x, V, CAP, args.dt)

    renderer = Renderer(stdscr, args, x, V)

    psi = initial_psi(x, args)
    t = 0.0
    paused = False

    # 安全のための最大シミュレーション時間 (往復2回分程度で強制リセット)
    t_max = 2.0 * (2.0 * args.xmax) / max(abs(args.k0), 0.5)

    frame_ms = max(1, int(1000.0 / max(args.fps, 1e-6)))
    stdscr.timeout(frame_ms)

    restarting = False
    restart_timer = 0.0

    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("p"), ord("P")):
            paused = not paused
        elif key in (ord("r"), ord("R")):
            psi = initial_psi(x, args)
            t = 0.0
            restarting = False

        if not paused:
            if not restarting:
                for _ in range(args.steps_per_frame):
                    psi = evolver.step(psi)
                t += args.dt * args.steps_per_frame

                norm_total = float(trapz(np.abs(psi) ** 2, x))
                if (norm_total < 0.03 or t > t_max) and not args.no_loop:
                    restarting = True
                    restart_timer = time.time()
                elif norm_total < 0.03 and args.no_loop:
                    renderer.render(psi, t, paused, restarting=False)
                    time.sleep(1.5)
                    break
            else:
                if time.time() - restart_timer > 1.2:
                    psi = initial_psi(x, args)
                    t = 0.0
                    restarting = False

        renderer.render(psi, t, paused, restarting)


def main():
    args = parse_args()
    if args.grid_points < 64:
        print("grid-points is too small.", file=sys.stderr)
        sys.exit(1)
    try:
        curses.wrapper(run, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
