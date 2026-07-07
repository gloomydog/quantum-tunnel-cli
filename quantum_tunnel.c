/*
 * quantum_tunnel.c
 * =================
 * Solves the 1D time-dependent Schrodinger equation with the split-step
 * Fourier method (operator splitting) and renders, in real time inside a
 * terminal (ncurses), how a wave packet moving left-to-right interacts with
 * a potential barrier / well / harmonic-oscillator potential (tunneling,
 * reflection, partial transmission).
 *
 * Units: natural units with hbar = 1, m = 1.
 * A complex absorbing potential (CAP) is added near the domain edges, so the
 * wave packet's amplitude decays and "disappears" once it reaches the edge
 * of the screen (i.e. leaves the simulated region).
 *
 * Dependencies: libc (complex.h, math.h) + ncurses only. FFT is a small
 * self-contained implementation (no external library needed).
 *
 * Build:
 *     gcc -O2 -std=c99 -o quantum_tunnel quantum_tunnel.c -lncurses -lm
 *
 * Controls: q=quit  p=pause/resume  r=reset
 *
 * Examples:
 *     ./quantum_tunnel
 *     ./quantum_tunnel --potential well --height 8 --width 4
 *     ./quantum_tunnel --potential harmonic --height 20
 *     ./quantum_tunnel --potential rectangular --num-barriers 3 --spacing 6 --height 12
 *     ./quantum_tunnel --display all --k0 6 --height 10
 */

#define _POSIX_C_SOURCE 199309L

#include <complex.h>
#include <ctype.h>
#include <getopt.h>
#include <math.h>
#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum { POT_RECTANGULAR, POT_WELL, POT_HARMONIC, POT_COULOMB } PotentialType;
typedef enum { DISP_PROB, DISP_REAL, DISP_IMAG, DISP_ALL } DisplayMode;

/* 色ペア番号 */
enum {
    CP_PROB = 1,
    CP_REAL,
    CP_IMAG,
    CP_BARRIER_POS,
    CP_BARRIER_NEG,
    CP_TEXT,
    CP_DIM
};

typedef struct {
    PotentialType potential;
    double height;
    double width;
    int num_barriers;
    double spacing;      /* < 0 なら自動 (width*3) */
    double k0;
    double sigma;
    double x0;           /* NAN なら自動 (-xmax*0.6) */
    double xmax;
    int grid_points;     /* 内部で2のべき乗に切り上げ */
    double dt;
    int steps_per_frame;
    double fps;
    DisplayMode display;
    double absorb_strength;
    double absorb_width;
    int no_loop;
} Args;

/* ------------------------------------------------------------------ */
/* 引数のデフォルト値                                                  */
/* ------------------------------------------------------------------ */
static void args_default(Args *a) {
    a->potential = POT_RECTANGULAR;
    a->height = 15.0;
    a->width = 2.0;
    a->num_barriers = 1;
    a->spacing = -1.0;
    a->k0 = 5.0;
    a->sigma = 3.0;
    a->x0 = NAN;
    a->xmax = 60.0;
    a->grid_points = 2048;
    a->dt = 0.005;
    a->steps_per_frame = 8;
    a->fps = 24.0;
    a->display = DISP_PROB;
    a->absorb_strength = 6.0;
    a->absorb_width = 0.15;
    a->no_loop = 0;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n\n"
           "  --potential {rectangular,well,harmonic,coulomb}  potential type (default: rectangular)\n"
           "  --height V        barrier height / well depth / harmonic curvature scale (default: 15.0)\n"
           "  --width W         width of each barrier/well (default: 2.0)\n"
           "  --num-barriers N  number of barriers/wells (ignored for harmonic) (default: 1)\n"
           "  --spacing S       center-to-center spacing for multiple barriers (default: width*3)\n"
           "  --k0 K            initial momentum of the wave packet (default: 5.0)\n"
           "  --sigma S         initial width of the wave packet (default: 3.0)\n"
           "  --x0 X            initial position of the wave packet (default: -xmax*0.6)\n"
           "  --xmax X          half-width of the spatial domain [-xmax,xmax] (default: 60.0)\n"
           "  --grid-points N   number of grid points, rounded up to a power of 2 (default: 2048)\n"
           "  --dt DT           simulation time step (default: 0.005)\n"
           "  --steps-per-frame N  time-evolution steps per rendered frame (default: 8)\n"
           "  --fps F           rendering frame rate (default: 24)\n"
           "  --display {prob,real,imag,all}  display mode (default: prob)\n"
           "  --absorb-strength S  absorbing-boundary strength (default: 6.0)\n"
           "  --absorb-width W     absorbing-boundary width, fraction of domain (default: 0.15)\n"
           "  --no-loop         do not auto-restart after the wave packet is absorbed\n"
           "  -h, --help        show this help\n\n"
           "Controls: q=quit  p=pause/resume  r=reset\n",
           prog);
}

static int parse_args(int argc, char **argv, Args *a) {
    args_default(a);

    enum {
        OPT_POTENTIAL = 1000, OPT_HEIGHT, OPT_WIDTH, OPT_NUM_BARRIERS,
        OPT_SPACING, OPT_K0, OPT_SIGMA, OPT_X0, OPT_XMAX, OPT_GRID_POINTS,
        OPT_DT, OPT_STEPS_PER_FRAME, OPT_FPS, OPT_DISPLAY,
        OPT_ABSORB_STRENGTH, OPT_ABSORB_WIDTH, OPT_NO_LOOP
    };

    static struct option longopts[] = {
        {"potential", required_argument, 0, OPT_POTENTIAL},
        {"height", required_argument, 0, OPT_HEIGHT},
        {"width", required_argument, 0, OPT_WIDTH},
        {"num-barriers", required_argument, 0, OPT_NUM_BARRIERS},
        {"spacing", required_argument, 0, OPT_SPACING},
        {"k0", required_argument, 0, OPT_K0},
        {"sigma", required_argument, 0, OPT_SIGMA},
        {"x0", required_argument, 0, OPT_X0},
        {"xmax", required_argument, 0, OPT_XMAX},
        {"grid-points", required_argument, 0, OPT_GRID_POINTS},
        {"dt", required_argument, 0, OPT_DT},
        {"steps-per-frame", required_argument, 0, OPT_STEPS_PER_FRAME},
        {"fps", required_argument, 0, OPT_FPS},
        {"display", required_argument, 0, OPT_DISPLAY},
        {"absorb-strength", required_argument, 0, OPT_ABSORB_STRENGTH},
        {"absorb-width", required_argument, 0, OPT_ABSORB_WIDTH},
        {"no-loop", no_argument, 0, OPT_NO_LOOP},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "h", longopts, NULL)) != -1) {
        switch (c) {
            case OPT_POTENTIAL:
                if (strcmp(optarg, "rectangular") == 0) a->potential = POT_RECTANGULAR;
                else if (strcmp(optarg, "well") == 0) a->potential = POT_WELL;
                else if (strcmp(optarg, "harmonic") == 0) a->potential = POT_HARMONIC;
                else if (strcmp(optarg, "coulomb") == 0) a->potential = POT_COULOMB;
                else { fprintf(stderr, "unknown --potential: %s\n", optarg); return -1; }
                break;
            case OPT_HEIGHT: a->height = atof(optarg); break;
            case OPT_WIDTH: a->width = atof(optarg); break;
            case OPT_NUM_BARRIERS: a->num_barriers = atoi(optarg); break;
            case OPT_SPACING: a->spacing = atof(optarg); break;
            case OPT_K0: a->k0 = atof(optarg); break;
            case OPT_SIGMA: a->sigma = atof(optarg); break;
            case OPT_X0: a->x0 = atof(optarg); break;
            case OPT_XMAX: a->xmax = atof(optarg); break;
            case OPT_GRID_POINTS: a->grid_points = atoi(optarg); break;
            case OPT_DT: a->dt = atof(optarg); break;
            case OPT_STEPS_PER_FRAME: a->steps_per_frame = atoi(optarg); break;
            case OPT_FPS: a->fps = atof(optarg); break;
            case OPT_DISPLAY:
                if (strcmp(optarg, "prob") == 0) a->display = DISP_PROB;
                else if (strcmp(optarg, "real") == 0) a->display = DISP_REAL;
                else if (strcmp(optarg, "imag") == 0) a->display = DISP_IMAG;
                else if (strcmp(optarg, "all") == 0) a->display = DISP_ALL;
                else { fprintf(stderr, "unknown --display: %s\n", optarg); return -1; }
                break;
            case OPT_ABSORB_STRENGTH: a->absorb_strength = atof(optarg); break;
            case OPT_ABSORB_WIDTH: a->absorb_width = atof(optarg); break;
            case OPT_NO_LOOP: a->no_loop = 1; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return -1;
        }
    }
    if (isnan(a->x0)) a->x0 = -a->xmax * 0.6;
    if (a->spacing < 0) a->spacing = a->width * 3.0;
    return 0;
}

/* ------------------------------------------------------------------ */
/* 自前FFT (反復版 Cooley-Tukey, in-place, N=2のべき乗)                 */
/* invert=0: 順変換 (numpy.fft.fft と同じ規約, exp(-i..))               */
/* invert=1: 逆変換 (1/N 正規化込み, numpy.fft.ifft と同じ規約)          */
/* ------------------------------------------------------------------ */
static void fft(double complex *a, int n, int invert) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double complex tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / (double)len * (invert ? 1.0 : -1.0);
        double complex wlen = cexp(I * ang);
        for (int istart = 0; istart < n; istart += len) {
            double complex w = 1.0;
            int half = len / 2;
            for (int k = 0; k < half; k++) {
                double complex u = a[istart + k];
                double complex v = a[istart + k + half] * w;
                a[istart + k] = u + v;
                a[istart + k + half] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (int i = 0; i < n; i++) a[i] /= (double)n;
    }
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ------------------------------------------------------------------ */
/* ポテンシャル / 吸収境界 / 初期波束                                    */
/* ------------------------------------------------------------------ */
static void build_potential(const double *x, int n, const Args *a, double *V) {
    if (a->potential == POT_HARMONIC) {
        double half_w = a->width > 1e-9 ? a->width : 1e-9;
        double omega2 = 2.0 * a->height / (half_w * half_w);
        for (int i = 0; i < n; i++) V[i] = 0.5 * omega2 * x[i] * x[i];
        return;
    }

    if (a->potential == POT_COULOMB) {
        /* 反発クーロン障壁: V(x) = height / sqrt(x^2 + eps^2)
         * アルファ崩壊/核融合で有名な「クーロン障壁のトンネル効果」を模したもの。
         * width をソフトニング長として使い、x=0 での発散を防ぐ。 */
        double eps = a->width > 1e-9 ? a->width : 1e-9;
        for (int i = 0; i < n; i++) V[i] = a->height / sqrt(x[i] * x[i] + eps * eps);
        return;
    }

    for (int i = 0; i < n; i++) V[i] = 0.0;
    double height = (a->potential == POT_RECTANGULAR) ? a->height : -fabs(a->height);
    int nb = a->num_barriers > 1 ? a->num_barriers : 1;
    double half = a->width / 2.0;

    for (int b = 0; b < nb; b++) {
        double c = (b - (nb - 1) / 2.0) * a->spacing;
        double lo = c - half, hi = c + half;
        for (int i = 0; i < n; i++) {
            if (x[i] >= lo && x[i] < hi) V[i] += height;
        }
    }
}

static void build_absorber(const double *x, int n, const Args *a, double *CAP) {
    double L = 2.0 * a->xmax;
    double edge_w = a->absorb_width * L;
    if (edge_w < 1e-9) edge_w = 1e-9;
    double x_edge = a->xmax - edge_w;
    for (int i = 0; i < n; i++) {
        double d = fabs(x[i]) - x_edge;
        if (d < 0.0) d = 0.0;
        if (d > edge_w) d = edge_w;
        double frac = d / edge_w;
        CAP[i] = a->absorb_strength * frac * frac;
    }
}

static void initial_psi(const double *x, int n, const Args *a, double complex *psi) {
    double sigma = a->sigma;
    double norm = pow(2.0 * M_PI * sigma * sigma, -0.25);
    for (int i = 0; i < n; i++) {
        double dx = x[i] - a->x0;
        double envelope = norm * exp(-(dx * dx) / (4.0 * sigma * sigma));
        psi[i] = envelope * cexp(I * a->k0 * x[i]);
    }
}

static double trapz(const double *y, const double *x, int n) {
    if (n < 2) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n - 1; i++) {
        s += (y[i] + y[i + 1]) * 0.5 * (x[i + 1] - x[i]);
    }
    return s;
}

/* ------------------------------------------------------------------ */
/* 時間発展 (split-step Fourier法)                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    double complex *expV_half;
    double complex *expK;
    int n;
} Evolver;

static void evolver_init(Evolver *ev, const double *x, const double *V,
                          const double *CAP, double dt, int n) {
    ev->n = n;
    ev->expV_half = malloc(sizeof(double complex) * n);
    ev->expK = malloc(sizeof(double complex) * n);

    double dx = x[1] - x[0];
    for (int i = 0; i < n; i++) {
        double complex Veff = V[i] - I * CAP[i];
        ev->expV_half[i] = cexp(-I * Veff * dt / 2.0);

        double kk;
        if (i < n / 2) kk = 2.0 * M_PI * (double)i / (n * dx);
        else kk = 2.0 * M_PI * (double)(i - n) / (n * dx);
        ev->expK[i] = cexp(-I * (kk * kk / 2.0) * dt);
    }
}

static void evolver_free(Evolver *ev) {
    free(ev->expV_half);
    free(ev->expK);
}

static void evolve_step(Evolver *ev, double complex *psi) {
    int n = ev->n;
    for (int i = 0; i < n; i++) psi[i] *= ev->expV_half[i];
    fft(psi, n, 0);
    for (int i = 0; i < n; i++) psi[i] *= ev->expK[i];
    fft(psi, n, 1);
    for (int i = 0; i < n; i++) psi[i] *= ev->expV_half[i];
}

/* ------------------------------------------------------------------ */
/* 描画                                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    double prob_scale;
    double amp_scale;
    double v_scale;
} Scales;

static void scales_init(Scales *sc, const double *x, int n, const Args *a, const double *V) {
    double complex *psi0 = malloc(sizeof(double complex) * n);
    initial_psi(x, n, a, psi0);
    double pmax = 0.0, amax = 0.0;
    for (int i = 0; i < n; i++) {
        double amp = cabs(psi0[i]);
        double p = amp * amp;
        if (p > pmax) pmax = p;
        if (amp > amax) amax = amp;
    }
    free(psi0);
    sc->prob_scale = (pmax > 1e-9 ? pmax : 1e-9) * 1.05;
    sc->amp_scale = (amax > 1e-9 ? amax : 1e-9) * 1.15;

    double vmax = 0.0;
    for (int i = 0; i < n; i++) {
        double v = fabs(V[i]);
        if (v > vmax) vmax = v;
    }
    sc->v_scale = (vmax > 1e-9 ? vmax : 1e-9);
}

static const char *potential_name(PotentialType p) {
    switch (p) {
        case POT_RECTANGULAR: return "rectangular";
        case POT_WELL: return "well";
        case POT_HARMONIC: return "harmonic";
        case POT_COULOMB: return "coulomb";
    }
    return "?";
}

static const char *display_name(DisplayMode d) {
    switch (d) {
        case DISP_PROB: return "prob";
        case DISP_REAL: return "real";
        case DISP_IMAG: return "imag";
        case DISP_ALL: return "all";
    }
    return "?";
}

static void render(const Args *a, const double *x, int n, const double *V,
                    const double complex *psi, double t, int paused,
                    int restarting, const Scales *sc) {
    int rows, cols;
    getmaxyx(stdscr, rows, cols);
    erase();

    int header_rows = 2;
    int footer_rows = 3;
    int plot_rows = rows - header_rows - footer_rows;
    int plot_cols = cols - 2;

    if (plot_rows < 6 || plot_cols < 20) {
        attron(COLOR_PAIR(CP_TEXT));
        mvprintw(0, 0, "Terminal too small. Please widen the window.");
        attroff(COLOR_PAIR(CP_TEXT));
        refresh();
        return;
    }

    attron(COLOR_PAIR(CP_TEXT) | A_BOLD);
    mvprintw(0, 0, " Quantum Wave Packet Tunneling  |  potential=%s ",
              potential_name(a->potential));
    attroff(COLOR_PAIR(CP_TEXT) | A_BOLD);

    const char *status = paused ? "PAUSED" : (restarting ? "RESTARTING" : "RUNNING");
    attron(COLOR_PAIR(CP_TEXT));
    mvprintw(1, 0, " t=%6.2f  V0=%.2f  width=%.2f  n=%d  disp=%s  [%s]",
              t, a->height, a->width, a->num_barriers, display_name(a->display), status);
    attroff(COLOR_PAIR(CP_TEXT));

    double dx_phys = (2.0 * a->xmax) / (double)n;
    int bar_bottom_row = header_rows + plot_rows - 1;
    int baseline_row = header_rows + plot_rows / 2;
    int max_barrier_rows = plot_rows / 3 > 2 ? plot_rows / 3 : 2;

    for (int c = 0; c < plot_cols; c++) {
        double xl = -a->xmax + (double)c * (2.0 * a->xmax) / plot_cols;
        double xr = -a->xmax + (double)(c + 1) * (2.0 * a->xmax) / plot_cols;
        int i0 = (int)round((xl + a->xmax) / dx_phys);
        int i1 = (int)round((xr + a->xmax) / dx_phys);
        if (i0 < 0) i0 = 0;
        if (i1 > n) i1 = n;
        if (i1 <= i0) i1 = i0 + 1;
        if (i1 > n) i1 = n;
        if (i0 >= n) continue;

        int col_screen = c + 1;
        if (col_screen >= cols - 1) break;

        double pval = 0.0, vval = 0.0, vabs_max = -1.0;
        double rval = 0.0, ival = 0.0;
        int mid = i0 + (i1 - i0) / 2;
        if (mid >= n) mid = n - 1;
        rval = creal(psi[mid]);
        ival = cimag(psi[mid]);

        for (int i = i0; i < i1; i++) {
            double amp2 = creal(psi[i]) * creal(psi[i]) + cimag(psi[i]) * cimag(psi[i]);
            if (amp2 > pval) pval = amp2;
            double va = fabs(V[i]);
            if (va > vabs_max) { vabs_max = va; vval = V[i]; }
        }

        /* --- ポテンシャル(障壁/井戸)の背景描画 --- */
        if (vval != 0.0) {
            double frac = fabs(vval) / sc->v_scale;
            if (frac > 1.0) frac = 1.0;
            int nrows = (int)round(frac * max_barrier_rows);
            int pair = vval > 0 ? CP_BARRIER_POS : CP_BARRIER_NEG;
            char ch = vval > 0 ? '%' : ':';
            attron(COLOR_PAIR(pair));
            for (int r = 0; r < nrows; r++) {
                int row = bar_bottom_row - r;
                if (row < header_rows) break;
                mvaddch(row, col_screen, ch);
            }
            attroff(COLOR_PAIR(pair));
        }

        DisplayMode mode = a->display;

        if (mode == DISP_PROB || mode == DISP_ALL) {
            double frac = pval / sc->prob_scale;
            if (frac < 0) frac = 0;
            if (frac > 1) frac = 1;
            int nrows = (int)round(frac * plot_rows);
            attron(COLOR_PAIR(CP_PROB) | (mode == DISP_ALL ? A_DIM : A_NORMAL));
            for (int r = 0; r < nrows; r++) {
                int row = bar_bottom_row - r;
                if (row < header_rows) break;
                mvaddch(row, col_screen, '#');
            }
            attroff(COLOR_PAIR(CP_PROB) | (mode == DISP_ALL ? A_DIM : A_NORMAL));
        }

        if (mode == DISP_REAL || mode == DISP_ALL) {
            double frac = rval / sc->amp_scale;
            if (frac < -1) frac = -1;
            if (frac > 1) frac = 1;
            int row = baseline_row - (int)round(frac * (plot_rows / 2));
            if (row < header_rows) row = header_rows;
            if (row > bar_bottom_row) row = bar_bottom_row;
            attron(COLOR_PAIR(CP_REAL) | A_BOLD);
            mvaddch(row, col_screen, 'o');
            attroff(COLOR_PAIR(CP_REAL) | A_BOLD);
        }

        if (mode == DISP_IMAG || mode == DISP_ALL) {
            double frac = ival / sc->amp_scale;
            if (frac < -1) frac = -1;
            if (frac > 1) frac = 1;
            int row = baseline_row - (int)round(frac * (plot_rows / 2));
            if (row < header_rows) row = header_rows;
            if (row > bar_bottom_row) row = bar_bottom_row;
            attron(COLOR_PAIR(CP_IMAG) | A_BOLD);
            mvaddch(row, col_screen, '*');
            attroff(COLOR_PAIR(CP_IMAG) | A_BOLD);
        }
    }

    if (a->display == DISP_REAL || a->display == DISP_IMAG || a->display == DISP_ALL) {
        attron(COLOR_PAIR(CP_DIM) | A_DIM);
        for (int c = 1; c < plot_cols + 1 && c < cols - 1; c++) {
            mvaddch(baseline_row, c, '-');
        }
        attroff(COLOR_PAIR(CP_DIM) | A_DIM);
    }

    /* --- フッター: 凡例 + 統計値 --- */
    double *prob_arr = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        prob_arr[i] = creal(psi[i]) * creal(psi[i]) + cimag(psi[i]) * cimag(psi[i]);
    }
    double norm_total = trapz(prob_arr, x, n);

    int mid_idx = n / 2;
    double norm_left = trapz(prob_arr, x, mid_idx > 1 ? mid_idx : 1);
    double norm_right = norm_total - norm_left;
    free(prob_arr);

    attron(COLOR_PAIR(CP_TEXT));
    mvprintw(rows - 3, 0,
              " prob=#(yellow)  real=o(cyan)  imag=*(magenta)  barrier=%%(red)  well=:(blue) | [q]quit [p]pause [r]reset");
    mvprintw(rows - 2, 0,
              " norm=%.3f  left=%.3f  right(transmitted)=%.3f", norm_total, norm_left, norm_right);
    attroff(COLOR_PAIR(CP_TEXT));

    refresh();
}

/* ------------------------------------------------------------------ */
/* メイン                                                              */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    Args a;
    if (parse_args(argc, argv, &a) != 0) return 1;

    int n = next_pow2(a.grid_points);

    double *x = malloc(sizeof(double) * n);
    double *V = malloc(sizeof(double) * n);
    double *CAP = malloc(sizeof(double) * n);
    double complex *psi = malloc(sizeof(double complex) * n);

    double dx_phys = (2.0 * a.xmax) / (double)n;
    for (int i = 0; i < n; i++) x[i] = -a.xmax + i * dx_phys;

    build_potential(x, n, &a, V);
    build_absorber(x, n, &a, CAP);
    initial_psi(x, n, &a, psi);

    Evolver ev;
    evolver_init(&ev, x, V, CAP, a.dt, n);

    Scales sc;
    scales_init(&sc, x, n, &a, V);

    /* --- ncurses 初期化 --- */
    initscr();
    curs_set(0);
    noecho();
    cbreak();
    keypad(stdscr, TRUE);
    start_color();
    use_default_colors();
    init_pair(CP_PROB, COLOR_YELLOW, -1);
    init_pair(CP_REAL, COLOR_CYAN, -1);
    init_pair(CP_IMAG, COLOR_MAGENTA, -1);
    init_pair(CP_BARRIER_POS, COLOR_RED, -1);
    init_pair(CP_BARRIER_NEG, COLOR_BLUE, -1);
    init_pair(CP_TEXT, COLOR_WHITE, -1);
    init_pair(CP_DIM, COLOR_GREEN, -1);

    int frame_ms = (int)(1000.0 / (a.fps > 1e-6 ? a.fps : 1.0));
    if (frame_ms < 1) frame_ms = 1;
    timeout(frame_ms);

    double t = 0.0;
    int paused = 0;
    int restarting = 0;
    struct timespec restart_start;
    double t_max = 2.0 * (2.0 * a.xmax) / (fabs(a.k0) > 0.5 ? fabs(a.k0) : 0.5);

    for (;;) {
        int key = getch();
        if (key == 'q' || key == 'Q') break;
        else if (key == 'p' || key == 'P') paused = !paused;
        else if (key == 'r' || key == 'R') {
            initial_psi(x, n, &a, psi);
            t = 0.0;
            restarting = 0;
        }

        if (!paused) {
            if (!restarting) {
                for (int s = 0; s < a.steps_per_frame; s++) evolve_step(&ev, psi);
                t += a.dt * a.steps_per_frame;

                double *prob_arr = malloc(sizeof(double) * n);
                for (int i = 0; i < n; i++)
                    prob_arr[i] = creal(psi[i]) * creal(psi[i]) + cimag(psi[i]) * cimag(psi[i]);
                double norm_total = trapz(prob_arr, x, n);
                free(prob_arr);

                if ((norm_total < 0.03 || t > t_max) && !a.no_loop) {
                    restarting = 1;
                    clock_gettime(CLOCK_MONOTONIC, &restart_start);
                } else if (norm_total < 0.03 && a.no_loop) {
                    render(&a, x, n, V, psi, t, paused, 0, &sc);
                    napms(1500);
                    break;
                }
            } else {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (now.tv_sec - restart_start.tv_sec) +
                                  (now.tv_nsec - restart_start.tv_nsec) / 1e9;
                if (elapsed > 1.2) {
                    initial_psi(x, n, &a, psi);
                    t = 0.0;
                    restarting = 0;
                }
            }
        }

        render(&a, x, n, V, psi, t, paused, restarting, &sc);
    }

    endwin();

    evolver_free(&ev);
    free(x);
    free(V);
    free(CAP);
    free(psi);

    return 0;
}
