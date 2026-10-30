#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define REGION_NAME "v7: |x|+|y|<2, y<1"
#define XMIN (-2.1)
#define XMAX ( 2.1)
#define YMIN (-2.1)
#define YMAX ( 1.1)

static int inside(double x, double y) {
    return fabs(x) + fabs(y) < 2.0 && y < 1.0;
}


#define NSUB 16
#define IDX(i,j) ((size_t)(i)*N1 + (j))

static double *xcalloc(size_t n) {
    double *p = calloc(n, sizeof(double));
    if (!p) { fprintf(stderr, "calloc(%zu) failed\n", n); exit(1); }
    return p;
}

static void apply_A(const double * restrict w, double * restrict out,
                    const double * restrict Q,
                    double ih1sq, double ih2sq, int M, int N, int N1) {
    for (int i = 1; i < M; i++)
        for (int j = 1; j < N; j++) {
            size_t k = IDX(i,j);
            double lap = (w[k-N1] + w[k+N1] - 2.0*w[k]) * ih1sq
                       + (w[k-1]  + w[k+1]  - 2.0*w[k]) * ih2sq;
            out[k] = -lap + Q[k]*w[k];
        }
}

static double dot(const double * restrict a, const double * restrict b,
                  int M, int N, int N1, double h1, double h2) {
    double s = 0.0;
    for (int i = 1; i < M; i++)
        for (int j = 1; j < N; j++) s += a[IDX(i,j)] * b[IDX(i,j)];
    return s * h1 * h2;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 40;
    int N = (argc > 2) ? atoi(argv[2]) : 40;
    const double delta = 1e-8;
    const int max_iter = 200000;

    double h1 = (XMAX - XMIN) / M;
    double h2 = (YMAX - YMIN) / N;
    double h  = fmax(h1, h2);
    double eps = h * h;                    /* см. претензию в прошлом ответе */
    double inv_eps2 = 1.0 / (eps * eps);
    double ih1sq = 1.0 / (h1 * h1);
    double ih2sq = 1.0 / (h2 * h2);

    int M1 = M + 1, N1 = N + 1;
    size_t sz = (size_t)M1 * N1;

    double *w  = xcalloc(sz), *B  = xcalloc(sz);
    double *r  = xcalloc(sz), *z  = xcalloc(sz);
    double *p  = xcalloc(sz), *Ap = xcalloc(sz);
    double *Q  = xcalloc(sz), *D  = xcalloc(sz);

    /* Q, B, D — через подвыборку NSUB×NSUB на ячейку */
    for (int i = 1; i < M; i++) {
        double xc = XMIN + i * h1;
        for (int j = 1; j < N; j++) {
            double yc = YMIN + j * h2;
            int cnt = 0;
            for (int si = 0; si < NSUB; si++) {
                double xs = xc + ((si + 0.5)/NSUB - 0.5) * h1;
                for (int sj = 0; sj < NSUB; sj++) {
                    double ys = yc + ((sj + 0.5)/NSUB - 0.5) * h2;
                    cnt += inside(xs, ys);
                }
            }
            double frac = (double)cnt / (NSUB * NSUB);
            size_t k = IDX(i,j);
            Q[k] = (1.0 - frac) * inv_eps2;
            B[k] = frac + Q[k] * sin(xc + yc);
            D[k] = 2.0*ih1sq + 2.0*ih2sq + Q[k];
        }
    }

    /* Дирихле w = sin(x+y) на Γ */
    for (int i = 0; i <= M; i++) {
        double x = XMIN + i * h1;
        w[IDX(i, 0)] = sin(x + YMIN);
        w[IDX(i, N)] = sin(x + YMAX);
    }
    for (int j = 0; j <= N; j++) {
        double y = YMIN + j * h2;
        w[IDX(0, j)] = sin(XMIN + y);
        w[IDX(M, j)] = sin(XMAX + y);
    }

    /* r = B - Aw; z = D^{-1}r; p = z */
    apply_A(w, r, Q, ih1sq, ih2sq, M, N, N1);
    for (int i = 1; i < M; i++)
        for (int j = 1; j < N; j++) {
            size_t k = IDX(i,j);
            r[k] = B[k] - r[k];
            z[k] = r[k] / D[k];
            p[k] = z[k];
        }
    double rho_old = dot(z, r, M, N, N1, h1, h2);
    double r0 = sqrt(dot(r, r, M, N, N1, h1, h2));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int it;
    double step = 0.0;
    for (it = 1; it <= max_iter; it++) {
        apply_A(p, Ap, Q, ih1sq, ih2sq, M, N, N1);
        double Ap_p = dot(Ap, p, M, N, N1, h1, h2);
        if (Ap_p <= 0.0) { fprintf(stderr, "CG breakdown, it=%d\n", it); break; }
        double alpha = rho_old / Ap_p;
        step = alpha * sqrt(dot(p, p, M, N, N1, h1, h2));

        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                size_t k = IDX(i,j);
                w[k] += alpha * p[k];
                r[k] -= alpha * Ap[k];
                z[k]  = r[k] / D[k];
            }
        double rho_new = dot(z, r, M, N, N1, h1, h2);
        double beta = rho_new / rho_old;
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                size_t k = IDX(i,j);
                p[k] = z[k] + beta * p[k];
            }
        rho_old = rho_new;
        if (step < delta) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);
    double rnorm = sqrt(dot(r, r, M, N, N1, h1, h2));

    printf("region : %s\n", REGION_NAME);
    printf("grid   : %d x %d   unknowns=%d\n", M, N, (M-1)*(N-1));
    printf("h1,h2  : %.3e  %.3e   eps=h^2=%.3e\n", h1, h2, eps);
    printf("iters  : %d%s\n", it, it > max_iter ? "  (MAX_ITER HIT)" : "");
    printf("time   : %.4f s\n", elapsed);
    printf("resid  : ||r||/||r0|| = %.3e\n", rnorm / r0);
    printf("last   : step = %.3e   (delta=%.1e)\n", step, delta);

    FILE *f = fopen("sol.dat", "w");
    if (f) {
        for (int i = 0; i <= M; i++) {
            double x = XMIN + i * h1;
            for (int j = 0; j <= N; j++)
                fprintf(f, "%.6e %.6e %.6e\n", x, YMIN + j*h2, w[IDX(i,j)]);
            fprintf(f, "\n");
        }
        fclose(f);
    }

    free(w); free(B); free(r); free(z); free(p); free(Ap); free(Q); free(D);
    return 0;
}