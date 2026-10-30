#include <bits/stdc++.h>
using namespace std;

struct Point { double x, y; };

// ================== БЛОК ВАРИАНТА 7 ==================
// Квадрат с отсеченной вершиной: |x| + |y| < 2, y < 1
// Вершины против часовой стрелки.
const vector<Point> DOMAIN = {
    {-2.0,  0.0},
    { 0.0, -2.0},
    { 2.0,  0.0},
    { 1.0,  1.0},
    {-1.0,  1.0}
};

double f(double x, double y)   { return 1.0; }
double phi(double x, double y) { return sin(x + y); }
double v0(double x, double y)  { return sin(x + y); } // продолжение phi в фиктивную область

// Проверка точки внутри многоугольника (ray casting).
// Для другого варианта можно заменить тело функции на нужное неравенство.
bool insideDomain(double x, double y) {
    bool inside = false;
    const int n = (int)DOMAIN.size();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const Point& pi = DOMAIN[i];
        const Point& pj = DOMAIN[j];
        bool intersect = ((pi.y > y) != (pj.y > y)) &&
                         (x < (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y) + pi.x);
        if (intersect) inside = !inside;
    }
    return inside;
}
// =====================================================

int main(int argc, char** argv) {
    int M = 80, N = 80;
    if (argc > 1) M = stoi(argv[1]);
    if (argc > 2) N = stoi(argv[2]);

    // Bounding box по многоугольнику
    double A1 =  1e100, B1 = -1e100;
    double A2 =  1e100, B2 = -1e100;
    for (const auto& p : DOMAIN) {
        A1 = min(A1, p.x); B1 = max(B1, p.x);
        A2 = min(A2, p.y); B2 = max(B2, p.y);
    }

    const double h1 = (B1 - A1) / M;
    const double h2 = (B2 - A2) / N;
    const double h  = max(h1, h2);
    const double eps = h * h;          // по заданию
    const double inv_h1_2 = 1.0 / (h1 * h1);
    const double inv_h2_2 = 1.0 / (h2 * h2);

    const int Nx = M - 1;
    const int Ny = N - 1;
    const int n  = Nx * Ny;

    auto idx = [Ny](int i, int j) { return (i - 1) * Ny + (j - 1); };
    auto X   = [A1, h1](int i) { return A1 + i * h1; };
    auto Y   = [A2, h2](int j) { return A2 + j * h2; };

    vector<double> Q(n), F(n), B(n);

    // Вычисление Q_ij, F_ij через долю площади ячейки в D.
    // Для гибкости используется подсетка K x K; точность повышается увеличением K.
    const int K = 12;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= M - 1; ++i) {
        for (int j = 1; j <= N - 1; ++j) {
            const double xl = A1 + (i - 0.5) * h1;
            const double yl = A2 + (j - 0.5) * h2;

            int cnt = 0;
            double fsum = 0.0;
            for (int a = 0; a < K; ++a) {
                for (int b = 0; b < K; ++b) {
                    const double x = xl + (a + 0.5) * h1 / K;
                    const double y = yl + (b + 0.5) * h2 / K;
                    if (insideDomain(x, y)) {
                        ++cnt;
                        fsum += f(x, y);
                    }
                }
            }

            const double frac = double(cnt) / double(K * K);
            const int k = idx(i, j);
            Q[k] = (1.0 - frac) / (eps * eps);
            F[k] = fsum / double(K * K);
        }
    }

    // Правая часть B = F + Q*v0 + вклад граничных узлов
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= M - 1; ++i) {
        for (int j = 1; j <= N - 1; ++j) {
            const int k = idx(i, j);
            double rhs = F[k] + Q[k] * v0(X(i), Y(j));

            if (i == 1)     rhs += v0(X(0), Y(j)) * inv_h1_2;
            if (i == M - 1) rhs += v0(X(M), Y(j)) * inv_h1_2;
            if (j == 1)     rhs += v0(X(i), Y(0)) * inv_h2_2;
            if (j == N - 1) rhs += v0(X(i), Y(N)) * inv_h2_2;

            B[k] = rhs;
        }
    }

    vector<double> w(n, 0.0), r(n), z(n), p(n), Ap(n);

    auto applyA = [&](const vector<double>& u, vector<double>& out) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= M - 1; ++i) {
            for (int j = 1; j <= N - 1; ++j) {
                const int k = idx(i, j);
                double val = (2.0 * inv_h1_2 + 2.0 * inv_h2_2 + Q[k]) * u[k];

                if (i > 1)     val -= u[idx(i - 1, j)] * inv_h1_2;
                if (i < M - 1) val -= u[idx(i + 1, j)] * inv_h1_2;
                if (j > 1)     val -= u[idx(i, j - 1)] * inv_h2_2;
                if (j < N - 1) val -= u[idx(i, j + 1)] * inv_h2_2;

                out[k] = val;
            }
        }
    };

    auto dot = [&](const vector<double>& a, const vector<double>& b) {
        double s = 0.0;
        #pragma omp parallel for reduction(+:s) schedule(static)
        for (int k = 0; k < n; ++k) s += a[k] * b[k];
        return s;
    };

    // Начальное приближение w = 0 => r = B
    r = B;

    // Диагональное предобуславливание
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) {
        const double D = 2.0 * inv_h1_2 + 2.0 * inv_h2_2 + Q[k];
        z[k] = r[k] / D;
    }

    p = z;
    double rz = dot(z, r);

    const double delta = 1e-8;
    const int maxIter = n;

    auto t0 = chrono::steady_clock::now();
    int it = 0;

    for (it = 1; it <= maxIter; ++it) {
        applyA(p, Ap);
        const double denom = dot(Ap, p);
        if (denom <= 0.0) break;

        const double alpha = rz / denom;

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            w[k] += alpha * p[k];
            r[k] -= alpha * Ap[k];
        }

        double diff2 = 0.0;
        #pragma omp parallel for reduction(+:diff2) schedule(static)
        for (int k = 0; k < n; ++k) {
            const double d = alpha * p[k];
            diff2 += d * d;
        }
        if (sqrt(diff2) < delta) break;

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            const double D = 2.0 * inv_h1_2 + 2.0 * inv_h2_2 + Q[k];
            z[k] = r[k] / D;
        }

        const double rzNew = dot(z, r);
        const double beta = rzNew / rz;

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            p[k] = z[k] + beta * p[k];
        }
        rz = rzNew;
    }

    auto t1 = chrono::steady_clock::now();
    const double time = chrono::duration<double>(t1 - t0).count();

    // Невязка и максимум решения
    applyA(w, Ap);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) r[k] = B[k] - Ap[k];

    const double res = sqrt(dot(r, r));
    double maxw = 0.0;
    for (double val : w) maxw = max(maxw, fabs(val));

    cout << "Variant 7: |x|+|y|<2, y<1\n";
    cout << "M=" << M << " N=" << N << "\n";
    cout << "h1=" << h1 << " h2=" << h2 << " eps=" << eps << "\n";
    cout << "iterations=" << it << "\n";
    cout << "time=" << time << " sec\n";
    cout << "||r||_E=" << res << "\n";
    cout << "max|w|=" << maxw << "\n";

    return 0;
}