#ifdef _OPENMP
#include <omp.h>
#endif

#include <bits/stdc++.h>
using namespace std;

struct Point { double x, y; };

// v7: |x| + |y| < 2, y < 1
const vector<Point> domain_polygon = {
    {-2.0,  0.0},
    { 0.0, -2.0},
    { 2.0,  0.0},
    { 1.0,  1.0},
    {-1.0,  1.0}
};

double f(double x, double y)   { return 1.0; }
double phi(double x, double y) { return sin(x + y); }
double v0(double x, double y)  { return sin(x + y); }

bool insideDomain(double x, double y) {
    bool inside = false;
    const int n = (int)domain_polygon.size();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const Point& pi = domain_polygon[i];
        const Point& pj = domain_polygon[j];
        // y between pi.y and pj.y
        const bool isCorrectY = (pi.y > y) != (pj.y > y);
        if (!isCorrectY) continue;
        // y - pi.y = delta y / delta x + (x - pi.x)
        const auto xOfY = (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y) + pi.x;
        if (x < xOfY) inside = !inside;
    }
    return inside;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024;
    if (argc > 1) M = stoi(argv[1]);
    if (argc > 2) N = stoi(argv[2]);

    double A1 =  1e100, B1 = -1e100;
    double A2 =  1e100, B2 = -1e100;
    const double pad = 0.25;
    for (const auto& p : domain_polygon) {
        A1 = min(A1, p.x - pad); B1 = max(B1, p.x + pad);
        A2 = min(A2, p.y - pad); B2 = max(B2, p.y + pad);
    }

    const double h1 = (B1 - A1) / M;
    const double h2 = (B2 - A2) / N;
    const double h  = max(h1, h2);
    const double eps = h * h;
    const double inv_h1_2 = 1.0 / (h1 * h1);
    const double inv_h2_2 = 1.0 / (h2 * h2);

    const int Nx = M - 1;
    const int Ny = N - 1;
    const int n  = Nx * Ny;

    auto idx = [Ny](int i, int j) { return (i - 1) * Ny + (j - 1); };
    auto X   = [A1, h1](int i) { return A1 + i * h1; };
    auto Y   = [A2, h2](int j) { return A2 + j * h2; };

    vector<double> Q(n), F(n), B(n);

    const int K = 100; // accurancy
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

    vector<double> w(n, 0.0), r = B, z(n), p(n), Ap(n);

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

    auto applyPreconditioner = [&]() {
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            const double D = 2.0 * inv_h1_2 + 2.0 * inv_h2_2 + Q[k];
            z[k] = r[k] / D;
        }
    };

    // Evaluate J(w) directly instead of deriving it from the recursively updated residual.
    auto functionalAndScale = [&](const vector<double>& u) {
        applyA(u, Ap);
        double sumAu = 0.0;
        double sumBu = 0.0;
        double absSum = 0.0;
        #pragma omp parallel for reduction(+:sumAu,sumBu,absSum) schedule(static)
        for (int k = 0; k < n; ++k) {
            const double au = Ap[k] * u[k];
            const double bu = B[k] * u[k];
            sumAu += au;
            sumBu += bu;
            absSum += 0.5 * abs(au) + abs(bu);
        }
        const double cellArea = h1 * h2;
        return pair<double, double>{
            cellArea * (0.5 * sumAu - sumBu),
            cellArea * absSum
        };
    };

    applyPreconditioner();
    p = z;
    double rz = dot(z, r);

    const double delta = 1e-8;
    const int maxIter = n;
    const double cellArea = h1 * h2;
    // Allow for accumulation error in the parallel reductions.
    const double roundoffFactor =
        8.0 * (n + 1.0) * numeric_limits<double>::epsilon();
    vector<double> lastMonotoneW = w;
    double lastJ = 0.0;
    double lastJScale = 0.0;
    double stepNorm = 0.0;
    int restarts = 0;
    int iterations = 0;
    bool converged = false;
    bool breakdown = false;

    #ifdef _OPENMP
    double t0 = omp_get_wtime();
    #else
    auto t0 = std::chrono::steady_clock::now();
    #endif

    while (iterations < maxIter && !converged && !breakdown) {
        ++iterations;
        if (!(rz > 0.0) || !isfinite(rz)) {
            breakdown = true;
            break;
        }

        applyA(p, Ap);
        const double denom = dot(Ap, p);
        if (!(denom > 0.0) || !isfinite(denom)) {
            breakdown = true;
            break;
        }

        const double alpha = rz / denom;
        if (!isfinite(alpha)) {
            breakdown = true;
            break;
        }

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            w[k] += alpha * p[k];
            r[k] -= alpha * Ap[k];
        }

        stepNorm = abs(alpha) * sqrt(cellArea * dot(p, p));

        const auto [candidateJ, candidateJScale] = functionalAndScale(w);
        if (!isfinite(stepNorm) || !isfinite(candidateJ) ||
            !isfinite(candidateJScale)) {
            w = lastMonotoneW;
            breakdown = true;
            break;
        }
        const double jTolerance = roundoffFactor *
            max({lastJScale, candidateJScale, abs(lastJ), abs(candidateJ)});
        if (candidateJ > lastJ + jTolerance) {
            w = lastMonotoneW;
            applyA(w, Ap);
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n; ++k) r[k] = B[k] - Ap[k];
            applyPreconditioner();
            p = z;
            rz = dot(z, r);
            stepNorm = 0.0;
            ++restarts;
            continue;
        }

        lastJ = candidateJ;
        lastJScale = candidateJScale;
        lastMonotoneW = w;
        if (stepNorm < delta) {
            converged = true;
            break;
        }

        applyPreconditioner();
        const double rzNew = dot(z, r);
        if (!(rzNew > 0.0) || !isfinite(rzNew)) {
            breakdown = true;
            break;
        }
        const double beta = rzNew / rz;

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            p[k] = z[k] + beta * p[k];
        }
        rz = rzNew;
    }

    #ifdef _OPENMP
    double time = omp_get_wtime() - t0;
    #else
    double time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    #endif


    applyA(w, Ap);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) r[k] = B[k] - Ap[k];

    const double res = sqrt(cellArea * dot(r, r));
    double maxw = 0.0;
    for (double val : w) maxw = max(maxw, fabs(val));

    cout << "Variant 7: |x|+|y|<2, y<1\n";
    cout << "M=" << M << " N=" << N << "\n";
    cout << "h1=" << h1 << " h2=" << h2 << " eps=" << eps << "\n";
    cout << "iterations=" << iterations << "\n";
    cout << "time=" << time << " sec\n";
    cout << "||r||_E=" << res << "\n";
    cout << "step_norm_E=" << stepNorm << " delta=" << delta << "\n";
    cout << "J=" << lastJ << " J_restarts=" << restarts << "\n";
    cout << "converged=" << (converged ? "yes" : "no") << "\n";
    cout << "max|w|=" << maxw << "\n";
    if (breakdown) {
        cerr << "CG breakdown after " << iterations << " iteration(s)\n";
    } else if (!converged) {
        cerr << "CG did not converge within " << maxIter << " iteration(s)\n";
    }

    std::ofstream fout("solution.csv");
    for (int i = 1; i <= M - 1; ++i)
        for (int j = 1; j <= N - 1; ++j)
            fout << X(i) << " " << Y(j) << " " << w[idx(i,j)] << "\n";

    return converged ? 0 : 1;
}