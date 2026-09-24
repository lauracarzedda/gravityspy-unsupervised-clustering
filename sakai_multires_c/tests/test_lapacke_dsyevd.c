#include <lapacke.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const lapack_int n = 3;
    const double original[9] = {
        4.0, 1.0, 1.0,
        1.0, 3.0, 0.0,
        1.0, 0.0, 2.0
    };
    double eigenvectors[9];
    double eigenvalues[3];
    const double residual_tolerance = 1.0e-10;
    const double orthonormality_tolerance = 1.0e-10;
    double max_residual = 0.0;
    double max_orthonormality_error = 0.0;
    lapack_int info;
    lapack_int i;
    lapack_int j;
    lapack_int k;
    int eigenvalues_ok = 1;
    int residuals_ok = 1;
    int orthonormality_ok = 1;

    for (i = 0; i < n * n; ++i) {
        eigenvectors[i] = original[i];
    }

    info = LAPACKE_dsyevd(
        LAPACK_COL_MAJOR, 'V', 'U', n, eigenvectors, n, eigenvalues);

    if (info != 0) {
        fprintf(stderr, "FAIL info == 0 (info=%lld)\n", (long long)info);
        return EXIT_FAILURE;
    }
    printf("PASS info == 0\n");

    for (i = 0; i < n; ++i) {
        if (!isfinite(eigenvalues[i])) {
            eigenvalues_ok = 0;
        }
    }
    for (i = 0; i + 1 < n; ++i) {
        if (eigenvalues[i] > eigenvalues[i + 1]) {
            eigenvalues_ok = 0;
        }
    }
    if (!eigenvalues_ok) {
        fprintf(stderr, "FAIL eigenvalues are finite and ascending\n");
        return EXIT_FAILURE;
    }
    printf("PASS eigenvalues are finite and ascending\n");

    for (j = 0; j < n; ++j) {
        for (i = 0; i < n; ++i) {
            double matrix_vector = 0.0;
            double residual;

            for (k = 0; k < n; ++k) {
                matrix_vector +=
                    original[i + k * n] * eigenvectors[k + j * n];
            }
            residual = fabs(
                matrix_vector - eigenvalues[j] * eigenvectors[i + j * n]);
            if (!isfinite(residual)) {
                residuals_ok = 0;
            } else if (residual > max_residual) {
                max_residual = residual;
            }
        }
    }
    if (!residuals_ok || max_residual > residual_tolerance) {
        fprintf(
            stderr,
            "FAIL eigenpair residuals (max=%.17g, tolerance=%.17g)\n",
            max_residual,
            residual_tolerance);
        return EXIT_FAILURE;
    }
    printf(
        "PASS eigenpair residuals (max=%.17g, tolerance=%.17g)\n",
        max_residual,
        residual_tolerance);

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            double dot_product = 0.0;
            double expected = (i == j) ? 1.0 : 0.0;
            double error;

            for (k = 0; k < n; ++k) {
                dot_product +=
                    eigenvectors[k + i * n] * eigenvectors[k + j * n];
            }
            error = fabs(dot_product - expected);
            if (!isfinite(error)) {
                orthonormality_ok = 0;
            } else if (error > max_orthonormality_error) {
                max_orthonormality_error = error;
            }
        }
    }
    if (!orthonormality_ok ||
        max_orthonormality_error > orthonormality_tolerance) {
        fprintf(
            stderr,
            "FAIL eigenvector orthonormality "
            "(max=%.17g, tolerance=%.17g)\n",
            max_orthonormality_error,
            orthonormality_tolerance);
        return EXIT_FAILURE;
    }
    printf(
        "PASS eigenvector orthonormality "
        "(max=%.17g, tolerance=%.17g)\n",
        max_orthonormality_error,
        orthonormality_tolerance);

    printf("ALL TESTS PASSED\n");
    return EXIT_SUCCESS;
}
