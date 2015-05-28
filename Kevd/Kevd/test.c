// Copyright by Shuhei Kudo, May 2015.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include "dsyevdt.h"

extern void dgemm_(const char*, const char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

static void calc_error(int n, double* a, int lda, double* p, int ldp, double* d, double* work)
{
	int i, j;
	double one = 1.;
	double zero = 0.;
	double x = 0.;
	double t=0.;
	double t2=0.;
	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j)
			x += a[i*lda + j] * a[i*lda + j];
	dgemm_("T", "N", &n, &n, &n, &one, p, &ldp, p, &ldp, &zero, work, &n);
	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j)
			t += (i != j ? work[i*n + j] * work[i*n + j] : 0.);

	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j)
			work[i*lda + j] = p[i*ldp + j] * d[i];
	dgemm_("N", "T", &n, &n, &n, &one, work, &ldp, p, &ldp, &zero, work + n*lda, &lda);
	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j){
			double tt = a[lda*i + j] - work[n*lda + i*lda + j];
			t2 += tt*tt;
		}

	printf("ort=%.15e\nres=%.15e\n", sqrt(t), sqrt(t2/x));
}

static void set_matrix(int n, double* a, int lda, int type)
{
	int i, j;
	if (type == 0){
		for (i = 0; i < n; ++i)
			for (j = 0; j < n; ++j)
				a[i*lda + j] = (double)(i < j ? i : j) + 1;
	}
}


int main()
{
	int n = 213;
	int lda = (n + 1) / 2 * 2;
	int lw, liw;
	dsyevdt_worksizes('V', 'U', n, omp_get_max_threads(), &lw, &liw);


	double* a = (double*)bje_alloc(sizeof(double)*n*lda);
	double* as = (double*)bje_alloc(sizeof(double)*n*lda);
	double* d = (double*)bje_alloc(sizeof(double)*lda);
	double* w = (double*)bje_alloc(sizeof(double)*lw);
	int* iw = (int*)bje_alloc(sizeof(int)*liw);

	if (!a || !as || !w || !iw) abort();
	set_matrix(n, a, lda, 0);
	set_matrix(n, as, lda, 0);

	dsyevdt('V', 'U', n, a, lda, d, w, lw, iw, liw);
	calc_error(n, as, lda, a, lda, d, w);

	bje_free(a); bje_free(as); bje_free(d); bje_free(w); bje_free(iw);
	return 0;
}