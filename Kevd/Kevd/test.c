// Copyright by Shuhei Kudo, May 2015.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "common.h"
#include "dsyevdt.h"

extern void dgemm_(const char*, const char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

static void calc_error(int n, double* a, int lda, double* p, int ldp, double* d)
{
	int i, j;
	double one = 1.;
	double zero = 0.;
	double x = 0.;
	double t=0.;
	double t2=0.;
	double* work =(double*)bje_alloc(n*n*sizeof(double));
	if(!work) abort();

	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j)
			x += a[i*lda + j] * a[i*lda + j];
	// calculate orthgonality  || P^T P - I||_F
	dgemm_("T", "N", &n, &n, &n, &one, p, &ldp, p, &ldp, &zero, work, &n);
	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j) {
			double tt = work[i*n + j] - (i!=j ? 0.: 1.);
			t += tt * tt;
		}

	// calculate residual ||AP - P D||_F
	dgemm_("N", "N", &n, &n, &n, &one, a, &lda, p, &ldp, &zero, work, &n);
	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j){
			double tt = work[i*n + j] - p[i*ldp + j] * d[i];
			t2 += tt*tt;
		}

	printf("ort=%.15e\nres=%.15e\n", sqrt(t), sqrt(t2/x));
	bje_free(work);
}

static void set_matrix(int n, double* a, int lda, int type)
{
	int i, j;
	switch(type){
	default:
	case 0:
		for (i = 0; i < n; ++i)
			for (j = 0; j < n; ++j)
				a[i*lda + j] = (double)(i < j ? i : j) + 1;
		break;
	case 1:
		for (i = 0; i < n; ++i)
			for (j = i; j < n; ++j)
				a[j*lda + i] = a[i*lda + j] = (double)rand() / RAND_MAX;
		break;
	}
}


int main()
{
	int n = 213;
	int lda = (n + 1) / 2 * 2;


	double* a = (double*)bje_alloc(sizeof(double)*n*lda);
	double* as = (double*)bje_alloc(sizeof(double)*n*lda);
	double* d = (double*)bje_alloc(sizeof(double)*lda);
	if (!a || !as || !d) abort();
	set_matrix(n, as, lda, 1);


	// test 1
	memcpy(a, as, n*lda*sizeof(double));
	dsyevdeasy_(&n, a, &lda, d);
	calc_error(n, as, lda, a, lda, d);

	// test 2
	int lw, liw;
	dsyevdt_worksizes('V', 'U', n, omp_get_max_threads(), &lw, &liw);
	double* w = (double*)bje_alloc(sizeof(double)*lw);
	int* iw = (int*)bje_alloc(sizeof(int)*liw);
	if (!w || !iw) abort();

	memcpy(a, as, n*lda*sizeof(double));
	dsyevdt('V', 'U', n, a, lda, d, w, lw, iw, liw);
	calc_error(n, as, lda, a, lda, d);

	bje_free(a); bje_free(as); bje_free(d); bje_free(w); bje_free(iw);
	return 0;
}