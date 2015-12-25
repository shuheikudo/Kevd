// Copyright by Shuhei Kudo, May 2015.
//
// This source code is derived from LAPACK source code.

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "common.h"
#include "timed.h"
#include "dsytdb.h"
#include "dsyevdt.h"
#include "dormtrb.h"
#include "simd_tools.h"


static int MAXI(int a, int b) { return a < b ? b : a; }
static int MINI(int a, int b) { return a < b ? a : b; }

extern double dlamch_(const char*);
extern void dlascl_(const char*, const int*, const int*, const double*, const double*, const int*, const int*, double*, const int*, int*);
static int dlascl_c(char uplo, int kl, int ku, double cfrom, double cto, int m, int n, double* a, int lda)
{
	int info;
	dlascl_(&uplo, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);
	return info;
}

extern void dscal_(const int*, const double*, double*, const int*);
static void dscal_c(int n, double scl, double* a, int step)
{
	dscal_(&n, &scl, a, &step);
}

extern double dlanst_(const char*, const int*, const double*, const double*);

extern void dcopy_(const int*, const double*, const int*, double*, const int*);
void dcopy_c(int n, const double* a, int t1, double* d, int t2)
{
	dcopy_(&n, a, &t1, d, &t2);
}

extern void dswap_(const int*, double*, const int*, double*, const int*);

extern void dlacpy_(const char*, const int*, const int*, const double*, const int*, double*, const int*);
static void dlacpy_c(char uplo, int m, int n, const double* a, int lda, double* b, int ldb)
{
	dlacpy_(&uplo, &m, &n, a, &lda, b, &ldb);
}

extern void dlaset_(const char*, const int*, const int*, const double*, const double*, double*, const int*);
extern double dlansy_(const char*, const char*, const int*, double*, const int*, double*);
extern void dlaed1_(const int*, double*, double*, const int*, int*, double*, int*, double*, int*, int*);

extern void dsytrd_(const char*, const int*, double*, const int*, double*, double*, double*, double*, const int*, int*);
extern void dsteqr_(const char*, const int*, double*, double*, double*, const int*, double*, int*);
extern void dormtr_(const char*, const char*, const char*, const int*, const int*, double*, const int*, double*, double*, const int*, double*, const int*, int*);


static int dormtrb_worksize(int n, int num_threads)
{
	int n4 = (n + 15) / 16 * 16;
	return n4 * (n4 + 1) / 2 + (num_threads) * 8 * n4;
}

static int lwork_size(int n, int num_threads) 
{
	n = (n + 1) / 2 * 2;
	return  MAXI(1 + 2 * n + 3 * n * n, 2 * n + MAXI(n * n + MAXI(1 + n * n, dormtrb_worksize(n, num_threads)), dsytdd_worksize(n, num_threads)));
}

static int check_arg(char jobz, char uplo, int n, int lda, int lwork, int liwork)
{
	if (jobz != 'V') return -1;
	else if ( uplo != 'U') return -2;
	else if (n < 0) return -3;
	else if (lda < MAXI(1, n)) return -5;

	if (n <= 1){
		if (lwork < 1) return -8;
		if (liwork < 1) return -10;
	}
	else{
		// sanity_check...
		if (lwork < lwork_size(n,1)) return -8;
		if (liwork < 3 + 6 * n) return -10;

		// alignment
		if (lda % 2) return -11;
	}
	return 0;
}

void dsyevdt_worksizes(char jobz, char uplo, int n, int num_threads, int* lwork, int* liwork)
{
	jobz = jobz; // unused
	uplo = uplo; // unused
	if (n <= 1){
		*lwork = *liwork = 1;
	}
	else {
		*liwork = 3 + 6 * n;
		*lwork = lwork_size(n, num_threads);
	}
}
extern void dstedc_(const char*, int*, double*, double*, double*, int*, double*, int*, int*, int*, int*);

int dstedcI(int N, double* D, double* E, double* Z, int LDZ, double* WORK, int LWORK, int* IWORK, int LIWORK);
static int dlaed0I(int N, double* D, double* E, double* Q, int LDQ, double* WORK, int* IWORK);

static int dlaed1x(int n, double* d, double* q, int ldq, int* indxq, double rho, int cutpnt, double* work, int* iwork);

void dsyevdeasy_(int* n, double* a, int* lda, double* w)
{
	int mx = omp_get_max_threads();
	int lwork, liwork;
	dsyevdt_worksizes('V', 'U', *n, mx, &lwork, &liwork);
	int* iwork = (int*)bje_alloc(liwork*sizeof(int));
	double* work = (double*)bje_alloc(lwork*sizeof(double));
	if (!iwork || !work) abort();
	if ((*lda) % 2){
		int i;
		int lda2 = *n + 1;
		double* t, *t2;
		if (!(t = (double*)bje_alloc(*n*lda2*sizeof(double)))) abort();
		if (!(t2 = (double*)bje_alloc(lda2*sizeof(double)))) abort();
		for (i = 0; i < *n; ++i)
			memcpy(t + lda2*i, a + *n * i, sizeof(double)**n);
		if (dsyevdt('V', 'U', *n, t, lda2, t2, work, lwork, iwork, liwork)) abort();
		for (i = 0; i < *n; ++i)
			memcpy(a + *n * i, t + lda2*i, sizeof(double)**n);
		memcpy(t, t2, sizeof(double)*lda2);
		bje_free(t);
		bje_free(t2);
	}
	else
		if (dsyevdt('V', 'U', *n, a, *lda, w, work, lwork, iwork, liwork)) abort();
	bje_free(work);
	bje_free(iwork);
}

// fortran interface
void dsyevdt_(const char* jobz, const char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int * info)
{
	if (*lwork == -1){
		dsyevdt_worksizes(*jobz, *uplo, *n, omp_get_max_threads(), lwork, liwork);
	}
	else {
		*info = dsyevdt(*jobz, *uplo, *n, a, *lda, w, work, *lwork, iwork, *liwork);
	}
}

int dsyevdt(char jobz, char uplo, int n, double* a, int lda, double* w, double* work, int lwork, int* iwork, int liwork)
{
	// Quick return if possible
	if (n == 0) return 0;
	if (n == 1) {
		w[0] = a[0];
		a[0] = 1.0;
		return 0;
	}

	int info = 0;
	if ((info = check_arg(jobz, uplo, n, lda, lwork, liwork))) return info;
	if (!is_octword_aligned(a) || !is_octword_aligned(w) || !is_octword_aligned(work)) return -11;

	// Get machine constants.
	double safemin = dlamch_("Safe minimum");
	double eps = dlamch_("Precision");
	double rmin = sqrt(safemin / eps);
	double rmax = sqrt(eps / safemin);

	// Scale matrix to allowable range, if necessary.
	double sigma = 0.;
	double anrm = dlansy_("M", &uplo, &n, a, &lda, work);
	if (anrm > 0. && anrm < rmin) sigma = rmin / anrm;
	else if (anrm > rmax) sigma = rmax / anrm;

	if (sigma != 0.) dlascl_c(uplo, 0, 0, 1.0, sigma, n, n, a, lda);

	// Call DSYTRD to reduce symmetric matrix to tridiagonal form.
	int neven = (n + 1) / 2 * 2;
	double* wtau = work + neven; // must be 16-byte aligned
	double* wwrk = wtau + neven; // also

	MEASUREI(info, "dsytrd", dsytdc, n, a, lda, w, work, wtau, wwrk);
	if (info) return info;

	int ldc = (n + 1) / 2 * 2;
	double* wwk2 = wwrk + n * ldc;
	int LLWRK2 = lwork - 2 * neven - n*ldc;
	MEASUREI(info, "dsyedc", dstedcI, n, w, work, wwrk, ldc, wwk2, LLWRK2, iwork, liwork);
	if (info) return info;

	MEASUREI(info, "dormtr", dormtrb, n, n, a, lda, wtau, wwrk, ldc, wwk2);
	if (info) return info;

	dlacpy_("A", &n, &n, wwrk, &ldc, a, &lda);

	if (sigma != 0.) dscal_c(n, 1.0 / sigma, w, 1);
	return 0;
}

int dstedcI(int N, double* D, double* E, double* Z, int LDZ, double* WORK, int LWORK, int* IWORK, int LIWORK)
{
	double   ZERO = 0.0, ONE = 1.0;
	int LQUERY;
	int FINISH, I, II, J, K, LIWMIN, LWMIN, START;

	int INFO = 0;
	LQUERY = (LWORK == -1 || LIWORK == -1);

	if (N < 0)
		return -2;
	else if (LDZ < 1 || LDZ < N)
		return -6;

	// Compute the workspace requirements
	int SMLSIZ = 25;
	if (N < 1){
		LIWMIN = 1;
		LWMIN = 1;
	}
	else if (N < SMLSIZ){
		LIWMIN = 1;
		LWMIN = 2 * (N - 1);
	}
	else {
		LWMIN = 1 + 4 * N + N*N;
		LIWMIN = 3 + 5 * N;
	}
	WORK[0] = LWMIN;
	IWORK[0] = LIWMIN;

	if (LWORK < LWMIN && !LQUERY)
		return -8;
	else if (LIWORK < LIWMIN && !LQUERY)
		return -10;
	if (LQUERY) return 0;


	if (N <= SMLSIZ){
		dsteqr_("I", &N, D, E, Z, &LDZ, WORK, &INFO);
		return INFO;
	}

	dlaset_("Full", &N, &N, &ZERO, &ONE, Z, &LDZ);

	double ORGNRM = dlanst_("M", &N, D, E);
	if (ORGNRM == ZERO) return INFO;
	double EPS = dlamch_("Epsilon");

	for (START = 0; START < N - 1; START = FINISH + 1){
		for (FINISH = START; FINISH < N - 1; ++FINISH){
			double tiny = EPS*sqrt(fabs(D[FINISH]))
				* sqrt(fabs(D[FINISH + 1]));
			if (fabs(E[FINISH]) <= tiny) break;
		}

		// (Sub) Problem determined.  Compute its size and solve it.
		int M = FINISH - START + 1;
		if (M == 1) continue;

		if (M > SMLSIZ){
			// Scale.
			double nrm = dlanst_("M", &M, D + START, E + START);
			dlascl_c('G', 0, 0, nrm, 1.0, M, 1, D + START, M);
			dlascl_c('G', 0, 0, nrm, 1.0, M - 1, 1, E + START, M - 1);

			INFO = dlaed0I(M, D + START, E + START, Z + START + LDZ*START,
				LDZ, WORK, IWORK);
			if (INFO != 0)
				return (INFO / (M + 1) + (START + 1))*(N + 1) + INFO % (M + 1) + START + 1;

			// Scale back.
			dlascl_c('G', 0, 0, 1.0, nrm, M, 1, D + START, M);
		}
		else
			dsteqr_("I", &M, D + START, E + START,
			Z + START + LDZ*START, &LDZ, WORK, &INFO);

		if (INFO != 0)
			return (START + 1)*(N + 1) + FINISH + 1;
	}

	// if problem was splitted
	if (START != 0) {
		// Use Selection Sort to minimize swaps of eigenvectors
		for (II = 1; II < N; ++II){
			I = II - 1;
			K = I;
			double p = D[I];
			for (J = II; J < N; ++J){
				if (D[J] < p) {
					K = J;
					p = D[J];
				}
			}
			if (K != I) {
				D[K] = D[I];
				D[I] = p;
				int t = 1;
				dswap_(&N, Z + LDZ*I, &t, Z + LDZ*K, &t);
			}
		}
	}
	return INFO;
}
extern void dlaed2_();
extern void dlaed3_();
extern void dlaed3x_();
extern void dlamrg_();
static int dlaed0I(int N, double* D, double* E, double* Q, int LDQ, double* WORK, int* IWORK)
{

	int I, J;
	int INFO = 0;
	int SMLSIZ = 25;
	// Determine the size and placement of the submatrices, and save in
	// the leading elements of IWORK.
	IWORK[0] = N;
	int SUBPBS = 1;
	for (; IWORK[SUBPBS - 1] > SMLSIZ; SUBPBS *= 2)
		for (J = SUBPBS - 1; J >= 0; --J){
		IWORK[2 * J + 1] = (IWORK[J] + 1) / 2;
		IWORK[2 * J] = (IWORK[J]) / 2;
		}
	for (J = 0; J < SUBPBS - 1; ++J) IWORK[J + 1] += IWORK[J];

	// Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
	// using rank-1 modifications (cuts).
	for (I = 0; I < SUBPBS - 1; ++I){
		int indx = IWORK[I];
		double sd = fabs(E[indx - 1]);
		D[indx - 1] = D[indx - 1] - sd;
		D[indx] = D[indx] - sd;
	}

	int* indxq = IWORK + 5 * N + 3;
	// Solve each submatrix eigenproblem at the bottom of the divide and
	// conquer tree.

#pragma omp parallel
	{
		int id = omp_get_thread_num();
#pragma omp for
		for (I = 0; I < SUBPBS; ++I){
			int indx, size, info = 0, j;
			if (I == 0){
				indx = 0;
				size = IWORK[0];
			}
			else {
				indx = IWORK[I - 1];
				size = IWORK[I] - indx;
			}
			dsteqr_("I", &size, D + indx, E + indx,
				Q + indx + indx*LDQ, &LDQ, WORK + 2 * SMLSIZ*id, &info);
			if (info != 0) INFO = (indx + 1)*(N + 1) + indx + size + 1;
			for (j = indx; j < IWORK[I]; ++j) indxq[j] = j - indx + 1;
		}
	}
	if (INFO != 0) return INFO;

	// Successively merge eigensystems of adjacent submatrices
	// into eigensystem for the corresponding larger matrix.
	int subpbs = SUBPBS;
	for (; subpbs > 1; subpbs /= 2){
		for (I = 0; I < subpbs - 1; I += 2){
			int indx, size, msd2, info = 0;
			if (I == 0){
				indx = 0;
				size = IWORK[1];
				msd2 = IWORK[0];
			}
			else {
				indx = IWORK[I - 1];
				size = IWORK[I + 1] - IWORK[I - 1];
				msd2 = size / 2;
			}
			info = dlaed1x(size, D + indx, Q + indx + indx * LDQ,
				LDQ, indxq + indx, E[indx + msd2 - 1], msd2,
				WORK, IWORK + subpbs);
			if (info != 0) return (indx + 1)*(N + 1) + indx + size + 1;
			IWORK[I / 2] = IWORK[I + 1];
		}
	}

	// Re-merge the eigenvalues/vectors which were deflated at the final
	// merge step.
#pragma omp parallel for
	for (I = 0; I < N; ++I){
		int J = indxq[I] - 1;
		WORK[I] = D[J];
		memcpy(WORK + N*I + N, Q + J*LDQ, N*sizeof(double));
	}
	memcpy(D, WORK, N*sizeof(double));
	if (N == LDQ)
		memcpy(Q, WORK + N, N*N*sizeof(double));
	else
		dlacpy_c('A', N, N, WORK + N, N, Q, LDQ);

	return 0;
}

int dlaed1x(int n, double* d, double* q, int ldq, int* indxq, double rho, int cutpnt, double* work, int* iwork)
{
	if (n < 0) return -1;
	if (ldq < MAXI(1, n)) return -4;
	if (MINI(1, n / 2) > cutpnt || (n / 2) < cutpnt) return -7;

	if (n == 0) return 0;
	int iz = 1;
	int idlmda = iz + n;
	int iw = idlmda + n;
	int iq2 = iw + n;

	int indx = 1;
	int indxc = indx + n;
	int coltyp = indxc + n;
	int indxp = coltyp + n;
	int k;
	int info = 0;
	//*     Form the z - vector which consists of the last row of Q_1 and the
	//*     first row of Q_2.
	dcopy_c(cutpnt, q + cutpnt - 1, ldq, work + iz - 1, 1);
	dcopy_c(n - cutpnt, q + cutpnt + cutpnt * ldq, ldq, work + iz + cutpnt - 1, 1);

	dlaed2_(&k, &n, &cutpnt, d, q, &ldq, indxq, &rho, work + iz - 1,
		work + idlmda - 1, work + iw - 1, work + iq2 - 1,
		iwork + indx - 1, iwork + indxc - 1, iwork + indxp - 1,
		iwork + coltyp - 1, &info);
	if (info) return info;
	//*     Solve Secular Equation.
	if (k){
		int i0 = iwork[coltyp - 1];
		int i1 = iwork[coltyp + 0];
		int i2 = iwork[coltyp + 1];
		int is = (i0 + i1) * cutpnt + (i1 + i2)*(n - cutpnt) + iq2;
#ifdef __FUJITSU
#define ED3 dlaed3x_
#else
#define ED3 dlaed3_
#endif
		ED3(&k, &n, &cutpnt, d, q, &ldq, &rho, work + idlmda -1, 
			work + iq2 - 1, iwork + indxc -1, iwork + coltyp - 1, 
			work + iw - 1, work + is - 1, &info);
		if (info) return info;
		int n1 = k;
		int n2 = n - k;
		int o = 1;
		int mo = -1;
		dlamrg_(&n1, &n2, d, &o, &mo, indxq);
	}
	else {
		int i;
		for (i = 0; i < n; ++i) indxq[i] = i + 1;
	}
	return 0;
}
