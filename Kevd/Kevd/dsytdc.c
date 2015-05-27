// Copyright by Shuhei Kudo, May 2015.
#include <omp.h>
#include "r24mv.h"
#include "common.h"
#include "simd_tools.h"

#ifdef __FUJITSU
#define MPSIZE4 1300
#define MPSIZE 72
#define MPSIZEL1 750
#else
#define MPSIZE4 200
#define MPSIZE 50
#define MPSIZEL1 100
#endif

extern void dlarfg_(const int*, double*, double*, const int*, double*);
static double dlarfg_c(int n, double* ap, double* b);
#define omp_single PRAGMA(omp single)
#define omp_for1  PRAGMA(loop unroll 4) PRAGMA(loop noalias) PRAGMA(loop simd) PRAGMA(omp for)
#define omp_for2 PRAGMA(loop unroll 4) PRAGMA(loop noalias) PRAGMA(loop simd) PRAGMA(omp for nowait)
#define omp_barrier PRAGMA(omp barrier)
#define omp_atomic PRAGMA(omp atomic)
#ifdef __FUJITSU
#define simd_for PRAGMA(loop unroll 8) PRAGMA(loop noalias) PRAGMA(loop simd aligned)
#else
#define simd_for
#endif

int dsytdd_worksize(int n, int num_threads)
{
	n = (n + 3) / 4 * 4;
	return n * (n + 1) / 2 + (num_threads + 13) * n;
}

void draxpyz(int n, double r, double a, double* x, double* y, double* z)
{
	int j;
	double ar = a * r;
	simd_for for (j = 0; j < n; ++j) z[j] = r * y[j] + ar * x[j];
}
void draxpyz2(int n, double r, double a, double* x, double* y, double* z)
{
	int j;
	double ar = a * r;
	omp_for1 for (j = 0; j < n; ++j) z[j] = r * y[j] + ar * x[j];
}
void merge(int n, int nt, double** ybuf, double* to);
void merge1(int n, int nt, double** ybuf, double* to);
#define MERGE(I, NT, YS, Y) if(i < MPSIZEL1){ if(!id) merge1((I), (NT), (YS), (Y)); omp_barrier;} else merge((I), (NT), (YS), (Y));

double red0, red1, red2, red3, red4, red5, red6;

#define MAKEU0(N)					\
	omp_single {					\
		b0 = u0[(N) - 1];			\
		r0 = dlarfg_c((N), &b0, u0);\
		u0[(N) - 1] = 1.0;			\
		red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;\
	}

static void make01(int id, int n, double* u0, double* v0, double* u1, double* y, double r0, double* b1, double *r1)
{
	if (n < MPSIZEL1){
		omp_single{
			int j;
			double t = 0.;
			simd_for for (j = 0; j < n; ++j)
				t += y[j] * u0[j];
			draxpyz(n, -r0, -0.5 * r0 * t, u0, y, v0);
			simd_for for (j = 0; j < n; ++j)
				u1[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1];
			*b1 = u1[n - 2];
			*r1 = dlarfg_c(n - 1, b1, u1);
			u1[n - 2] = 1.;
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
	else {
		int j;
		double t = 0.;
		omp_for2 for (j = 0; j < n; ++j)
			t += y[j] * u0[j];
		omp_atomic red0 += t;
		omp_barrier;
		draxpyz2(n, -r0, -0.5 * r0 * red0, u0, y, v0);
		omp_for1 for (j = 0; j < n; ++j)
			u1[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1];
		omp_single {
			*b1 = u1[n - 2];
			*r1 = dlarfg_c(n - 1, b1, u1);
			u1[n - 2] = 1.;
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
}

static void make011(int n, double* u0, double* v0, double* u1, double* y, double r0, double* b1, double *r1)
{
	int j;
	double t = 0.;
	simd_for for (j = 0; j < n; ++j)
		t += y[j] * u0[j];
	draxpyz(n, -r0, -0.5 * r0 * t, u0, y, v0);
	simd_for for (j = 0; j < n; ++j)
		u1[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1];
	*b1 = u1[n - 2];
	*r1 = dlarfg_c(n - 1, b1, u1);
	u1[n - 2] = 1.;
	red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
}

static void make12(int id, int n, double* u0, double* v0, double* u1, double* v1, double* u2, double* y, double r1, double* b2, double *r2)
{
	if (n < MPSIZEL1 - 1) {
		omp_single{
			int j;
			double t0 = 0., t1 = 0.;
			simd_for for (j = 0; j < n; ++j) {
				t0 += v0[j] * u1[j];
				t1 += u0[j] * u1[j];
			}
			double vvu3 = t0, uuu3 = t1;
			double t2 = 0.;
			simd_for for (j = 0; j < n; ++j) {
				double yy = y[j] + u0[j] * vvu3 + v0[j] * uuu3;
				t2 += yy * u1[j];
				y[j] = yy;
			}
			draxpyz(n, -r1, -0.5 * r1 * t2, u1, y, v1);

			simd_for for (j = 0; j < n; ++j)
				u2[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1]
				+ u1[j] * v1[n - 1] + v1[j] * u1[n - 1];
			*b2 = u2[n - 2];
			*r2 = dlarfg_c(n - 1, b2, u2);
			u2[n - 2] = 1.;
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
	else {
		int j;
		double t1 = 0., t2 = 0.;
		omp_for2 for (j = 0; j < n; ++j) {
			t1 += v0[j] * u1[j];
			t2 += u0[j] * u1[j];
		}
		omp_atomic red1 += t1;
		omp_atomic red2 += t2;
		omp_barrier;
		double vvu3 = red1, uuu3 = red2;
		double t0 = 0.;
		omp_for2 for (j = 0; j < n; ++j) {
			double yy = y[j] + u0[j] * vvu3 + v0[j] * uuu3;
			t0 += yy * u1[j];
			y[j] = yy;
		}
		omp_atomic red0 += t0;
		omp_barrier;
		draxpyz2(n, -r1, -0.5 * r1 * red0, u1, y, v1);

		omp_for1 for (j = 0; j < n; ++j)
			u2[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1]
			+ u1[j] * v1[n - 1] + v1[j] * u1[n - 1];
		omp_single{
			*b2 = u2[n - 2];
			*r2 = dlarfg_c(n - 1, b2, u2);
			u2[n - 2] = 1.;
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
}

static void make23(int id, int n, double* u0, double* v0, double* u1, double* v1, double* u2, double* v2, double* u3,
	double* y, double r2, double* b3, double *r3)
{
	if (n < MPSIZEL1 - 2) {
		omp_single{
			int j;
			double t0 = 0., t1 = 0., t2 = 0., t3 = 0.;
			simd_for for (j = 0; j < n; ++j) {
				t0 += v0[j] * u2[j];
				t1 += u0[j] * u2[j];
				t2 += v1[j] * u2[j];
				t3 += u1[j] * u2[j];
			}
			double vvu3 = t0, uuu3 = t1, vvu2 = t2, uuu2 = t3;
			double t4 = 0.;
			simd_for for (j = 0; j < n; ++j) {
				double yy = y[j]
					+ u0[j] * vvu3 + v0[j] * uuu3
					+ u1[j] * vvu2 + v1[j] * uuu2;
				t4 += yy * u2[j];
				y[j] = yy;
			}
			draxpyz(n, -r2, -0.5 * r2 * t4, u2, y, v2);

			simd_for for (j = 0; j < n; ++j)
				u3[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1]
				+ u1[j] * v1[n - 1] + v1[j] * u1[n - 1]
				+ u2[j] * v2[n - 1] + v2[j] * u2[n - 1];
			*b3 = u3[n - 2];
			*r3 = dlarfg_c(n - 1, b3, u3);
			u3[n - 2] = 1.;
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
	else {
		int j;
		double t1 = 0., t2 = 0., t3 = 0., t4 = 0.;
		omp_for2 for (j = 0; j < n; ++j) {
			t1 += v0[j] * u2[j];
			t2 += u0[j] * u2[j];
			t3 += v1[j] * u2[j];
			t4 += u1[j] * u2[j];
		}
		omp_atomic red1 += t1;
		omp_atomic red2 += t2;
		omp_atomic red3 += t3;
		omp_atomic red4 += t4;
		omp_barrier;
		double vvu3 = red1, uuu3 = red2, vvu2 = red3, uuu2 = red4;
		double t0 = 0.;
		omp_for2 for (j = 0; j < n; ++j) {
			double yy = y[j]
				+ u0[j] * vvu3 + v0[j] * uuu3
				+ u1[j] * vvu2 + v1[j] * uuu2;
			t0 += yy * u2[j];
			y[j] = yy;
		}
		omp_atomic red0 += t0;
		omp_barrier;
		draxpyz2(n, -r2, -0.5 * r2 * red0, u2, y, v2);

		omp_for1 for (j = 0; j < n; ++j)
			u3[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1]
			+ u1[j] * v1[n - 1] + v1[j] * u1[n - 1]
			+ u2[j] * v2[n - 1] + v2[j] * u2[n - 1];
		omp_single{
			*b3 = u3[n - 2];
			*r3 = dlarfg_c(n - 1, b3, u3);
			u3[n - 2] = 1.;
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
}

static void make34(int id, int n, double* u0, double* v0, double* u1, double* v1, double* u2, double* v2, double* u3, double* v3, double* u4,
	double* y, double r3, double* b4, double *r4)
{
	if (n < MPSIZEL1 - 3){
		omp_single{
			int j;
			double t0 = 0., t1 = 0., t2 = 0.;
			double t3 = 0., t4 = 0., t5 = 0.;
			simd_for for (j = 0; j < n; ++j) {
				t0 += v0[j] * u3[j];
				t1 += u0[j] * u3[j];
				t2 += v1[j] * u3[j];
				t3 += u1[j] * u3[j];
				t4 += v2[j] * u3[j];
				t5 += u2[j] * u3[j];
			}
			double vvu3 = t0, uuu3 = t1;
			double vvu2 = t2, uuu2 = t3;
			double vvu1 = t4, uuu1 = t5;
			double t6 = 0.;
			simd_for for (j = 0; j < n; ++j) {
				double yy = y[j]
					+ u0[j] * vvu3 + v0[j] * uuu3
					+ u1[j] * vvu2 + v1[j] * uuu2
					+ u2[j] * vvu1 + v2[j] * uuu1;
				t6 += yy * u3[j];
				y[j] = yy;
			}
			draxpyz(n, -r3, -0.5 * r3 * t6, u3, y, v3);

			simd_for for (j = 0; j < n; ++j)
				u4[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1]
				+ u1[j] * v1[n - 1] + v1[j] * u1[n - 1]
				+ u2[j] * v2[n - 1] + v2[j] * u2[n - 1]
				+ u3[j] * v3[n - 1] + v3[j] * u3[n - 1];
			*b4 = u4[n - 2];
			*r4 = dlarfg_c(n - 1, b4, u4);
			u4[n - 2] = 1.;
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
	else {
		int j;
		double t1 = 0., t2 = 0., t3 = 0., t4 = 0., t5 = 0., t6 = 0.;
		omp_for2 for (j = 0; j < n; ++j) {
			t1 += v0[j] * u3[j];
			t2 += u0[j] * u3[j];
			t3 += v1[j] * u3[j];
			t4 += u1[j] * u3[j];
			t5 += v2[j] * u3[j];
			t6 += u2[j] * u3[j];
		}
		omp_atomic red1 += t1;
		omp_atomic red2 += t2;
		omp_atomic red3 += t3;
		omp_atomic red4 += t4;
		omp_atomic red5 += t5;
		omp_atomic red6 += t6;
		omp_barrier;
		double vvu3 = red1, uuu3 = red2, vvu2 = red3, uuu2 = red4, vvu1 = red5, uuu1 = red6;
		double t0 = 0.;
		omp_for2 for (j = 0; j < n; ++j) {
			double yy = y[j]
				+ u0[j] * vvu3 + v0[j] * uuu3
				+ u1[j] * vvu2 + v1[j] * uuu2
				+ u2[j] * vvu1 + v2[j] * uuu1;
			t0 += yy * u3[j];
			y[j] = yy;
		}
		omp_atomic red0 += t0;
		omp_barrier;
		draxpyz2(n, -r3, -0.5 * r3 * red0, u3, y, v3);

		omp_for1 for (j = 0; j < n; ++j)
			u4[j] += u0[j] * v0[n - 1] + v0[j] * u0[n - 1]
			+ u1[j] * v1[n - 1] + v1[j] * u1[n - 1]
			+ u2[j] * v2[n - 1] + v2[j] * u2[n - 1]
			+ u3[j] * v3[n - 1] + v3[j] * u3[n - 1];
		omp_single{
			*b4 = u4[n - 2];
			*r4 = dlarfg_c(n - 1, b4, u4);
			u4[n - 2] = 1.;
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
}

static void makev3(int id, int n, double* u0, double* v0, double* u1, double* v1, double* u2, double* v2, double* u3, double* v3,  double* y, double r3)
{
	if (n < MPSIZEL1 - 3){
		omp_single{
			int j;
			double t0 = 0., t1 = 0., t2 = 0.;
			double t3 = 0., t4 = 0., t5 = 0.;
			simd_for for (j = 0; j < n; ++j) {
				t0 += v0[j] * u3[j];
				t1 += u0[j] * u3[j];
				t2 += v1[j] * u3[j];
				t3 += u1[j] * u3[j];
				t4 += v2[j] * u3[j];
				t5 += u2[j] * u3[j];
			}
			double vvu3 = t0, uuu3 = t1;
			double vvu2 = t2, uuu2 = t3;
			double vvu1 = t4, uuu1 = t5;
			double t6 = 0.;
			simd_for for (j = 0; j < n; ++j) {
				double yy = y[j]
					+ u0[j] * vvu3 + v0[j] * uuu3
					+ u1[j] * vvu2 + v1[j] * uuu2
					+ u2[j] * vvu1 + v2[j] * uuu1;
				t6 += yy * u3[j];
				y[j] = yy;
			}
			draxpyz(n, -r3, -0.5 * r3 * t6, u3, y, v3);
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
	else {
		int j;
		double t1 = 0., t2 = 0., t3 = 0., t4 = 0., t5 = 0., t6 = 0.;
		omp_for2 for (j = 0; j < n; ++j) {
			t1 += v0[j] * u3[j];
			t2 += u0[j] * u3[j];
			t3 += v1[j] * u3[j];
			t4 += u1[j] * u3[j];
			t5 += v2[j] * u3[j];
			t6 += u2[j] * u3[j];
		}
		omp_atomic red1 += t1;
		omp_atomic red2 += t2;
		omp_atomic red3 += t3;
		omp_atomic red4 += t4;
		omp_atomic red5 += t5;
		omp_atomic red6 += t6;
		omp_barrier;
		double vvu3 = red1, uuu3 = red2, vvu2 = red3, uuu2 = red4, vvu1 = red5, uuu1 = red6;
		double t0 = 0.;
		omp_for2 for (j = 0; j < n; ++j) {
			double yy = y[j]
				+ u0[j] * vvu3 + v0[j] * uuu3
				+ u1[j] * vvu2 + v1[j] * uuu2
				+ u2[j] * vvu1 + v2[j] * uuu1;
			t0 += yy * u3[j];
			y[j] = yy;
		}
		omp_atomic red0 += t0;
		omp_barrier;
		draxpyz2(n, -r3, -0.5 * r3 * red0, u3, y, v3);
		omp_single{
			red0 = red1 = red2 = red3 = red4 = red5 = red6 = 0.;
		}
	}
}

int dsytdc(int n, double* a, int lda, double* d, double* e, double* tau, double* work)
{
#ifdef __FUJITSU
	double* array[16]; // it seemed that alloca doesn't works on K.
#endif

	int mx = omp_get_max_threads();
	int doomp = (n - 1 >= MPSIZE && mx > 1);

	double* vt = work;
	double* vv = work + 9 * lda;
	double** ys = 0;
	double* y = work + 13 * lda;

	if (doomp) {
#ifdef __FUJITSU
		ys = array;
#else
		ys = (double**)bje_alloca(mx*sizeof(double*));
#endif
		int i;
		for (i = 0; i < mx; ++i) {
			ys[i] = work + (13 + i) * lda;
		}
	}

	double* wa = work + (13 + mx) * lda;
	int i = n - 1;
	double b0, r0;

	if (doomp) {
		double b1, r1;
		double b2, r2;
		double b3, r3;
		double b4, r4;
#pragma omp parallel private(i)
		{
			i = n - 1;
			int id = omp_get_thread_num();
			int nt = omp_get_num_threads();
			double* my = ys[id];
			double* u0 = a + i * lda;
			reorder(n, a, lda, wa);
			MAKEU0(i);
			dsymvb(i, wa, u0, my); MERGE(i, nt, ys, y);

			for (; i > MPSIZE; i -= 4){
				double* uu = a + (i - 3) * lda;
				double* u0 = a + (i - 0) * lda;
				double* u1 = a + (i - 1) * lda;
				double* u2 = a + (i - 2) * lda;
				double* u3 = a + (i - 3) * lda;
				double* u4 = a + (i - 4) * lda;
				int ldv = (i + 1) / 2 * 2;
				double* v0 = vv + 3 * ldv;
				double* v1 = vv + 2 * ldv;
				double* v2 = vv + 1 * ldv;
				double* v3 = vv + 0 * ldv;
				if (i - 1 < MPSIZEL1) {
					if (!id) {
						getcol1(i - 1, wa, u1);
						getcol1(i - 2, wa, u2);
						getcol1(i - 3, wa, u3);
						getcol1(i - 4, wa, u4);
					}
				}
				else {
					getcol(i - 1, wa, u1);
					getcol(i - 2, wa, u2);
					getcol(i - 3, wa, u3);
					getcol(i - 4, wa, u4);
				}
				omp_barrier;

				make01(id, i, u0, v0, u1, y, r0, &b1, &r1);
				dsymvb(i - 1, wa, u1, my); 
				MERGE(i - 1, nt, ys, y);

				make12(id, i - 1, u0, v0, u1, v1, u2, y, r1, &b2, &r2);
				dsymvb(i - 2, wa, u2, my); MERGE(i - 2, nt, ys, y);

				make23(id, i - 2, u0, v0, u1, v1, u2, v2, u3, y, r2, &b3, &r3);
				dsymvb(i - 3, wa, u3, my); MERGE(i - 3, nt, ys, y);

				make34(id, i - 3, u0, v0, u1, v1, u2, v2, u3, v3, u4, y, r3, &b4, &r4);
				rotate_uvb(i - 4, u4, uu, vv, lda, ldv, vt);
				dsyr24_mvb(i - 4, wa, vt, my); MERGE(i - 4, nt, ys, y);
				if(!id) {
					d[i] = u0[i];
					d[i - 1] = u1[i - 1];
					d[i - 2] = u2[i - 2];
					d[i - 3] = u3[i - 3];
					e[i - 1] = b0;
					e[i - 2] = b1;
					e[i - 3] = b2;
					e[i - 4] = b3;
					tau[i - 1] = r0;
					tau[i - 2] = r1;
					tau[i - 3] = r2;
					tau[i - 4] = r3;
					b0 = b4;
					r0 = r4;
				}
			}
		}
		for (i = n - 1; i > MPSIZE; i -= 4);
	}
	else {
		double* u0 = a + i * lda;
		reorder(n, a, lda, wa);
		MAKEU0(i);
		dsymvb(i, wa, u0, y);
	}

	// do lasting updates
	for (; i >= 2; --i) {
		double* u0 = a + (i - 0) * lda;
		double* u1 = a + (i - 1) * lda;
		double* v0 = vv;
		double b1, r1;
		getcol1(i - 1, wa, u1);
		make011(i, u0, v0, u1, y, r0, &b1, &r1);
		rotate_uvb11(i - 1, u1, u0, v0, vt);
		dsyr21_mvb(i - 1, wa, vt, y);
		d[i] = u0[i];
		e[i - 1] = b0;
		tau[i - 1] = r0;
		b0 = b1;
		r0 = r1;
	}

	d[0] = wa[0];
	d[1] = a[1 + lda];
	e[0] = b0;
	tau[0] = r0;
	return 0;
}

double dlarfg_c(int n, double* ap, double* b)
{
	int incx = 1;
	double ret;
	dlarfg_(&n, ap, b, &incx, &ret);
	return ret;
}

void merge(int n, int nt, double** ybuf, double* to)
{
	if (nt == 8) {
		int i;
		omp_for1 for (i = 0; i < n; ++i)
			to[i] += ybuf[1][i] + ybuf[2][i] + ybuf[3][i] + ybuf[4][i] + ybuf[5][i] + ybuf[6][i] + ybuf[7][i];
	}
	else {
		int i, j;
		omp_for1 for (i = 0; i < n; ++i) {
			for (j = 1; j < nt; ++j)
				to[i] += ybuf[j][i];
		}
	}
}

void merge1(int n, int nt, double** ybuf, double* to)
{
	if (nt == 8) {
		int i;
		simd_for for (i = 0; i < n; ++i)
			to[i] += ybuf[1][i] + ybuf[2][i] + ybuf[3][i] + ybuf[4][i] + ybuf[5][i] + ybuf[6][i] + ybuf[7][i];
	}
	else {
		int i, j;
		simd_for for (i = 0; i < n; ++i) {
			for (j = 1; j < nt; ++j)
				to[i] += ybuf[j][i];
		}
	}
}
