// Copyright by Shuhe Kudo, May 2015.
#include <omp.h>
#include "common.h"
#include "simd_tools.h"

#ifdef __FUJITSU
#define opt_loop PRAGMA(loop noalias) PRAGMA(loop swp) PRAGMA(loop unroll 2) PRAGMA(loop simd aligned)
#define opt_loop4 PRAGMA(loop noalias) PRAGMA(loop swp) PRAGMA(loop unroll 4) PRAGMA(loop simd aligned) PRAGMA(loop prefetch)
#define pragma_loop(A) PRAGMA__(loop A)
#define pragma_proc(A) PRAGMA__(procedure A)
#else
#define opt_loop
#define opt_loop4
#define pragma_loop(A)
#define pragma_proc(A)
#endif


void rotate_uvb(int n, double* x, double* uf, double* vf, int lda, int ldv, double* to)
{
	int i;
#pragma omp for
	opt_loop for (i = 0; i < n; i += 2){
		st(to + 9 * i + 0, ld(x + i));

		st(to + 9 * i + 2, ld(uf + 0 * lda + i));
		st(to + 9 * i + 4, ld(uf + 1 * lda + i));
		st(to + 9 * i + 6, ld(uf + 2 * lda + i));
		st(to + 9 * i + 8, ld(uf + 3 * lda + i));

		st(to + 9 * i + 10, ld(vf + 0 * ldv + i));
		st(to + 9 * i + 12, ld(vf + 1 * ldv + i));
		st(to + 9 * i + 14, ld(vf + 2 * ldv + i));
		st(to + 9 * i + 16, ld(vf + 3 * ldv + i));
	}
}


void rotate_uvb1(int n, double* x, double* uf, double* vf, double* to)
{
	int i;
#pragma omp for
	opt_loop4 for (i = 0; i < n; i += 2){
		st(to + 3 * i + 0, ld(x + i));
		st(to + 3 * i + 2, ld(uf + i));
		st(to + 3 * i + 4, ld(vf + i));
	}
}

void rotate_uvb11(int n, double* x, double* uf, double* vf, double* to)
{
	int i;
	opt_loop4
	for (i = 0; i < n; i += 2){
		st(to + 3 * i + 0, ld(x + i));
		st(to + 3 * i + 2, ld(uf + i));
		st(to + 3 * i + 4, ld(vf + i));
	}
}

#define IDef0(A, B)
#define IDef1(A, B) A B##0
#define IDef2(A, B) IDef1(A, B), B##1
#define IDef3(A, B) IDef2(A, B), B##2
#define IDef4(A, B) IDef3(A, B), B##3
#define IDef5(A, B) IDef4(A, B), B##4
#define IDef6(A, B) IDef5(A, B), B##5
#define IDef7(A, B) IDef6(A, B), B##6
#define IDef8(A, B) IDef7(A, B), B##7
#define IDef9(A, B) IDef8(A, B), B##8
#define IDef10(A, B) IDef9(A, B), B##9
#define IDef(A, B, N) IDef##N(A, B)

#define IDefmv0(T, A, B)
#define IDefmv1(T, A, B) IDef(T, A##0, B)
#define IDefmv2(T, A, B) IDefmv1(T, A, B); IDef(T, A##1, B)
#define IDefmv3(T, A, B) IDefmv2(T, A, B); IDef(T, A##2, B)
#define IDefmv4(T, A, B) IDefmv3(T, A, B); IDef(T, A##3, B)
#define IDefmv5(T, A, B) IDefmv4(T, A, B); IDef(T, A##4, B)
#define IDefmv6(T, A, B) IDefmv5(T, A, B); IDef(T, A##5, B)
#define IDefmv(T, A, B, C) IDefmv##C(T, A, B)

#define Def(A, N) IDef##N(d2v, A)
#define SDef(A, N) IDef##N(double, A)
#define Defmv(A, M, N) IDefmv(d2v, A, M, N)
#define SDefmv(A, M, N) IDefmv(double, A, M, N)

#define Zero0(A)
#define Zero1(A) A##0 = _mm_setzero_pd()
#define Zero2(A) Zero1(A); A##1 = _mm_setzero_pd()
#define Zero3(A) Zero2(A); A##2 = _mm_setzero_pd()
#define Zero4(A) Zero3(A); A##3 = _mm_setzero_pd()
#define Zero5(A) Zero4(A); A##4 = _mm_setzero_pd()
#define Zero6(A) Zero5(A); A##5 = _mm_setzero_pd()
#define Zero7(A) Zero6(A); A##6 = _mm_setzero_pd()
#define Zero8(A) Zero7(A); A##7 = _mm_setzero_pd()
#define Zero9(A) Zero8(A); A##8 = _mm_setzero_pd()
#define Zero10(A) Zero9(A); A##9 = _mm_setzero_pd()
#define Zero(A, N) Zero##N(A)

#define Set0(A, B)
#define Set1(A, B) A##0 = set1(B##0)
#define Set2(A, B) Set1(A, B); A##1 = set1(B##1)
#define Set3(A, B) Set2(A, B); A##2 = set1(B##2)
#define Set4(A, B) Set3(A, B); A##3 = set1(B##3)
#define Set5(A, B) Set4(A, B); A##4 = set1(B##4)
#define Set6(A, B) Set5(A, B); A##5 = set1(B##5)
#define Set7(A, B) Set6(A, B); A##6 = set1(B##6)
#define Set8(A, B) Set7(A, B); A##7 = set1(B##7)
#define Set9(A, B) Set8(A, B); A##8 = set1(B##8)
#define Set10(A, B) Set9(A, B); A##9 = set1(B##9)
#define Set(A, B, N) Set##N(A, B)

#define Setmv0(A, B, N)
#define Setmv1(A, B, N) Set(A##0, B##0, N)
#define Setmv2(A, B, N) Setmv1(A, B, N); Set(A##1, B##1, N)
#define Setmv3(A, B, N) Setmv2(A, B, N); Set(A##2, B##2, N)
#define Setmv4(A, B, N) Setmv3(A, B, N); Set(A##3, B##3, N)
#define Setmv5(A, B, N) Setmv4(A, B, N); Set(A##4, B##4, N)
#define Setmv6(A, B, N) Setmv5(A, B, N); Set(A##5, B##5, N)
#define Setmv(A, B, M, N) Setmv##N(A, B, M)

#define Ld0(A, W, K, N)
#define Ld1(A, W, K, N) A##0 = ld((W) + (N) * (K) + 0)
#define Ld2(A, W, K, N) Ld1(A, W, K, N); A##1 = ld((W) + (N) * (K) + 2)
#define Ld3(A, W, K, N) Ld2(A, W, K, N); A##2 = ld((W) + (N) * (K) + 4)
#define Ld4(A, W, K, N) Ld3(A, W, K, N); A##3 = ld((W) + (N) * (K) + 6)
#define Ld5(A, W, K, N) Ld4(A, W, K, N); A##4 = ld((W) + (N) * (K) + 8)
#define Ld6(A, W, K, N) Ld5(A, W, K, N); A##5 = ld((W) + (N) * (K) + 10)
#define Ld7(A, W, K, N) Ld6(A, W, K, N); A##6 = ld((W) + (N) * (K) + 12)
#define Ld8(A, W, K, N) Ld7(A, W, K, N); A##7 = ld((W) + (N) * (K) + 14)
#define Ld9(A, W, K, N) Ld8(A, W, K, N); A##8 = ld((W) + (N) * (K) + 16)
#define Ld10(A, W, K, N) Ld9(A, W, K, N); A##9 = ld((W) + (N) * (K) + 18)
#define Ld(A, W, K, M, N) Ld##M(A, W, K, N)

#define SLd0(A, W)
#define SLd1(A, W) A##0 = (W)[0]
#define SLd2(A, W) SLd1(A, W); A##1 = (W)[1]
#define SLd3(A, W) SLd2(A, W); A##2 = (W)[2]
#define SLd4(A, W) SLd3(A, W); A##3 = (W)[3]
#define SLd5(A, W) SLd4(A, W); A##4 = (W)[4]
#define SLd6(A, W) SLd5(A, W); A##5 = (W)[5]
#define SLd7(A, W) SLd6(A, W); A##6 = (W)[6]
#define SLd8(A, W) SLd7(A, W); A##7 = (W)[7]
#define SLd9(A, W) SLd8(A, W); A##8 = (W)[8]
#define SLd10(A, W) SLd9(A, W); A##9 = (W)[9]
#define SLd(A, W, N) SLd##N(A, W)

#define St0(A, W, K, N)
#define St1(A, W, K, N) st((W) + (N) * (K) + 0, A##0)
#define St2(A, W, K, N) St1(A, W, K, N); st((W) + (N) * (K) + 2, A##1)
#define St3(A, W, K, N) St2(A, W, K, N); st((W) + (N) * (K) + 4, A##2)
#define St4(A, W, K, N) St3(A, W, K, N); st((W) + (N) * (K) + 6, A##3)
#define St5(A, W, K, N) St4(A, W, K, N); st((W) + (N) * (K) + 8, A##4)
#define St6(A, W, K, N) St5(A, W, K, N); st((W) + (N) * (K) + 10, A##5)
#define St7(A, W, K, N) St6(A, W, K, N); st((W) + (N) * (K) + 12, A##6)
#define St8(A, W, K, N) St7(A, W, K, N); st((W) + (N) * (K) + 14, A##7)
#define St9(A, W, K, N) St8(A, W, K, N); st((W) + (N) * (K) + 16, A##8)
#define St10(A, W, K, N) St9(A, W, K, N); st((W) + (N) * (K) + 18, A##9)
#define St(A, W, K, M, N) St##M(A, W, K, N)

#define Madd0(A, B, C)
#define Madd1(A, B, C) A##0 = madd(B, C##0, A##0)
#define Madd2(A, B, C) Madd1(A, B, C); A##1 = madd(B, C##1, A##1)
#define Madd3(A, B, C) Madd2(A, B, C); A##2 = madd(B, C##2, A##2)
#define Madd4(A, B, C) Madd3(A, B, C); A##3 = madd(B, C##3, A##3)
#define Madd5(A, B, C) Madd4(A, B, C); A##4 = madd(B, C##4, A##4)
#define Madd6(A, B, C) Madd5(A, B, C); A##5 = madd(B, C##5, A##5)
#define Madd7(A, B, C) Madd6(A, B, C); A##6 = madd(B, C##6, A##6)
#define Madd8(A, B, C) Madd7(A, B, C); A##7 = madd(B, C##7, A##7)
#define Madd9(A, B, C) Madd8(A, B, C); A##8 = madd(B, C##8, A##8)
#define Madd10(A, B, C) Madd9(A, B, C); A##9 = madd(B, C##9, A##9)
#define Madd(A, B, C, N) Madd##N(A, B, C)

#define Mred0(A, B, C)
#define Mred1(A, B, C) A = madd(B##0, C##0, A)
#define Mred2(A, B, C) Mred1(A, B, C); A = madd(B##1, C##1, A)
#define Mred3(A, B, C) Mred2(A, B, C); A = madd(B##2, C##2, A)
#define Mred4(A, B, C) Mred3(A, B, C); A = madd(B##3, C##3, A)
#define Mred5(A, B, C) Mred4(A, B, C); A = madd(B##4, C##4, A)
#define Mred6(A, B, C) Mred5(A, B, C); A = madd(B##5, C##5, A)
#define Mred7(A, B, C) Mred6(A, B, C); A = madd(B##6, C##6, A)
#define Mred8(A, B, C) Mred7(A, B, C); A = madd(B##7, C##7, A)
#define Mred9(A, B, C) Mred8(A, B, C); A = madd(B##8, C##8, A)
#define Mred10(A, B, C) Mred9(A, B, C); A = madd(B##9, C##9, A)
#define Mred(A, B, C, N) Mred##N(A, B, C)

#define Mredr1_0(A, B, C, K)
#define Mredr1_1(A, B, C, K) Mred(A##0, B, C##0, K)
#define Mredr1_2(A, B, C, K) Mredr1_1(A, B, C, K); Mred(A##1, B, C##1, K)
#define Mredr1_3(A, B, C, K) Mredr1_2(A, B, C, K); Mred(A##2, B, C##2, K)
#define Mredr1_4(A, B, C, K) Mredr1_3(A, B, C, K); Mred(A##3, B, C##3, K)
#define Mredr1_5(A, B, C, K) Mredr1_4(A, B, C, K); Mred(A##4, B, C##4, K)
#define Mredr1_6(A, B, C, K) Mredr1_5(A, B, C, K); Mred(A##5, B, C##5, K)
#define Mredr1(A, B, C, K, M) Mredr1_##M(A, B, C, K)


void reorder(int n, const double* a, int lda, double* b)
{
	pragma_proc(noalias);
	pragma_proc(nofltld);

	int i;
#pragma omp for schedule(static, 2)
	for (i = 0; i < n; i += 4) {
		int j;
		double* p = b + i * (i + 1) / 2;
		if (i + 4 <= n) {
pragma_loop(xfill)
pragma_loop(noalias)
pragma_loop(unroll)
pragma_loop(simd aligned)
			for (j = 0; j + 2 <= i; j += 2) {
				st(p + 4 * j + 0, ld(a + (i + 0) * lda + j));
				st(p + 4 * j + 2, ld(a + (i + 1) * lda + j));
				st(p + 4 * j + 4, ld(a + (i + 2) * lda + j));
				st(p + 4 * j + 6, ld(a + (i + 3) * lda + j));
			}
			p[4 * j + 0] = a[lda*(i + 0) + j + 0];
			p[4 * j + 1] = a[lda*(i + 1) + j + 0];
			p[4 * j + 2] = a[lda*(i + 2) + j + 0];
			p[4 * j + 3] = a[lda*(i + 3) + j + 0];
			p[4 * j + 4] = a[lda*(i + 1) + j + 1];
			p[4 * j + 5] = a[lda*(i + 2) + j + 1];
			p[4 * j + 6] = a[lda*(i + 3) + j + 1];
			p[4 * j + 7] = a[lda*(i + 2) + j + 2];
			p[4 * j + 8] = a[lda*(i + 3) + j + 2];
			p[4 * j + 9] = a[lda*(i + 3) + j + 3];
		}
		else {
pragma_loop(xfill)
pragma_loop(noalias)
pragma_loop(simd aligned)
			for (j = 0; j + 2 <= i; j += 2) {
				p[4 * j + 0] = a[lda*i + j];
				p[4 * j + 1] = a[lda*i + j + 1];
				p[4 * j + 2] = (n - i > 1 ? a[lda*(i + 1) + j] : 0.0);
				p[4 * j + 3] = (n - i > 1 ? a[lda*(i + 1) + j + 1] : 0.0);
				p[4 * j + 4] = (n - i > 2 ? a[lda*(i + 2) + j] : 0.0);
				p[4 * j + 5] = (n - i > 2 ? a[lda*(i + 2) + j + 1] : 0.0);
				p[4 * j + 6] = 0.0;
				p[4 * j + 7] = 0.0;
			}
			p[4 * j + 0] = a[lda*i + j + 0];
			p[4 * j + 1] = (n - i > 1 ? a[lda*(i + 1) + j + 0] : 0.0);
			p[4 * j + 2] = (n - i > 2 ? a[lda*(i + 2) + j + 0] : 0.0);
			p[4 * j + 3] = 0.0;
			p[4 * j + 4] = (n - i > 1 ? a[lda*(i + 1) + j + 1] : 0.0);
			p[4 * j + 5] = (n - i > 2 ? a[lda*(i + 2) + j + 1] : 0.0);
			p[4 * j + 6] = 0.0;
			p[4 * j + 7] = (n - i > 2 ? a[lda*(i + 2) + j + 2] : 0.0);
			p[4 * j + 8] = 0.0;
			p[4 * j + 9] = 0.0;
		}
	}
}

void getcol(int i, const double* a, double* b)
{
	int bi = i / 4 * 4;
	a += bi*(bi + 1) / 2;
	int jend = bi - 2;
	int j;
#pragma omp for nowait
	for (j = 0; j <= jend; j += 2) {
		st(b + j, ld((a + 2 * (i % 4)) + 4 * j));
	}
#pragma omp single nowait
	{
		a += 4 * bi;
		switch (i - bi) {
		case 0:
			b[bi + 0] = a[0];
			break;
		case 1:
			b[bi + 0] = a[1];
			b[bi + 1] = a[4];
			break;
		case 2:
			b[bi + 0] = a[2];
			b[bi + 1] = a[5];
			b[bi + 2] = a[7];
			break;
		case 3:
			b[bi + 0] = a[3];
			b[bi + 1] = a[6];
			b[bi + 2] = a[8];
			b[bi + 3] = a[9];
			break;
		}
	}
}

void getcol1(int i, const double* a, double* b)
{
	int bi = i / 4 * 4;
	a += bi*(bi + 1) / 2;
	int jend = bi - 2;
	int j;
	for (j = 0; j <= jend; j += 2) {
		st(b + j, ld((a + 2 * (i % 4)) + 4 * j));
	}
	{
		a += 4 * bi;
		switch (i - bi) {
		case 0:
			b[bi + 0] = a[0];
			break;
		case 1:
			b[bi + 0] = a[1];
			b[bi + 1] = a[4];
			break;
		case 2:
			b[bi + 0] = a[2];
			b[bi + 1] = a[5];
			b[bi + 2] = a[7];
			break;
		case 3:
			b[bi + 0] = a[3];
			b[bi + 1] = a[6];
			b[bi + 2] = a[8];
			b[bi + 3] = a[9];
			break;
		}
	}
}

#define mvdef(M, N)			\
	SDef(x, M);				\
	SLd(x, (x + i), M);		\
	Def(xx, M);				\
	Def(xy, M);				\
	Set(xx, x, M);			\
	Zero(xy, M);			\
	d2v xxj, xyj;			\
	Def(xa, M);				\
	xxj = ld(x + j);		\
	xyj = ld(y + j);		\
	Ld(xa, ca, 0, M, N);
#define mvkernel(M, N)		\
	d2v yxj, yyj;			\
	Def(ya, M);				\
	Ld(ya, ca, j + 2, M, N);\
	yyj = ld(y + j + 2);	\
	yxj = ld(x + j + 2);	\
	Mred(xyj, xa, xx, M);	\
	Madd(xy, xxj, xa, M);	\
	st(y + j, xyj);			\
	Ld(xa, ca, j + 4, M, N);\
	xyj = ld(y + j + 4);	\
	xxj = ld(x + j + 4);	\
	Mred(yyj, ya, xx, M);	\
	Madd(xy, yxj, ya, M);	\
	st(y + j + 2, yyj);

void dsymvb(int n, const double* a, const double* x, double* y)
{
	pragma_proc(nofltld)
	int i;
	for (i = 0; i < n; ++i) y[i] = 0.;
#pragma omp for schedule(static, 2)
	for (i = 0; i < n; i += 4) {
		switch (n - i){
		default:
		{
			int j = 0;
			const double* ca = a + (i * (i + 1) / 2);;
			mvdef(4, 4);
			opt_loop for (; j + 4 <= i; j += 4){
				mvkernel(4, 4);
			}
			ca += 4 * j;
			double
				a11 = ca[0], a12 = ca[1], a13 = ca[2], a14 = ca[3],
				a22 = ca[4], a23 = ca[5], a24 = ca[6],
				a33 = ca[7], a34 = ca[8],
				a44 = ca[9];

			y[j + 0] = hadd(xy0) + a11*x0 + a12*x1 + a13*x2 + a14*x3;
			y[j + 1] = hadd(xy1) + a12*x0 + a22*x1 + a23*x2 + a24*x3;
			y[j + 2] = hadd(xy2) + a13*x0 + a23*x1 + a33*x2 + a34*x3;
			y[j + 3] = hadd(xy3) + a14*x0 + a24*x1 + a34*x2 + a44*x3;
		}
		break;
		case 3:
		{
			int j = 0;
			const double* ca = a + (i * (i + 1) / 2);
			mvdef(3, 4);
			opt_loop for (; j + 4 <= i; j += 4) {
				mvkernel(3, 4);
			}
			ca += 4 * j;
			double
				a11 = ca[0], a12 = ca[1], a13 = ca[2],
				a22 = ca[4], a23 = ca[5],
				a33 = ca[7];
			y[j + 0] = hadd(xy0) + a11*x0 + a12*x1 + a13*x2;
			y[j + 1] = hadd(xy1) + a12*x0 + a22*x1 + a23*x2;
			y[j + 2] = hadd(xy2) + a13*x0 + a23*x1 + a33*x2;
			break;
		}
		case 2:
		{
			int j = 0;
			const double* ca = a + (i*(i + 1) / 2);
			mvdef(2, 4);
			opt_loop for (; j + 4 <= i; j += 4) {
				mvkernel(2, 4);
			}
			ca += 4 * j;
			double a12 = ca[1];
			y[j + 0] = hadd(xy0) + ca[0] * x0 + a12 * x1;
			y[j + 1] = hadd(xy1) + a12 * x0 + ca[4] * x1;
			break;
		}
		case 1:
		{
			int j = 0;
			const double* ca = a + (i*(i + 1) / 2);
			mvdef(1, 4);
			opt_loop for (; j + 4 <= i; j += 4) {
				mvkernel(1, 4);
			}
			ca += 4 * j;
			y[j] = hadd(xy0) + ca[0] * x0;
			break;
		}
		}
	}
}

#define r2mvdef(M, N)		\
	SDef(u, M);		\
	SDef(v, M);		\
	SDef(x, M);		\
	SLd(u, (u + i), M);	\
	SLd(v, (v + i), M);	\
	SLd(x, (x + i), M);	\
	Def(xu, M);		\
	Def(xv, M);		\
	Def(xx, M);		\
	Def(xy, M);		\
	Set(xu, u, M);		\
	Set(xv, v, M);		\
	Set(xx, x, M);		\
	Zero(xy, M);		\
	d2v xvj, xuj, xxj, xyj;	\
	d2v yvj, yuj, yxj, yyj;	\
	Def(xa, M);		\
	Def(ya, M);		\
	xvj = ld(v + j);	\
	xuj = ld(u + j);	\
	xxj = ld(x + j);	\
	xyj = ld(y + j);	\
	Ld(xa, ca, 0, M, N);
#define r2mvkernel(M, N)	\
	Ld(ya, ca, j + 2, M, N);\
	yuj = ld(u + j + 2);	\
	yvj = ld(v + j + 2);	\
	yyj = ld(y + j + 2);	\
	yxj = ld(x + j + 2);	\
	Madd(xa, xvj, xu, M);	\
	Madd(xa, xuj, xv, M);	\
	Mred(xyj, xa, xx, M);	\
	Madd(xy, xxj, xa, M);	\
	St(xa, ca, j, M, N);	\
	st(y + j, xyj);			\
	Ld(xa, ca, j + 4, M, N);\
	xuj = ld(u + j + 4);	\
	xvj = ld(v + j + 4);	\
	xyj = ld(y + j + 4);	\
	xxj = ld(x + j + 4);	\
	Madd(ya, yvj, xu, M);	\
	Madd(ya, yuj, xv, M);	\
	Mred(yyj, ya, xx, M);	\
	Madd(xy, yxj, ya, M);	\
	St(ya, ca, j + 2, M, N);\
	st(y + j + 2, yyj);


#define strd0(K, i) ((2 * K + 1) * i)
#define lduv0(I, K, D, U)
#define lduv1(I, K, D, U) U##I##0 = xuv[strd0(K, i) + (D) + 0]
#define lduv2(I, K, D, U) lduv1(I, K, D, U); U##I##1 = xuv[strd0(K, i) + (D) + 2]
#define lduv3(I, K, D, U) lduv2(I, K, D, U); U##I##2 = xuv[strd0(K, i) + (D) + 4]
#define lduv4(I, K, D, U) lduv3(I, K, D, U); U##I##3 = xuv[strd0(K, i) + (D) + 6]
#define lduv5(I, K, D, U) lduv4(I, K, D, U); U##I##4 = xuv[strd0(K, i) + (D) + 8]
#define lduv6(I, K, D, U) lduv5(I, K, D, U); U##I##5 = xuv[strd0(K, i) + (D) + 10]
#define lduv(I, K, D, U) lduv##K(I, K, D, U)

#define strd1(I, K) (I * (4 * K + 2))
#define ldxuv1(K)					\
	x0  = xuv[strd0(K, i) + strd1(0, K) + 0];	\
	lduv(0, K, strd1(0, K) + 2, u);			\
	lduv(0, K, strd1(0, K) + (2 * K + 2), v);
#define ldxuv2(K) ldxuv1(K);				\
	x1  = xuv[strd0(K, i) + strd1(0, K) + 1];	\
	lduv(1, K, strd1(0, K) + 3, u);			\
	lduv(1, K, strd1(0, K) + (2 * K + 3), v);
#define ldxuv3(K) ldxuv2(K);				\
	x2  = xuv[strd0(K, i) + strd1(1, K) + 0];	\
	lduv(2, K, strd1(1, K) + 2, u);			\
	lduv(2, K, strd1(1, K) + (2 * K + 2), v);
#define ldxuv4(K) ldxuv3(K);				\
	x3  = xuv[strd0(K, i) + strd1(1, K) + 1];	\
	lduv(3, K, strd1(1, K) + 3, u);			\
	lduv(3, K, strd1(1, K) + (2 * K + 3), v);
#define ldxuv5(K) ldxuv4(K);				\
	x4  = xuv[strd0(K, i) + strd1(2, K) + 0];	\
	lduv(4, K, strd1(2, K) + 2, u);			\
	lduv(4, K, strd1(2, K) + (2 * K + 2), v);
#define ldxuv6(K) ldxuv5(K);				\
	x5  = xuv[strd0(K, i) + strd1(2, K) + 1];	\
	lduv(5, K, strd1(2, K) + 3, u);			\
	lduv(5, K, strd1(2, K) + (2 * K + 3), v);
#define ldxuv(K, M) ldxuv##M(K)

#define Ldxuv(X, J, K)					\
	X##xj = ld(xuv + (2 * K + 1) * (J));		\
	Ld(X##uj, (xuv + 2), J, K, (2 * K + 1));	\
	Ld(X##vj, (xuv + 2 * K + 2), J, K, (2 * K + 1));

#define r2kmvdef(K, M, N)		\
	SDefmv(u, K, M);		\
	SDefmv(v, K, M);		\
	SDef(x, M);			\
	ldxuv(K, M);			\
	Defmv(xu, K, M);		\
	Defmv(xv, K, M);		\
	Def(xx, M);			\
	Def(xy, M);			\
	Setmv(xu, u, K, M);		\
	Setmv(xv, v, K, M);		\
	Set(xx, x, M);			\
	Zero(xy, M);			\
	Def(xuj, K);			\
	Def(xvj, K);			\
	Def(yuj, K);			\
	Def(yvj, K);			\
	d2v xxj, xyj;			\
	d2v yxj, yyj;			\
	Def(xa, M);			\
	Def(ya, M);			\
	xyj = ld(y + j);		\
	Ldxuv(x, j, K);			\
	Ld(xa, ca, 0, M, N);
#define r2kmvkernel(K, M, N)		\
	Ld(ya, ca, j + 2, M, N);	\
	Ldxuv(y, j + 2, K);		\
	yyj = ld(y + j + 2);		\
	Mredr1(xa, xuj, xv, K, M);	\
	Mredr1(xa, xvj, xu, K, M);	\
	Mred(xyj, xa, xx, M);		\
	Madd(xy, xxj, xa, M);		\
	St(xa, ca, j, M, N);		\
	st(y + j, xyj);			\
	Ld(xa, ca, j + 4, M, N);	\
	Ldxuv(x, j + 4, K);		\
	xyj = ld(y + j + 4);		\
	Mredr1(ya, yvj, xu, K, M);	\
	Mredr1(ya, yuj, xv, K, M);	\
	Mred(yyj, ya, xx, M);		\
	Madd(xy, yxj, ya, M);		\
	St(ya, ca, j + 2, M, N);	\
	st(y + j + 2, yyj);

#define r2k0(I, J)
#define r2k1(I, J) u##I##0 * v##J##0 + v##I##0 * u##J##0
#define r2k2(I, J) r2k1(I, J) + u##I##1 * v##J##1 + v##I##1 * u##J##1
#define r2k3(I, J) r2k2(I, J) + u##I##2 * v##J##2 + v##I##2 * u##J##2
#define r2k4(I, J) r2k3(I, J) + u##I##3 * v##J##3 + v##I##3 * u##J##3
#define r2k5(I, J) r2k4(I, J) + u##I##4 * v##J##4 + v##I##4 * u##J##4
#define r2k6(I, J) r2k5(I, J) + u##I##5 * v##J##5 + v##I##5 * u##J##5
#define r2kk(I, J, K) r2k##K(I, J)
#define r2k(I, J, K)  r2kk(I, J, K)

#define K 4
void dsyr24_mvb(int n, double* a, const double* xuv, double* y)
{
	pragma_proc(nofltld);
	pragma_proc(noalias);

	int i;
	for (i = 0; i < n; ++i) y[i] = 0.;
#pragma omp for schedule(static, 2)
	for (i = 0; i < n; i += 4) {
		switch (n - i) {
		default:
		{
			int j = 0;
			double* ca = a + (i * (i + 1) / 2);
			r2kmvdef(K, 4, 4);
			opt_loop for (; j + 4 <= i; j += 4){
				r2kmvkernel(K, 4, 4);
			}
			ca += 4 * j;
			double
				a11 = ca[0], a12 = ca[1], a13 = ca[2], a14 = ca[3],
				a22 = ca[4], a23 = ca[5], a24 = ca[6],
				a33 = ca[7], a34 = ca[8],
				a44 = ca[9];

			a11 += r2k(0, 0, K); a12 += r2k(1, 0, K); a13 += r2k(2, 0, K); a14 += r2k(3, 0, K);
			a22 += r2k(1, 1, K); a23 += r2k(2, 1, K); a24 += r2k(3, 1, K);
			a33 += r2k(2, 2, K); a34 += r2k(3, 2, K);
			a44 += r2k(3, 3, K);
			ca[0] = a11, ca[1] = a12, ca[2] = a13, ca[3] = a14;
			ca[4] = a22, ca[5] = a23, ca[6] = a24;
			ca[7] = a33, ca[8] = a34;
			ca[9] = a44;
			y[j + 0] = hadd(xy0) + a11*x0 + a12*x1 + a13*x2 + a14*x3;
			y[j + 1] = hadd(xy1) + a12*x0 + a22*x1 + a23*x2 + a24*x3;
			y[j + 2] = hadd(xy2) + a13*x0 + a23*x1 + a33*x2 + a34*x3;
			y[j + 3] = hadd(xy3) + a14*x0 + a24*x1 + a34*x2 + a44*x3;
		}
		break;
		case 3:
		{
			int j = 0;
			double* ca = a + (i*(i + 1) / 2);
			r2kmvdef(K, 3, 4);
			opt_loop for (; j + 4 <= i; j += 4){
				r2kmvkernel(K, 3, 4);
			}
			ca += 4 * j;
			double
				a11 = ca[0], a12 = ca[1], a13 = ca[2],
				a22 = ca[4], a23 = ca[5],
				a33 = ca[7];

			a11 += r2k(0, 0, K); a12 += r2k(1, 0, K); a13 += r2k(2, 0, K);
			a22 += r2k(1, 1, K); a23 += r2k(2, 1, K);
			a33 += r2k(2, 2, K);

			ca[0] = a11, ca[1] = a12, ca[2] = a13;
			ca[4] = a22, ca[5] = a23;
			ca[7] = a33;

			y[j + 0] = hadd(xy0) + a11*x0 + a12*x1 + a13*x2;
			y[j + 1] = hadd(xy1) + a12*x0 + a22*x1 + a23*x2;
			y[j + 2] = hadd(xy2) + a13*x0 + a23*x1 + a33*x2;
			break;
		}
		case 2:
		{
			int j = 0;
			double* ca = a + (i*(i + 1) / 2);
			r2kmvdef(K, 2, 4);
			opt_loop for (; j + 4 <= i; j += 4) {
				r2kmvkernel(K, 2, 4);
			}
			ca += 4 * j;
			double
				a11 = ca[0], a12 = ca[1],
				a22 = ca[4];

			a11 += r2k(0, 0, K); a12 += r2k(1, 0, K);
			a22 += r2k(1, 1, K);

			ca[0] = a11, ca[1] = a12;
			ca[4] = a22;

			y[j + 0] = hadd(xy0) + a11*x0 + a12*x1;
			y[j + 1] = hadd(xy1) + a12*x0 + a22*x1;
			break;
		}
		case 1:
		{
			int j = 0;
			double* ca = a + (i * (i + 1) / 2);
			r2kmvdef(K, 1, 4);
			opt_loop for (; j + 4 <= i; j += 4) {
				r2kmvkernel(K, 1, 4);
			}
			ca += 4 * j;
			double a11 = ca[0];
			a11 += r2k(0, 0, K);
			ca[0] = a11;
			y[j + 0] = hadd(xy0) + a11*x0;
			break;
		}
		}
	}
}

#undef K
#define K 1
void dsyr21_mvb(int n, double* a, const double* xuv, double* y)
{
	pragma_proc(nofltld);
	pragma_proc(noalias);

	int i;
	for (i = 0; i < n; ++i) y[i] = 0.;
#pragma omp for schedule(static, 2)
	for (i = 0; i < n; i += 4) {
		switch (n - i) {
		default:
		{
			int j = 0;
			double* ca = a + (i * (i + 1) / 2);
			r2kmvdef(K, 4, 4);
			opt_loop for (; j + 4 <= i; j += 4){
				r2kmvkernel(K, 4, 4);
			}
			ca += 4 * j;
			double
				a11 = ca[0], a12 = ca[1], a13 = ca[2], a14 = ca[3],
				a22 = ca[4], a23 = ca[5], a24 = ca[6],
				a33 = ca[7], a34 = ca[8],
				a44 = ca[9];

			a11 += r2k(0, 0, K); a12 += r2k(1, 0, K); a13 += r2k(2, 0, K); a14 += r2k(3, 0, K);
			a22 += r2k(1, 1, K); a23 += r2k(2, 1, K); a24 += r2k(3, 1, K);
			a33 += r2k(2, 2, K); a34 += r2k(3, 2, K);
			a44 += r2k(3, 3, K);
			ca[0] = a11, ca[1] = a12, ca[2] = a13, ca[3] = a14;
			ca[4] = a22, ca[5] = a23, ca[6] = a24;
			ca[7] = a33, ca[8] = a34;
			ca[9] = a44;
			y[j + 0] = hadd(xy0) + a11*x0 + a12*x1 + a13*x2 + a14*x3;
			y[j + 1] = hadd(xy1) + a12*x0 + a22*x1 + a23*x2 + a24*x3;
			y[j + 2] = hadd(xy2) + a13*x0 + a23*x1 + a33*x2 + a34*x3;
			y[j + 3] = hadd(xy3) + a14*x0 + a24*x1 + a34*x2 + a44*x3;
		}
		break;
		case 3:
		{
			int j = 0;
			double* ca = a + (i*(i + 1) / 2);
			r2kmvdef(K, 3, 4);
			opt_loop for (; j + 4 <= i; j += 4){
				r2kmvkernel(K, 3, 4);
			}
			ca += 4 * j;
			double
				a11 = ca[0], a12 = ca[1], a13 = ca[2],
				a22 = ca[4], a23 = ca[5],
				a33 = ca[7];

			a11 += r2k(0, 0, K); a12 += r2k(1, 0, K); a13 += r2k(2, 0, K);
			a22 += r2k(1, 1, K); a23 += r2k(2, 1, K);
			a33 += r2k(2, 2, K);

			ca[0] = a11, ca[1] = a12, ca[2] = a13;
			ca[4] = a22, ca[5] = a23;
			ca[7] = a33;

			y[j + 0] = hadd(xy0) + a11*x0 + a12*x1 + a13*x2;
			y[j + 1] = hadd(xy1) + a12*x0 + a22*x1 + a23*x2;
			y[j + 2] = hadd(xy2) + a13*x0 + a23*x1 + a33*x2;
			break;
		}
		case 2:
		{
			int j = 0;
			double* ca = a + (i*(i + 1) / 2);
			r2kmvdef(K, 2, 4);
			opt_loop for (; j + 4 <= i; j += 4) {
				r2kmvkernel(K, 2, 4);
			}
			ca += 4 * j;
			double
				a11 = ca[0], a12 = ca[1],
				a22 = ca[4];

			a11 += r2k(0, 0, K); a12 += r2k(1, 0, K);
			a22 += r2k(1, 1, K);

			ca[0] = a11, ca[1] = a12;
			ca[4] = a22;

			y[j + 0] = hadd(xy0) + a11*x0 + a12*x1;
			y[j + 1] = hadd(xy1) + a12*x0 + a22*x1;
			break;
		}
		case 1:
		{
			int j = 0;
			double* ca = a + (i * (i + 1) / 2);
			r2kmvdef(K, 1, 4);
			opt_loop for (; j + 4 <= i; j += 4) {
				r2kmvkernel(K, 1, 4);
			}
			ca += 4 * j;
			double a11 = ca[0];
			a11 += r2k(0, 0, K);
			ca[0] = a11;
			y[j + 0] = hadd(xy0) + a11*x0;
			break;
		}
		}
	}
}
