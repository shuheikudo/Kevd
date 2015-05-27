// Copyright by Shuhei Kudo, May 2015.
#include "dormtrb.h"
#include <omp.h>
#include "simd_tools.h"

static int dorm2lb15(int m, int n, double *a, int lda, double *tau, double *c, int ldc, double *work);

// this only works for side == 'L' and uplo == 'U', and breaks a...
int dormtrb(int m, int n, double* a, int lda, double* tau, double* c, int ldc, double* work)
{
	// quick return if possible
	if (m <= 1 || n == 0) {
		work[0] = 1.;
		return 0;
	}
	return dorm2lb15(m - 1, n, a+lda, lda, tau, c, ldc, work);
}


static void range(int n, int* begin, int* end)
{
	int nt = omp_get_num_threads();
	int id = omp_get_thread_num();
	int nn = n / nt;
	int res = n % nt;
	*begin = nn * id + (id < res ? id : res);
	*end = *begin + nn + (id < res ? 1 : 0);
}


double* getw(double* ww, int m, int N)
{
	m = N * ((m + 1) / 2 * 2);
	m = (m + 15) / 16 * 16;
	if ((m % 1024) == 0) m += 16;
	ww = cache_line(ww);
	return ww + m * omp_get_thread_num();
}


static int index_4soa2(int i)
{
	return i * i / 2 - i / 2;
}

static void reorder_4soa2(int m, int n, const double* a, int lda, double* work, double** h, double** ww, double* tau)
{
	double* vu = work;
	double* hh = work + ((m + 3) / 4) * 10;
	int i;
#pragma omp for schedule(static,4)
	for (i = 0; i < m; i += 4) {
		switch (m - i){
		default:
		{
			const double* h1 = a + (i + 0) * lda;
			const double* h2 = a + (i + 1) * lda;
			const double* h3 = a + (i + 2) * lda;
			const double* h4 = a + (i + 3) * lda;
			double* cur = hh + index_4soa2(i);
			double
				t12 = 0., t13 = 0., t14 = 0.,
				t23 = 0., t24 = 0.,
				t34 = 0.;
			d2v vt12, vt13, vt14, vt23, vt24, vt34;
			vt12 = vt13 = vt14 = vt23 = vt24 = vt34 = _mm_setzero_pd();
			int j;
#pragma loop noalias
#pragma loop smp
#pragma loop prefetch
#pragma loop simd aligned
			for (j = 0; j < i; j += 2) {
				d2v aj = ld(h1 + j);
				d2v bj = ld(h2 + j);
				d2v cj = ld(h3 + j);
				d2v dj = ld(h4 + j);
				vt12 = madd(aj, bj, vt12);
				vt13 = madd(aj, cj, vt13);
				vt14 = madd(aj, dj, vt14);
				vt23 = madd(bj, cj, vt23);
				vt24 = madd(bj, dj, vt24);
				vt34 = madd(cj, dj, vt34);
				st(cur + 4 * j + 0, aj);
				st(cur + 4 * j + 2, bj);
				st(cur + 4 * j + 4, cj);
				st(cur + 4 * j + 6, dj);
			}
			{
				t12 = hadd(vt12);
				t13 = hadd(vt13);
				t14 = hadd(vt14);
				t23 = hadd(vt23);
				t24 = hadd(vt24);
				t34 = hadd(vt34);
			}
			{
				double bj = h2[j], cj = h3[j], dj = h4[j];
				t12 += bj; t13 += cj; t14 += dj;
				t23 += bj * cj; t24 += bj * dj;
				t34 += cj * dj;
				double cj2 = h3[j + 1], dj2 = h4[j + 1];
				t23 += cj2; t24 += dj2;
				t34 += cj2 * dj2;
				double dj3 = h4[j + 2];
				t34 += dj3;
				cur[4 * j + 0] = bj;
				cur[4 * j + 1] = cj;
				cur[4 * j + 2] = cj2;
				cur[4 * j + 3] = dj;
				cur[4 * j + 4] = dj2;
				cur[4 * j + 5] = dj3;
			}

			double
				s11,
				s21, s22,
				s31, s32, s33,
				s41, s42, s43, s44;
			s11 = -tau[i + 0];
			s22 = -tau[i + 1];
			s33 = -tau[i + 2];
			s44 = -tau[i + 3];
			//
			s21 = s22 * t12 * s11;
			// s33 * [t13 t23] * T
			s31 = s33 * (t13 * s11 + t23 * s21);
			s32 = s33 * (t23 * s22);
			// s44 * [t14 t24 t34] * T
			s41 = s44 * (t14 * s11 + t24 * s21 + t34 * s31);
			s42 = s44 * (t24 * s22 + t34 * s32);
			s43 = s44 * (t34 * s33);

			double* vv = vu + i / 4 * 10;
			vv[0] = s11;
			vv[1] = s21;
			vv[2] = s22;
			vv[3] = s31;
			vv[4] = s41;
			vv[5] = s32;
			vv[6] = s42;
			vv[7] = s33;
			vv[8] = s43;
			vv[9] = s44;
		}
			break;
		case 3:
		{
			const double* h1 = a + (i + 0) * lda;
			const double* h2 = a + (i + 1) * lda;
			const double* h3 = a + (i + 2) * lda;
			double* cur = hh + index_4soa2(i);
			double
				t12 = 0., t13 = 0.,
				t23 = 0.;
			int j;
			for (j = 0; j < i; j += 2) { // not vectorized yet
				double aj = h1[j], bj = h2[j], cj = h3[j];
				t12 += aj * bj; t13 += aj * cj;
				t23 += bj * cj;
				double aj2 = h1[j + 1], bj2 = h2[j + 1], cj2 = h3[j + 1];
				t12 += aj2 * bj2; t13 += aj2 * cj2;
				t23 += bj2 * cj2;
				cur[3 * j] = aj;
				cur[3 * j + 1] = aj2;
				cur[3 * j + 2] = bj;
				cur[3 * j + 3] = bj2;
				cur[3 * j + 4] = cj;
				cur[3 * j + 5] = cj2;
			}
			{
				double bj = h2[j], cj = h3[j];
				t12 += bj; t13 += cj;
				t23 += bj * cj;
				double cj2 = h3[j + 1];
				t23 += cj2;
				cur[3 * j + 0] = bj;
				cur[3 * j + 1] = cj;
				cur[3 * j + 2] = cj2;
			}

			double
				s11,
				s21, s22,
				s31, s32, s33;
			s11 = -tau[i + 0];
			s22 = -tau[i + 1];
			s33 = -tau[i + 2];
			//
			s21 = s22 * t12 * s11;
			// s33 * [t13 t23] * T
			s31 = s33 * (t13 * s11 + t23 * s21);
			s32 = s33 * (t23 * s22);

			double* vv = vu + i / 4 * 10;
			vv[0] = s11;
			vv[1] = s21;
			vv[2] = s22;
			vv[3] = s31;
			vv[4] = s32;
			vv[5] = s33;
		}
			break;
		case 2:
		{
			const double* h1 = a + (i + 0) * lda;
			const double* h2 = a + (i + 1) * lda;
			double* cur = hh + index_4soa2(i);
			double t12 = 0.;
			int j;
			for (j = 0; j < i; j += 2) { // not vectorized yet
				double aj = h1[j], bj = h2[j];
				t12 += aj * bj;
				double aj2 = h1[j + 1], bj2 = h2[j + 1];
				t12 += aj2 * bj2;
				cur[2 * j] = aj;
				cur[2 * j + 1] = aj2;
				cur[2 * j + 2] = bj;
				cur[2 * j + 3] = bj2;
			}
			{
				double bj = h2[j];
				t12 += bj;
				cur[2 * j + 0] = bj;
			}

			double
				s11,
				s21, s22;
			s11 = -tau[i + 0];
			s22 = -tau[i + 1];
			//
			s21 = s22 * t12 * s11;

			double* vv = vu + i / 4 * 10;
			vv[0] = s11;
			vv[1] = s21;
			vv[2] = s22;
		}
			break;
		case 1:
		{
			const double* h1 = a + (i + 0) * lda;
			double* cur = hh + index_4soa2(i);
			int j;
			for (j = 0; j < i; j += 2) {
				double aj = h1[j];
				double aj2 = h1[j + 1];
				cur[j] = aj;
				cur[j + 1] = aj2;
			}
			double* vv = vu + i / 4 * 10;
			vv[0] = -tau[i];
		}
			break;
		}
	}
	*h = hh;
	int ii = (m + 3) / 4 * 4;
	*ww = hh + index_4soa2(ii);
	if (!is_octword_aligned(*ww))++*ww;
}


#define DO0(X, A, B, C) 
#define DO1(X, A, B, C) A ## 0 = X(B##0, C##0)
#define DO2(X, A, B, C) DO1(X, A, B, C); A ## 1 = X(B##1, C##1)
#define DO3(X, A, B, C) DO2(X, A, B, C); A ## 2 = X(B##2, C##2)
#define DO4(X, A, B, C) DO3(X, A, B, C); A ## 3 = X(B##3, C##3)
#define DO5(X, A, B, C) DO4(X, A, B, C); A ## 4 = X(B##4, C##4)
#define DO6(X, A, B, C) DO5(X, A, B, C); A ## 5 = X(B##5, C##5)
#define DO7(X, A, B, C) DO6(X, A, B, C); A ## 6 = X(B##6, C##6)
#define DO8(X, A, B, C) DO7(X, A, B, C); A ## 7 = X(B##7, C##7)
#define DO9(X, A, B, C) DO8(X, A, B, C); A ## 8 = X(B##8, C##8)
#define DO10(X, A, B, C) DO9(X, A, B, C); A ## 9 = X(B##9, C##9)
#define DO(X, A, B, C, N) DO##N(X, A, B, C)

#define DOB1(X, A, B, C) A ## 0 = X(B, C##0)
#define DOB2(X, A, B, C) DOB1(X, A, B, C); A ## 1 = X(B, C##1)
#define DOB3(X, A, B, C) DOB2(X, A, B, C); A ## 2 = X(B, C##2)
#define DOB4(X, A, B, C) DOB3(X, A, B, C); A ## 3 = X(B, C##3)
#define DOB5(X, A, B, C) DOB4(X, A, B, C); A ## 4 = X(B, C##4)
#define DOB6(X, A, B, C) DOB5(X, A, B, C); A ## 5 = X(B, C##5)
#define DOB7(X, A, B, C) DOB6(X, A, B, C); A ## 6 = X(B, C##6)
#define DOB8(X, A, B, C) DOB7(X, A, B, C); A ## 7 = X(B, C##7)
#define DOB9(X, A, B, C) DOB8(X, A, B, C); A ## 8 = X(B, C##8)
#define DOB10(X, A, B, C) DOB9(X, A, B, C); A ## 9 = X(B, C##9)
#define DOB(X, A, B, C, N) DOB##N(X, A, B, C)

#define Def0(A) 
#define Def1(A) d2v A##0
#define Def2(A) Def1(A), A##1
#define Def3(A) Def2(A), A##2
#define Def4(A) Def3(A), A##3
#define Def5(A) Def4(A), A##4
#define Def6(A) Def5(A), A##5
#define Def7(A) Def6(A), A##6
#define Def8(A) Def7(A), A##7
#define Def9(A) Def8(A), A##8
#define Def10(A) Def9(A), A##9
#define Def(A, N) Def##N(A)

#define Defmv0(A, B)
#define Defmv1(A, B) Def(A##0, B)
#define Defmv2(A, B) Defmv1(A, B); Def(A##1, B)
#define Defmv3(A, B) Defmv2(A, B); Def(A##2, B)
#define Defmv4(A, B) Defmv3(A, B); Def(A##3, B)
#define Defmv5(A, B) Defmv4(A, B); Def(A##4, B)
#define Defmv6(A, B) Defmv5(A, B); Def(A##5, B)
#define Defmv(A, B, C) Defmv##C(A, B)

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

#define Zeromv0(A, N)
#define Zeromv1(A, N) Zero(A##0, N)
#define Zeromv2(A, N) Zeromv1(A, N); Zero(A##1, N)
#define Zeromv3(A, N) Zeromv2(A, N); Zero(A##2, N)
#define Zeromv4(A, N) Zeromv3(A, N); Zero(A##3, N)
#define Zeromv5(A, N) Zeromv4(A, N); Zero(A##4, N)
#define Zeromv6(A, N) Zeromv5(A, N); Zero(A##5, N)
#define Zeromv(A, N, M) Zeromv##M(A, N)

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
#define Ld(A, W, K, N) Ld##N(A, W, K, N)

#define Ldd0(A, W, I, LDW, K)
#define Ldd1(A, W, I, LDW, K) A##0 = ld(W + (I + 0) * LDW + K)
#define Ldd2(A, W, I, LDW, K) Ldd1(A, W, I, LDW, K); A##1 = ld(W + (I + 1) * LDW + K)
#define Ldd3(A, W, I, LDW, K) Ldd2(A, W, I, LDW, K); A##2 = ld(W + (I + 2) * LDW + K)
#define Ldd4(A, W, I, LDW, K) Ldd3(A, W, I, LDW, K); A##3 = ld(W + (I + 3) * LDW + K)
#define Ldd5(A, W, I, LDW, K) Ldd4(A, W, I, LDW, K); A##4 = ld(W + (I + 4) * LDW + K)
#define Ldd6(A, W, I, LDW, K) Ldd5(A, W, I, LDW, K); A##5 = ld(W + (I + 5) * LDW + K)
#define Ldd7(A, W, I, LDW, K) Ldd6(A, W, I, LDW, K); A##6 = ld(W + (I + 6) * LDW + K)
#define Ldd8(A, W, I, LDW, K) Ldd7(A, W, I, LDW, K); A##7 = ld(W + (I + 7) * LDW + K)
#define Ldd9(A, W, I, LDW, K) Ldd8(A, W, I, LDW, K); A##8 = ld(W + (I + 8) * LDW + K)
#define Ldd10(A, W, I, LDW, K) Ldd9(A, W, I, LDW, K); A##9 = ld(W + (I + 9) * LDW + K)
#define Ldd(A, W, I, LDW, K, N) Ldd##N(A, W, I, LDW, K)

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
#define St(A, W, K, N) St##N(A, W, K, N)

#define Stt0(A, W, I, LDW, K)
#define Stt1(A, W, I, LDW, K) st(W + (I + 0) * LDW + K, A##0)
#define Stt2(A, W, I, LDW, K) Stt1(A, W, I, LDW, K); st(W + (I + 1) * LDW + K, A##1)
#define Stt3(A, W, I, LDW, K) Stt2(A, W, I, LDW, K); st(W + (I + 2) * LDW + K, A##2)
#define Stt4(A, W, I, LDW, K) Stt3(A, W, I, LDW, K); st(W + (I + 3) * LDW + K, A##3)
#define Stt5(A, W, I, LDW, K) Stt4(A, W, I, LDW, K); st(W + (I + 4) * LDW + K, A##4)
#define Stt6(A, W, I, LDW, K) Stt5(A, W, I, LDW, K); st(W + (I + 5) * LDW + K, A##5)
#define Stt7(A, W, I, LDW, K) Stt6(A, W, I, LDW, K); st(W + (I + 6) * LDW + K, A##6)
#define Stt8(A, W, I, LDW, K) Stt7(A, W, I, LDW, K); st(W + (I + 7) * LDW + K, A##7)
#define Stt9(A, W, I, LDW, K) Stt8(A, W, I, LDW, K); st(W + (I + 8) * LDW + K, A##8)
#define Stt10(A, W, I, LDW, K) Stt9(A, W, I, LDW, K); st(W + (I + 9) * LDW + K, A##9)
#define Stt(A, W, I, LDW, K, N) Stt##N(A, W, I, LDW, K)

#define Sttlo0(A, W, I, LDW, K)
#define Sttlo1(A, W, I, LDW, K) _mm_storel_pd(W + (I + 0) * LDW + K, A##0)
#define Sttlo2(A, W, I, LDW, K) Sttlo1(A, W, I, LDW, K); _mm_storel_pd(W + (I + 1) * LDW + K, A##1)
#define Sttlo3(A, W, I, LDW, K) Sttlo2(A, W, I, LDW, K); _mm_storel_pd(W + (I + 2) * LDW + K, A##2)
#define Sttlo4(A, W, I, LDW, K) Sttlo3(A, W, I, LDW, K); _mm_storel_pd(W + (I + 3) * LDW + K, A##3)
#define Sttlo5(A, W, I, LDW, K) Sttlo4(A, W, I, LDW, K); _mm_storel_pd(W + (I + 4) * LDW + K, A##4)
#define Sttlo6(A, W, I, LDW, K) Sttlo5(A, W, I, LDW, K); _mm_storel_pd(W + (I + 5) * LDW + K, A##5)
#define Sttlo7(A, W, I, LDW, K) Sttlo6(A, W, I, LDW, K); _mm_storel_pd(W + (I + 6) * LDW + K, A##6)
#define Sttlo8(A, W, I, LDW, K) Sttlo7(A, W, I, LDW, K); _mm_storel_pd(W + (I + 7) * LDW + K, A##7)
#define Sttlo9(A, W, I, LDW, K) Sttlo8(A, W, I, LDW, K); _mm_storel_pd(W + (I + 8) * LDW + K, A##8)
#define Sttlo10(A, W, I, LDW, K) Sttlo9(A, W, I, LDW, K); _mm_storel_pd(W + (I + 9) * LDW + K, A##9)
#define Sttlo(A, W, I, LDW, K, N) Sttlo##N(A, W, I, LDW, K)

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

#define Maddr1_0(A, B, C, N)
#define Maddr1_1(A, B, C, N) Madd##N(A, B##0, C##0)
#define Maddr1_2(A, B, C, N) Maddr1_1(A, B, C, N); Madd##N(A, B##1, C##1)
#define Maddr1_3(A, B, C, N) Maddr1_2(A, B, C, N); Madd##N(A, B##2, C##2)
#define Maddr1_4(A, B, C, N) Maddr1_3(A, B, C, N); Madd##N(A, B##3, C##3)
#define Maddr1_5(A, B, C, N) Maddr1_4(A, B, C, N); Madd##N(A, B##4, C##4)
#define Maddr1_6(A, B, C, N) Maddr1_5(A, B, C, N); Madd##N(A, B##5, C##5)
#define Maddr1(A, B, C, N, M) Maddr1_##M(A, B, C, N)

#define Maddmv_0(A, B, C, N)
#define Maddmv_1(A, B, C, N) Madd##N(A##0, B##0, C)
#define Maddmv_2(A, B, C, N) Maddmv_1(A, B, C, N); Madd##N(A##1, B##1, C)
#define Maddmv_3(A, B, C, N) Maddmv_2(A, B, C, N); Madd##N(A##2, B##2, C)
#define Maddmv_4(A, B, C, N) Maddmv_3(A, B, C, N); Madd##N(A##3, B##3, C)
#define Maddmv_5(A, B, C, N) Maddmv_4(A, B, C, N); Madd##N(A##4, B##4, C)
#define Maddmv_6(A, B, C, N) Maddmv_5(A, B, C, N); Madd##N(A##5, B##5, C)
#define Maddmv(A, B, C, N, M) Maddmv_##M(A, B, C, N)

#define Mul(A, B, C, N) DOB(_mm_mul_pd, A, B, C, N)

#define Red0(A)
#define Red1(A) A##0 = _mm_add_pd(_mm_shuffle_pd(A##0, A##0, 0x1u), A##0)
#define Red2(A) Red1(A); A##1 = _mm_add_pd(_mm_shuffle_pd(A##1, A##1, 0x1u), A##1)
#define Red3(A) Red2(A); A##2 = _mm_add_pd(_mm_shuffle_pd(A##2, A##2, 0x1u), A##2)
#define Red4(A) Red3(A); A##3 = _mm_add_pd(_mm_shuffle_pd(A##3, A##3, 0x1u), A##3)
#define Red5(A) Red4(A); A##4 = _mm_add_pd(_mm_shuffle_pd(A##4, A##4, 0x1u), A##4)
#define Red6(A) Red5(A); A##5 = _mm_add_pd(_mm_shuffle_pd(A##5, A##5, 0x1u), A##5)
#define Red7(A) Red6(A); A##6 = _mm_add_pd(_mm_shuffle_pd(A##6, A##6, 0x1u), A##6)
#define Red8(A) Red7(A); A##7 = _mm_add_pd(_mm_shuffle_pd(A##7, A##7, 0x1u), A##7)
#define Red9(A) Red8(A); A##8 = _mm_add_pd(_mm_shuffle_pd(A##8, A##8, 0x1u), A##8)
#define Red10(A) Red9(A); A##9 = _mm_add_pd(_mm_shuffle_pd(A##9, A##9, 0x1u), A##9)
#define Red(A, N) Red##N(A)

#define Unpacklo(A, B, N) DO(_mm_unpacklo_pd, A, B, B, N)
#define Unpackhi(A, B, N) DO(_mm_unpackhi_pd, A, B, B, N)

// r1 + mv kernel
// compute W += Y_1 A_1 and B_2 = Y^T W
// after this routine, compute A_2 = T B_2 where T is a lower triangular matrix of a compact WY representation
#define r1mvkernel_4(N, MC, MN)		\
	for (k = 0; k < i; k += 4){		\
		Def(c1, MC);				\
		Def(n1, MN);				\
		Def(x, N);					\
		Ld(c1, hc, k + 2, MC);		\
		Ld(n1, hn, k + 2, MN);		\
		Ld(x, w, k + 2, N);			\
		Maddr1(w, c0, t, N, MC);	\
		Maddmv(r, n0, w, N, MN);	\
		St(w, w, k, N);				\
		Ld(c0, hc, k + 4, MC);		\
		Ld(n0, hn, k + 4, MN);		\
		Ld(w, w, k + 4, N);			\
		Maddr1(x, c1, t, N, MC);	\
		Maddmv(r, n1, x, N, MN);	\
		St(x, w, k + 2, N);			\
	}

// trr1
// compute W += |1 * * *| A_1
//              |  1 * *|
//              |    1 *|
//              |      1|
// for M == 4
#define trr1_kernel4(N)						\
		d2v c12, c13;						\
		c00 = _mm_set_pd(0., 1.);			\
		c01 = _mm_set_pd(1., hh1[0]);		\
		c02 = _mm_set_pd(hh1[2], hh1[1]);	\
		c03 = _mm_set_pd(hh1[4], hh1[3]);	\
		c12 = c00;							\
		c13 = _mm_set_pd(1., hh1[5]);		\
		Maddr1(w, c0, t, N, 4);				\
		Madd(x, c12, t2, N);				\
		Madd(x,	c13, t3, N);
// for M == 3
#define trr1_kernel3(N)						\
		d2v c12;							\
		c00 = _mm_set_pd(0., 1.);			\
		c01 = _mm_set_pd(1., hh1[0]);		\
		c02 = _mm_set_pd(hh1[2], hh1[1]);	\
		c12 = c00;							\
		Maddr1(w, c0, t, N, 3);				\
		Madd(x, c12, t2, N);
// for M == 2
#define trr1_kernel2(N)						\
		c00 = _mm_set_pd(0., 1.);			\
		c01 = _mm_set_pd(1., hh1[0]);		\
		Maddr1(w, c0, t, N, 2);
// for M == 1
#define trr1_kernel1(N)						\
		c00 = _mm_set_pd(0., 1.);			\
		Maddr1(w, c0, t, N, 1);

#define trr1_kernel(N, M) {trr1_kernel##M(N);}


// trmv kernel
// |*      | |1 * * *|T |wl0 wl1 wl2 wl3 ...|
// |* *    | |  1 * *|  |wh0 wh1 wh2 wh3 ...|
// |* * *  | |    1 *|  |xl0 xl1 xl2 xl3 ...|
// |* * * *| |      1|  |xh0 xh1 xh2 xh3 ...|
// for M == 4
#define trmvU_kernelA4(N)					\
		Def(w, N);							\
		Def(x, N);							\
		d2v b10, b11, b12, b13;				\
		d2v b22, b23;						\
		Ldd(w, c, j, ldc, k, N);			\
		Ldd(x, c, j, ldc, k + 2, N);		\
		b10 = _mm_set_pd(0., 1.);			\
		b11 = _mm_set_pd(1., hh2[0]);		\
		b12 = _mm_set_pd(hh2[2], hh2[1]);	\
		b13 = _mm_set_pd(hh2[4], hh2[3]);	\
		b22 = b10;							\
		b23 = _mm_set_pd(1., hh2[5]);
#define trmvU_kernelB4(N)					\
		Madd(r0, b10, w, N);				\
		Madd(r1, b11, w, N);				\
		Madd(r2, b12, w, N);				\
		Madd(r3, b13, w, N);
#define trmvU_kernelB_init4(N)				\
		Mul(r0, b10, w, N);					\
		Mul(r1, b11, w, N);					\
		Mul(r2, b12, w, N);					\
		Mul(r3, b13, w, N);
#define trmvU_kernelC4(N)					\
		Madd(r2, b22, x, N);				\
		Madd(r3, b23, x, N);				\
		St(w, w, k, N);						\
		St(x, w, k + 2, N);					\
		Red(r0, N);							\
		Red(r1, N);							\
		Red(r2, N);							\
		Red(r3, N);
#define trmvL_kernel4(N)					\
		d2v d0 = _mm_set_pd(vv[1], vv[0]);	\
		d2v d1 = _mm_set_pd(vv[2], 0.);		\
		d2v d2 = _mm_set_pd(vv[4], vv[3]);	\
		d2v d3 = _mm_set_pd(vv[6], vv[5]);	\
		d2v d4 = _mm_set_pd(vv[8], vv[7]);	\
		d2v d5 = _mm_set_pd(vv[9], 0.);		\
		Mul(w, d0, r0, N);					\
		Madd(w, d1, r1, N);					\
		Mul(x, d2, r0, N);					\
		Madd(x, d3, r1, N);					\
		Madd(x, d4, r2, N);					\
		Madd(x, d5, r3, N);					\
		Unpacklo(t0, w, N);					\
		Unpackhi(t1, w, N);					\
		Unpacklo(t2, x, N);					\
		Unpackhi(t3, x, N);

// for M == 3
#define trmvU_kernelA3(N)					\
		Def(w, N);							\
		Def(x, N);							\
		d2v b10, b11, b12;					\
		d2v b22;							\
		Ldd(w, c, j, ldc, k, N);			\
		Ldd(x, c, j, ldc, k + 2, N);		\
		b10 = _mm_set_pd(0., 1.);			\
		b11 = _mm_set_pd(1., hh2[0]);		\
		b12 = _mm_set_pd(hh2[2], hh2[1]);	\
		b22 = b10;
#define trmvU_kernelB3(N)					\
		Madd(r0, b10, w, N);				\
		Madd(r1, b11, w, N);				\
		Madd(r2, b12, w, N);
#define trmvU_kernelB_init3(N)				\
		Mul(r0, b10, w, N);					\
		Mul(r1, b11, w, N);					\
		Mul(r2, b12, w, N);
#define trmvU_kernelC3(N)					\
		Madd(r2, b22, x, N);				\
		St(w, w, k, N);						\
		St(x, w, k + 2, N);					\
		Red(r0, N);							\
		Red(r1, N);							\
		Red(r2, N);
#define trmvL_kernel3(N)					\
		d2v d0 = _mm_set_pd(vv[1], vv[0]);	\
		d2v d1 = _mm_set_pd(vv[2], 0.);		\
		d2v d2 = _mm_set_pd(0., vv[3]);		\
		d2v d3 = _mm_set_pd(0., vv[4]);		\
		d2v d4 = _mm_set_pd(0., vv[5]);		\
		Mul(w, d0, r0, N);					\
		Madd(w, d1, r1, N);					\
		Mul(x, d2, r0, N);					\
		Madd(x, d3, r1, N);					\
		Madd(x, d4, r2, N);					\
		Unpacklo(t0, w, N);					\
		Unpackhi(t1, w, N);					\
		Unpacklo(t2, x, N);

// for M == 2
#define trmvU_kernelA2(N)					\
		Def(w, N);							\
		Def(x, N);							\
		d2v b10, b11;						\
		Ldd(w, c, j, ldc, k, N);			\
		b10 = _mm_set_pd(0., 1.);			\
		b11 = _mm_set_pd(1., hh2[0]);
#define trmvU_kernelB2(N)					\
		Madd(r0, b10, w, N);				\
		Madd(r1, b11, w, N);
#define trmvU_kernelB_init2(N)				\
		Mul(r0, b10, w, N);					\
		Mul(r1, b11, w, N);
#define trmvU_kernelC2(N)					\
		St(w, w, k, N);						\
		Red(r0, N);							\
		Red(r1, N);
#define trmvL_kernel2(N)					\
		d2v d0 = _mm_set_pd(vv[1], vv[0]);	\
		d2v d1 = _mm_set_pd(vv[2], 0.);		\
		Mul(w, d0, r0, N);					\
		Madd(w, d1, r1, N);					\
		Unpacklo(t0, w, N);					\
		Unpackhi(t1, w, N);

// for M == 1
#define trmvU_kernelA1(N)					\
		Def(w, N);							\
		Def(x, N);							\
		d2v b10;							\
		Ldd(w, c, j, ldc, k, N);			\
		b10 = _mm_set_pd(0., 1.);
#define trmvU_kernelB_init1(N)				\
		Mul(r0, b10, w, N);
#define trmvU_kernelB1(N)					\
		Madd(r0, b10, w, N);
#define trmvU_kernelC1(N)					\
		St(w, w, k, N);						\
		Red(r0, N);
#define trmvL_kernel1(N)					\
		d2v d0 = set1(vv[0]);				\
		Mul(w, d0, r0, N);					\
		Unpacklo(t0, w, N); 

#define trmv_init(N, M){					\
		Defmv(r, N, M);						\
		trmvU_kernelA##M(N);				\
		trmvU_kernelB_init##M(N);			\
		trmvU_kernelC##M(N);				\
		trmvL_kernel##M(N);					\
	}
#define trmv_kernel(N, M){					\
		trmvU_kernelA##M(N);				\
		trmvU_kernelB##M(N);				\
		trmvU_kernelC##M(N);				\
		trmvL_kernel##M(N);					\
	}


#define main_kernel(N, MC, MN){					\
		Defmv(r, N, MN);						\
		Zeromv(r, N, MN);						\
		const double* hc = h + index_4soa2(i);	\
		const double* hn = h + index_4soa2(i + MC);\
		int k = 0;								\
		Def(c0, MC);							\
		Def(n0, MN);							\
		Def(w, N);								\
		Ld(c0, hc, k, MC);						\
		Ld(n0, hn, k, MN);						\
		Ld(w, w, k, N);							\
		r1mvkernel_4(N, MC, MN);				\
		Def(x, N);								\
		Ld(x, w, k + 2, N);						\
		const double* hh1 = hc + (MC) * k;		\
		trr1_kernel(N, MC);						\
		Def(n1, MN);							\
		Ld(n1, hn, k + 2, MN);					\
		Maddmv(r, n0, w, N, MN);				\
		Maddmv(r, n1, x, N, MN);				\
		St(w, w, k, N);							\
		St(x, w, k + 2, N);						\
		k += MC;								\
		const double* hh2 = hn + (MN) * k;		\
		const double* vv = vu + (i + MC)/ MC * (MC * (MC + 1) / 2);\
		trmv_kernel(N, MN);						\
	}

#define rank1_kernelA(N, M)					\
	const double* hc = h + index_4soa2(i);	\
	int k = 0;								\
	Def(c0, M);								\
	Def(w, N);								\
	Ld(c0, hc, k, M);						\
	Ld(w, w, k, N);							\
	for (k = 0; k < i; k += 4){				\
		Def(c1, M);							\
		Def(x, N);							\
		Ld(c1, hc, k + 2, M);				\
		Ld(x, w, k + 2, N);					\
		Maddr1(w, c0, t, N, M);				\
		Stt(w, c, j, ldc, k, N);			\
		Ld(c0, hc, k + 4, M);				\
		Ld(w, w, k + 4, N);					\
		Maddr1(x, c1, t, N, M);				\
		Stt(x, c, j, ldc, k + 2, N);		\
	}

#define rank1_kernelB4(N, M)				\
	Def(x, N);								\
	Ld(x, w, k + 2, N);						\
	const double* hh1 = hc + (M) * k;		\
	trr1_kernel(N, M);						\
	Stt(w, c, j, ldc, k, N);				\
	Stt(x, c, j, ldc, k + 2, N);

#define rank1_kernelB3(N, M)				\
	Def(x, N);								\
	Ld(x, w, k + 2, N);						\
	const double* hh1 = hc + (M) * k;		\
	trr1_kernel(N, M);						\
	Stt(w, c, j, ldc, k, N);				\
	Sttlo(x, c, j, ldc, k + 2, N);

#define rank1_kernelB2(N, M)				\
	const double* hh1 = hc + (M) * k;		\
	trr1_kernel(N, M);						\
	Stt(w, c, j, ldc, k, N);

#define rank1_kernelB1(N, M)				\
	const double* hh1 = hc + (M) * k;		\
	trr1_kernel(N, M);						\
	Sttlo(w, c, j, ldc, k, N);

#define rank1_kernel(N, M) {				\
		rank1_kernelA(N,M);					\
		rank1_kernelB##M(N,M);				\
	}
#define load_rest(N) {						\
		for(k = i; k < m; k += 2) {			\
			Def(w, N);						\
			Ldd(w, c, j, ldc, k, N);		\
			St(w, w, k, N);					\
		}									\
	}

#define main_loop(N) {\
		int i = 0, k = 0;					\
		Defmv(t, N, 4);						\
		const double* hh2 = h;				\
		const double* vv = vu;				\
		trmv_init(N, 4);					\
		for (i = 0; i + 4 < m / 4 * 4; i += 4){\
			main_kernel(N, 4, 4);			\
			}								\
		switch (m - i){						\
		case 4:								\
			rank1_kernel(N, 4);				\
			break;							\
		case 5:								\
			load_rest(N);					\
			main_kernel(N, 4, 1);			\
			i += 4;							\
			rank1_kernel(N, 1);				\
			break;							\
		case 6:								\
			load_rest(N);					\
			main_kernel(N, 4, 2);			\
			i += 4;							\
			rank1_kernel(N, 2);				\
			break;							\
		case 7:								\
			load_rest(N);					\
			main_kernel(N, 4, 3);			\
			i += 4;							\
			rank1_kernel(N, 3);				\
			break;							\
		}									\
	}

#define def_main_loop(N)\
	static void main_loop##N(int m, int n, int j, double* c, int ldc, double* h, double* ww, double* w, const double* vu) \
	{ main_loop(N);}
//def_main_loop(9)
//def_main_loop(8)
def_main_loop(7)
def_main_loop(6)
def_main_loop(5)
def_main_loop(4)
def_main_loop(3)
def_main_loop(2)
//def_main_loop(1)

// 4-WY for k (k-simd)
int dorm2lb15(int m, int n, double *a, int lda, double *tau, double *c, int ldc, double *work)
{
#pragma procedure noalias
#pragma procedure nofltld

	if (!is_octword_aligned(work)) ++work;
#pragma omp parallel
	{
		double* h;
		double* ww;
		reorder_4soa2(m, n, a, lda, work, &h, &ww, tau);
		double* w = getw(ww, m, 8);
		const double* vu = work;
		int j, jbegin, jend;
		range(n, &jbegin, &jend);
		for (j = jbegin; j < jend; j += 8) {
			switch (jend - j){
			default:
				main_loop(8);
				break;
			case 7:
				main_loop7(m, n, j, c, ldc, h, ww, w, vu);
				break;
			case 6:
				main_loop6(m, n, j, c, ldc, h, ww, w, vu);
				break;
			case 5:
				main_loop5(m, n, j, c, ldc, h, ww, w, vu);
				break;
			case 4:
				main_loop4(m, n, j, c, ldc, h, ww, w, vu);
				break;
			case 3:
				main_loop3(m, n, j, c, ldc, h, ww, w, vu);
				break;
			case 2:
				main_loop2(m, n, j, c, ldc, h, ww, w, vu);
				break;
			case 1:
				main_loop(1);
				break;
			}

		}
	}
	return 0;
}

