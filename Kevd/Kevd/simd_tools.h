// Copyright by Shuhei Kudo, May 2015.
#ifndef SIMD_TOOLS_H
#define SIMD_TOOLS_H
#include <emmintrin.h>
#include "common.h"
typedef __m128d d2v;

#define ld(A) _mm_load_pd((A))
#define st(A,B) _mm_store_pd((A),(B))
#define set1(A) _mm_set_pd((A),(A))


#ifdef __FUJITSU
#define madd(A,B,C) _fjsp_madd_v2r8((A),(B),(C))
#define maddu(A,B,C) _fjsp_madd_cp_sr1_v2r8((A),(B),(C))
#define maddl(A,B,C) _fjsp_madd_cp_v2r8((A),(B),(C))
#define madds(A,B,C) _fjsp_madd_sr1_v2r8((A),(B),(C))
#define hadd_(A,B) _mm_storeh_pd((A),_mm_add_pd((B),_mm_unpacklo_pd((B),(B))))
#define hadd(A) ({	\
	double t;		\
	hadd_(&t, (A));	\
	t;				\
})
//#define hadd(A) (*((double*)&A)+*(((double*)&A)+1))
/*inline static double hadd(d2v a){
	double t;
	hadd_(&t, a);
	return t;
}*/ // this function cannot be inlined.
#define prefetch(A) __builtin_prefetch(A, 0, 2)
#define prefetch2(A) __builtin_prefetch(A, 0, 3)

#define likely(A) __builtin_expect(!!(A), 1)
#define unlikely(A) __builtin_expect(!!(A), 0)

#else
#define madd(A,B,C) _mm_add_pd(_mm_mul_pd((A),(B)),(C))
#define maddu(A,B,C) _mm_add_pd(_mm_mul_pd(_mm_unpackhi_pd((A),(A)),(B)),(C))
#define maddp(A,B,C) _mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd((A),(A)),(B)),(C))
// skewed-madd
#define madds(A,B,C) _mm_add_pd(_mm_mul_pd(_mm_shuffle_pd((A),(A),0x1u), (B)), (C))
// The code below can be compiled to sse3's hadd by gcc 4.8.1
#define hadd(A) (*((double*)&A)+*(((double*)&A)+1))
#define prefetch(A) _mm_prefetch(A, _MM_HINT_T1)
#define prefetch2(A) _mm_prefetch(A, _MM_HINT_T0)

#define likely(A) (!!(A))
#define unlikely(A) (!!(A))

#endif

#define rank2(U1,V1,U2,V2,A) madd((U1),(V1),madd((U2),(V2),(A)))
#define is_octword_aligned(A) ((((UPTR_T)(A)) & 0x0full) == 0)
#define cache_line(A) ((((UPTR_T)(A) + 127ull) & 0xffffffffffffff80ull))

#endif