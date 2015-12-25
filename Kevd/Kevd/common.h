// Copyright by Shuhei Kudo, May 2015.
#ifndef COMMON_H
#define COMMON_H
// functions for compatibility

#ifdef _MSC_VER
#define inline __inline
#define NOINLINE
// 16-byte alighned alloc
#include <stdlib.h>
#include <malloc.h>
inline void* bje_alloc(size_t size)
{
	return _aligned_malloc(size, 16);
}
inline void bje_free(void* p)
{
	_aligned_free(p);
}
#define bje_alloca(size) _malloca(size)
typedef uintptr_t UPTR_T;
#define PRAGMA(A) __pragma(A)
#else
#define NOINLINE __attribute__((noinline))
#include <stdlib.h>
void* bje_alloc(size_t size);
void bje_free(void* p);
#define bje_alloca(size) alloca(size)
typedef unsigned long long UPTR_T;
#define PRAGMA(A) _Pragma(#A)
#endif

#endif