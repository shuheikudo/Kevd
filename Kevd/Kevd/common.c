// Copyright by Shuhei Kudo, May 2015.
#ifndef _MSC_VER
#if defined(_WIN32) || defined(_WIN64)
#include <x86intrin.h>
void* bje_alloc(size_t size)
{
	return _mm_malloc(size, 64); 
}
void bje_free(void* p)
{
	_mm_free(p);
}

#else
#include <stdlib.h>
#include <alloca.h>
#include "common.h"

void* bje_alloc(size_t size)
{
	void* p;
	if (posix_memalign(&p, 16, size)) return 0;
	return p;
}
void bje_free(void* p)
{
	free(p);
}
#endif
#endif