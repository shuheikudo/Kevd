// Copyright by Shuhei Kudo, May 2015.
#ifndef _MSC_VER
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