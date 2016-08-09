#define _GNU_SOURCE
#include <stdlib.h>
#include <sys/mman.h>

#include "thp.h"

#define ALIGN __THP_ALIGNMENT

void * __malloc_thp(size_t size) {
	void * ptr = NULL;
	
	// get aligned memory
	int status = posix_memalign(&ptr, ALIGN, size);
	if (status != 0) {
		return NULL;
	}
	
	// suggest use of transparent hugepages
	madvise(ptr, size, MADV_HUGEPAGE);
	
	return ptr;
}

void * __malloc_thp_padded(size_t size) {
	size_t padded_size = ((size + ALIGN - 1) / ALIGN) * ALIGN;
	return malloc_thp(padded_size);
}
