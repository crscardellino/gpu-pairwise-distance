#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define __THP_ALIGNMENT (1u << 21)

#if defined __INTEL_COMPILER
#define __compiler_aligned_builtin(x) __assume_aligned(x, __THP_ALIGNMENT)
#elif defined __GNUC__
#define __compiler_aligned_builtin(x) __builtin_assume_aligned(x, __THP_ALIGNMENT)
#else
#define __compiler_aligned_builtin(x) x
#endif

void * __malloc_thp(size_t size);
void * __malloc_thp_padded(size_t size);

#define malloc_thp(sz) __compiler_aligned_builtin(__malloc_thp(sz))
#define malloc_thp_padded(sz) __compiler_aligned_builtin(__malloc_thp_padded(sz))

#ifdef __cplusplus
}
#endif
