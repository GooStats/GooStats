#pragma once

#include <thrust/detail/config.h>
#include <thrust/system_error.h>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <driver_types.h>      // Needed for cudaError_t
#endif

#include <cmath>
#include <string>
extern int host_callnumber;

#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
#include <cstring> // for std::memcpy
// OMP target - all 'device' memory is actually on host.
#define ALIGN(n)
#define MEM_DEVICE
#define MEM_SHARED
#define MEM_CONSTANT
#define EXEC_TARGET __host__
#define THREAD_SYNCH _Pragma("omp barrier") // valid in C99 and C++11, but probably not C++93
#define DEVICE_VECTOR thrust::host_vector
// Use char* here because I need +1 to mean "offset by one byte", not "by one sizeof(whatever)".
// Can't use void* because then the compiler doesn't know how to do pointer arithmetic.
// This will fail if sizeof(char) is more than 1. But that should never happen, right?
#define MEMCPY(target, source, count, direction) std::memcpy((char*) target, source, count)
#define MEMCPY_TO_SYMBOL(target, source, count, offset, direction) std::memcpy(((char*) target)+offset, source, count)
#define MEMCPY_FROM_SYMBOL(target, source, count, offset, direction) std::memcpy((char*) target, ((char*) source)+offset, count)
#define GET_FUNCTION_ADDR(fname) host_fcn_ptr = (void*) fname
#define SYNCH dummySynch
#define BLOCKIDX (1)
#define GRIDDIM (1)
void dummySynch();
#define CONST_PI M_PI
// Create my own error type to avoid __host__ redefinition
// conflict in Thrust from including driver_types.h
enum gooError {gooSuccess = 0, gooErrorMemoryAllocation};
#define RO_CACHE(x) (x)
#endif

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP

#define THREADIDX (omp_get_thread_num())
#define BLOCKDIM (omp_get_num_threads())

#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_TBB

#define THREADIDX (1)
#define BLOCKDIM (1)

#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA

// CUDA target - defaults
#define ALIGN(n) __align__(n)
#define MEM_DEVICE __device__
#define MEM_SHARED __shared__
#define MEM_CONSTANT __constant__
#define EXEC_TARGET __device__
#define SYNCH cudaDeviceSynchronize
#define THREAD_SYNCH __syncthreads();
#define DEVICE_VECTOR thrust::device_vector
#ifdef TARGET_SM35
#define RO_CACHE(x) __ldg(&x)
#else
#define RO_CACHE(x) (x)
#endif

#include <stdexcept>
#define GOOFIT_CUDA_CHECK(function)                                                                                    \
    {                                                                                                                  \
        cudaError err = function;                                                                                      \
        if(err != cudaSuccess) {                                                                                       \
	    throw std::runtime_error(std::string(cudaGetErrorString(err)));						       \
        }                                                                                                              \
    }

template <typename T>
void *get_device_symbol_address(const T &symbol) {
    void *result;
    GOOFIT_CUDA_CHECK(cudaMemcpyFromSymbol(&result, symbol, sizeof(void *)));
    return result;
}

#define MEMCPY(target, source, count, direction) GOOFIT_CUDA_CHECK(cudaMemcpy(target, source, count, direction));

#define MEMCPY_TO_SYMBOL(target, source, count, offset, direction) \
    GOOFIT_CUDA_CHECK(cudaMemcpyToSymbol(target, source, count, offset, direction));

#define MEMCPY_FROM_SYMBOL(target, source, count, offset, direction) \
    GOOFIT_CUDA_CHECK(cudaMemcpyFromSymbol(target, source, count, offset, direction));

// For CUDA case, just use existing errors, renamed
#include <driver_types.h>      // Needed for cudaError_t
enum gooError {gooSuccess = cudaSuccess,
               gooErrorMemoryAllocation = cudaErrorMemoryAllocation
              };
#define THREADIDX (threadIdx.x)
#define BLOCKDIM (blockDim.x)
#define BLOCKIDX (blockIdx.x)
#define GRIDDIM (gridDim.x)
#define CONST_PI CUDART_PI

#else

#define THREADIDX (1)
#define BLOCKDIM (1)

#endif

gooError gooMalloc(void** target, size_t bytes);
gooError gooFree(void* ptr);

#define DOUBLES 1

class PdfBase;
extern void abortWithCudaPrintFlush (std::string file, int line, std::string reason, const PdfBase* pdf = 0) ;

#ifdef DOUBLES
#define root2 1.4142135623730951
#define invRootPi 0.5641895835477563

typedef double fptype;
// Double math functions
#define ATAN2 atan2
#define COS cos
#define COSH cosh
#define SINH sinh
#define ERF erf
#define ERFC erfc
#define EXP exp
#define FABS fabs
#define FMOD fmod
#define LOG log
#define EVALLOG(x) ((x)<= 2.*2.2250738585072014e-308 ? (x)/2.*2.2250738585072014e-308+LOG(2.*2.2250738585072014e-308)-1: LOG(x))
#define MODF modf
#define SIN sin
#define SQRT sqrt
#ifdef TARGET_SM35
#define RSQRT rsqrt
#else
#define RSQRT 1.0/SQRT
#endif
#define FLOOR floor
#define POW pow
#else
typedef float fptype;

#define root2 1.4142135623730951f
#define invRootPi 0.5641895835477563f


// Float math functions
#define ATAN2 atan2f
#define COS cosf
#define COSH coshf
#define SINH sinhf
#define ERF erff
#define ERFC erfcf
#define EXP expf
#define FABS fabsf
#define FMOD fmodf
#define LOG logf
#define MODF modff
#define SIN sinf
#define SQRT sqrtf
#define FLOOR floorf
#define POW powf
#endif

