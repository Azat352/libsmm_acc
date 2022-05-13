
#include <iostream>
#include <vector>
#include <limits>
#include "cutlass_acc.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"


#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


extern "C"
{
// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<32, 32, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 16, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    double,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<double>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    double,                                   // <- data type of accumulator
    double>;                                  // <- data type for alpha/beta in linear combination function



using Gemm_TCF64 = cutlass::gemm::device::Gemm<
                                              double,
                                              LayoutInputA,
                                              double,
                                              LayoutInputB,
                                              double,
                                              LayoutOutput,
                                              double,
                                              cutlass::arch::OpClassTensorOp,
                                              cutlass::arch::Sm80,
                                              cutlass::gemm::GemmShape<32, 32, 16>,
                                              cutlass::gemm::GemmShape<16, 16, 16>,
                                              cutlass::gemm::GemmShape<8, 8, 4>,
                                              EpilogueOp,
                                              SwizzleThreadBlock,
                                              4>;

void cutlass_run(
        const double *A, const double *B, 
        const double *C, double *D,
        int m, int n, int k,
        double alpha, double beta) {
   // TensorViewWrite(std::cout, tensor_a_TCF64.host_view()); 
   // TensorViewWrite(std::cout, tensor_b_TCF64.host_view()); 
   // TensorViewWrite(std::cout, tensor_c_TCF64.host_view()); 

    // Initialize alpha and beta for dot product computation
       // Split K dimension into 1 partitions
    int split_k_slices = 1;

  ////////////////////////////////////////////////////////////////////////////////
  /// 3. Run  3xTF32 kernel within a profiling loop
  ////////////////////////////////////////////////////////////////////////////////
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  
  typename Gemm_TCF64::Arguments arguments_tcf64{{m, n, k},  // <- problem size of matrix multiplication
      {A, k} ,  // <- reference to matrix A on device
      {B, k},
      {C, n},  // <- reference to matrix B on device
      {D, n},  // <- reference to matrix C on device
      {alpha, beta},          // <- tuple of alpha and beta
      split_k_slices};        // <- k-dimension split factor
    
    size_t workspace_size_tcf64 = Gemm_TCF64::get_workspace_size(arguments_tcf64);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace_tcf64(workspace_size_tcf64);

    // Instantiate CUTLASS kernel depending on templates
    Gemm_TCF64 gemm_tcf64;

    // Check the problem size is supported or not
    cutlass::Status status_tcf64 = gemm_tcf64.can_implement(arguments_tcf64);
    CUTLASS_CHECK(status_tcf64);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status_tcf64 = gemm_tcf64.initialize(arguments_tcf64, workspace_tcf64.get());
    CUTLASS_CHECK(status_tcf64);  

    //
    // Construct events
    //

  // Launch initialized CUTLASS kernel
    status_tcf64 = gemm_tcf64();
    CUTLASS_CHECK(status_tcf64);
     
}


//int main() {
//  
//    bool notSupported = false;
//
//    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
//        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
//        notSupported = true;
//    }
//
//    cudaDeviceProp props;
//
//    cudaError_t error = cudaGetDeviceProperties(&props, 0);
//    if (error != cudaSuccess) {
//        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
//        return false;
//    }
//
//    if (!((props.major * 10 + props.minor) >= 80)) {
//        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
//                  << std::endl;
//        notSupported = true;
//    }
//
//    if (notSupported) {
//        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
//        return 0;
//    }
//    int M = 64;
//    int N = 64;
//    int K = 64;
//
//    double *A = 0;
//    double *B = 0;
//    double *C = 0;
//    double *D = 0;
//
//    A = (double *)malloc(M * K * sizeof(double));
//    B = (double *)malloc(N * K * sizeof(double));
//    C = (double *)malloc(M * N * sizeof(double));
//    D = (double *)malloc(M * N * sizeof(double)); 
//    for (int i = 0; i < M * K; ++i)
//        A[i] = (double )1.;
//    for (int i = 0; i < N * K; ++i)
//        B[i] = (double )1.;
//    for (int i = 0; i < M * N; ++i)
//        C[i] = (double )0.;
//
//
//    double *d_A = 0;
//    double *d_B = 0;
//    double *d_C = 0;
//
//    cudaMalloc((void **) &d_A, sizeof(double) * M * K);
//    cudaMalloc((void **) &d_B, sizeof(double) * N * K);
//    cudaMalloc((void **) &d_C, sizeof(double) * M * N);
//
//    cudaMemcpy(d_A, A, sizeof(double) * M * K, cudaMemcpyHostToDevice); 
//    cudaMemcpy(d_B, B, sizeof(double) * K * N, cudaMemcpyHostToDevice); 
//    cudaMemcpy(d_C, C, sizeof(double) * M * N, cudaMemcpyHostToDevice);
//
//    bool result = true;
//    result = run(d_A, d_B, d_C, d_C, M, N, K);
// 
//    cudaMemcpy(C, d_C, sizeof(double) * M * N, cudaMemcpyDeviceToHost);
////    for (int i = 0; i < M; ++i)
////    {
////        for (int j = 0; j < N; ++j)
////            std::cout << C[j*M + i] << " ";
////    std::cout << std::endl;
////    }
//    if (!result) return -1;
//
//    return 0;
//}
}
