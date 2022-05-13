#ifndef WMMA_ACC_H
#define WMMA_ACC_H
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

extern "C" __global__ void simple_wmma_gemm(const double *a, const double *b, double *c, double *d, 
        int m_ld, int n_ld, int k_ld, 
        double alpha, double beta);
#endif
