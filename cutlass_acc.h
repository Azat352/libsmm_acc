#ifndef CUTLASS_ACC_H
#define CUTLASS_ACC_H
extern "C"
void cutlass_run(
        const double *A, const double *B, 
        const double *C, double *D, 
        int m, int n, int k, double alpha, 
        double beta);



#endif
