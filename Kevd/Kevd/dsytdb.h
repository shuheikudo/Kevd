#ifndef DSYTDB_H
#define DSYTDB_H
int dsytdb(int n, double* a, int lda, double* d, double* e, double* v, double* work);
int dsytdc(int n, double* a, int lda, double* d, double* e, double* tau, double* work);
int dsytdd(int n, double* a, int lda, double* d, double* e, double* tau, double* work);
int dsytdb_worksize(int n, int num_threads);
int dsytdd_worksize(int n, int num_threads);
#endif