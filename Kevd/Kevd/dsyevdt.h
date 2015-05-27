// Copyright by Shuhei Kudo, May 2015.
#ifndef DSYEVDT_H
#define DSYEVDT_H
#ifdef __cplusplus
extern "C" {
#endif}
	void dsyevdeasy_(int* n, double* a, int* lda, double* w);
	void dsyevdt_(const char* jobz, const char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int * info);
	int dsyevdt(char jobz, char uplo, int n, double* a, int lda, double* w, double* work, int lwork, int* iwork, int liwork);
	void dsyevdt_worksizes(char jobz, char uplo, int n, int num_threads, int* lwork, int* liwork);
#ifdef __cplusplus
}
#endif
#endif
