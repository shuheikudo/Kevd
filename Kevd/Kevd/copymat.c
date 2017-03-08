void copymat(int m, int n, const double* a, int lda, double* b, int ldb)
{
#pragma omp parallel for
	for(int i=0; i<n; ++i)
		for(int j=0; j<m; ++j)
			b[j+i*ldb] = a[j+i*lda];
}
