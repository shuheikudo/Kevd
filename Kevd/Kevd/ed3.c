#include <math.h>
#include <omp.h>
#define MAX__(a,b) ((a)<(b)?(b):(a))
#define MIN__(a,b) ((a)<(b)?(a):(b))
static void dlacpy_c(char c, int n, int m, double* a, int lda, double* b, int ldb)
{
	extern void dlacpy_();
	dlacpy_(&c, &n, &m, a, &lda, b, &ldb);
}
static void dgemm_c(char t1, char t2, int n, int m, int k, double alpha, double* a, int lda,
	double* b, int ldb, double beta, double* c, int ldc)
{
	extern void dgemm_();
	dgemm_(&t1, &t2, &n, &m, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
static void dlaset_c(char c, int n, int m, double alpha, double beta, double* a, int lda)
{
	extern void dlaset_();
	dlaset_(&c, &n, &m, &alpha, &beta, a, &lda);
}
static int dlaed4_c(int n, int i, double* d, double* z, double* delta, double rho, double* dlam)
{
	extern void dlaed4_();
	int info;
	dlaed4_(&n, &i, d, z, delta, &rho, dlam, &info);
	return info;
}
static double dnrm2_c(int n, double* a, int inca)
{
	extern double dnrm2_();
	return dnrm2_(&n, a, &inca);
}

int ed3(int k, int n, int n1, double* d, double* q, int ldq, double rho, double* dlamda, 
	double* q2, int* indx, int* ctot, double* w, double* s)
{
	int info=0;
	// Test the input parameters.
	if(k < 0) return -1;
	if(n < k) return -2;
	if(ldq < MAX__(1,n)) return -6;
	if(k==0) return 0;

#ifdef NOGUARDDIGIT
	// for machine without guard digit
	extern double dlamc3_;
	for(int i=0; i<k; ++i) dlamda[i] = dlamc3_(dlamda+i, dlamda+i) - dlamda[i];
#endif

#pragma omp parallel reduction(|:info)
	{
		int nt = omp_get_num_threads();
		int ks = (k+nt-1)/nt;
		int id = omp_get_thread_num();
		int kb = ks*id;
		int ke = MIN__(kb+ks, k);
		for(int j=kb; j<ke; ++j)
			info = dlaed4_c(k, j+1, dlamda, w, q+ldq*j, rho, d+j);
#pragma omp barrier
		if(info==0 && k==2){
			for(int j=kb; j<ke; ++j){
				w[0] = q[0+ldq*j];
				w[1] = q[1+ldq*j];
				q[0+ldq*j] = w[indx[0]-1];
				q[1+ldq*j] = w[indx[1]-1];
			}
		}
		else if (info == 0 && k>2){
			double temp[k];
			for(int i=kb; i<ke; ++i) temp[i] = q[i+i*ldq];
			for(int j=0; j<k; ++j){
				for(int i=kb; i<ke; ++i)
					if(i!=j) temp[i] *= (q[i+j*ldq]/(dlamda[i]-dlamda[j]));
			}
			for(int i=kb; i<ke; ++i)
				w[i] = copysign(sqrt(-temp[i]), w[i]);
#pragma omp barrier
			for(int j=kb; j<ke; ++j){
				for(int i=0; i<k; ++i)
					temp[i] = w[i] / q[i+ldq*j];
				double t = dnrm2_c(k, temp, 1);
				for(int i=0; i<k; ++i)
					q[i+ldq*j] = temp[indx[i]-1]/t;
			}
		}

	}

	int n2 = n - n1;
	int n12 = ctot[0] + ctot[1];
	int n23 = ctot[1] + ctot[2];
	dlacpy_c('A', n23, k, q+ctot[0], ldq, s, n23);
	if(n23 != 0)
		dgemm_c('N', 'N', n2, k, n23, 1., q2+n1*n12, n2, s, n23, 0., q+n1, ldq);
	else
		dlaset_c('A', n2, k, 0., 0., q+n1, ldq);

	dlacpy_c('A', n12, k, q, ldq, s, n12);
	if(n12 != 0)
		dgemm_c('N', 'N', n1, k, n12, 1., q2, n1, s, n12, 0., q, ldq);
	else
		dlaset_c('A', n1, k, 0., 0., q, ldq);
	return 0;
}
