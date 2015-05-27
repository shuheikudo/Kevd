#ifndef R24MV_H
#define R24MV_H
// merge x, u, v to t
void rotate_uvb(int n, double* x, double* uf, double* vf, int lda, int ldv, double* t);
void rotate_uvb1(int n, double* x, double* uf, double* vf, double* t);
void rotate_uvb11(int n, double* x, double* uf, double* vf, double* t);

void reorder(int n, const double* a, int lda, double* b);
void getcol(int i, const double* a, double* b);
void getcol1(int i, const double* a, double* b);

void dsymvb(int n, double* a, const double* x, double* y);
void dsyr24_mvb(int n, double* a, const double* xuv, double* y);
void dsyr21_mvb(int n, double* a, const double* xuv, double* y);

#endif