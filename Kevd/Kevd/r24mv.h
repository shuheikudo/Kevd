#ifndef R24MV_H
#define R24MV_H
// merge x, u, v to t
void rotate_uvb(int n, double* x, double* uf, double* vf, int lda, int ldv, double* t);
void rotate_don(int n, double* uf, double* vf, int lda, int ldv, double* to);
void rotate_uvb3(int n, double* x, double* uf, double* vf, int lda, int ldv, double* t);
void rotate_uvb2(int n, double* x, double* uf, double* vf, int lda, int ldv, double* t);
void rotate_uvb1(int n, double* x, double* uf, double* vf, double* t);
void rotate_uvb11(int n, double* x, double* uf, double* vf, double* t);

void reorder(int n, const double* a, int lda, double* b);
void getcol(int i, const double* a, double* b);
void getcol1(int i, const double* a, double* b);
void getcol4(int id, int i, const double* a, double* b4, double* b3, double* b2, double* b1);

void dsymvb(int n, double* a, const double* x, double* y);
void dsyr2_mvb(int n, double* a, const double* u, const double* v, const double* x, double* y);
void dsyr24_mvb(int n, double* a, const double* xuv, double* y);
void dsyr23_mvb(int n, double* a, const double* xuv, double* y);
void dsyr22_mvb(int n, double* a, const double* xuv, double* y);
void dsyr21_mvb(int n, double* a, const double* xuv, double* y);
void dsyr24b(int n, double* a, const double* xuv, double* y);
void dsyr24_mvnb(int n, double* a, int lda, const double* xuv, double* y);

#endif