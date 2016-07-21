#include <boost/preprocessor.hpp>
#include <cstdlib>
#include <cstdio>
#include <cstring>

// test
#define BL 4
#define RJ 2

// simd intrinsic emulation.
struct vd_t {
	double data[RJ];
};

vd_t vload(const double* from)
{
	vd_t temp;
	for(int i=0; i<RJ; ++i) temp.data[i] = from[i];
	return temp;
}

void vstore(double* to, vd_t vd)
{
	for(int i=0; i<RJ; ++i) to[i] = vd.data[i];
}

vd_t vzero()
{
	vd_t temp;
	for(int i=0; i<RJ; ++i) temp.data[i] = 0;
	return temp;
}


// Fortran配列からタイルレイアウトへの変換
// タイルレイアウトは
// 1) RJ行だけ縦に
// 2) BL列だけ横に
// 3) 対角ブロックまで縦に
// という順でブロック列を格納し、
// 4) 次の列ブロックまでジャンプ
// することで、スレッドが担当するデータがメモリ上で連続するように並べる。
// 以下にRJ=2, BL=3, nth=3, n=11の例
// A = a b c|z y x|A B C|d e *
//       f g|w v u|D E F|h i *
//         j|t s r|G H I|k l *
//          |. . .|. . .|. . *
//          |  . .|. . .|. . *
//          以下略
// bb[0]= abbfcgczfgjtdhei00k.l.0.
//         ^1         ^2  ^3
//
//      ^1: 対称位置の要素
//      ^2: 次のブロック列までジャンプ
//      ^3: 端数列はゼロうめ
//
// アドレッシングが複雑なので、次のデータ並び替えのコードを参照のこと


#if (RJ <= BL) && (BL%RJ) == 0
#	define TILE_SCHEME 0 // easy case
#else
#	define TILE_SCHEME 2 // complex case
#endif

#define CNV_MOVE_A2B(z, n, ignore) vstore(b + RJ*(n), vload(a + lda*(n)));
#define CNV_MOVE_A2B_CHECK(z, n, ignore) vstore(b + RJ*(n), ((n)<isz ? vload(a + lda*(n)): vzero()) );
#define SAFE_INDEX(i,j, lda) ( (j)<(i) ? (i)*lda+(j) : (j)*lda+(i) )
#define CNV_MOVE_A2B_SERIAL_IMPL(z, m, n) b[(n)*RJ+(m)] = aa[SAFE_INDEX(i+(n),j+(m),lda)];
#define CNV_MOVE_A2B_SERIAL(z, n, ignore) BOOST_PP_REPEAT(RJ, CNV_MOVE_A2B_SERIAL_IMPL, n)
size_t address_tile_buf(double** bb, double* b, int n, int nth)
{
	bb[0] = b;
	int nbl = (n+BL-1)/BL;
	size_t t=0;
	for(int id=0; id<nth; ++id){
#if TILE_SCHEME == 0
		int mynbl = (nbl-id+nth-1) / nth;
		int sbl = mynbl*(2*id+2+nth*(mynbl-1))/2;
		t += BL*BL*sbl;
#else
		// 複雑なのでリニアタイムのアルゴリズムを使う。
		for(int I=id, i=id*BL; i<n; I+=nth, i+=nth*BL){
			t += (i+BL+RJ-1)/RJ*RJ*BL;
		}
#endif
		if(id<nth-1) bb[id+1] = b + t;
	}
	return t;
}

// データ並び替えのルーチン兼、データ順序の定義
// Fortran配列の行列aaを配列bb[id]へ移動する。
void convert_to_tile(int n, double* aa, int lda, double** bb, int id, int nth)
{
	double* b = bb[id];
	for(int i=id*BL; i<n; i+=nth*BL){
		double* a = aa + i * lda;
		int j = 0;
		int isz = n -i;
		if(isz >= BL){
			// simd copy
			for(; j+RJ < i; j+= RJ){
				BOOST_PP_REPEAT(BL, CNV_MOVE_A2B, ignore);
				b += RJ*BL;
				a += RJ;
			}
			// serial copy
			for(; j<i+BL; j+=RJ){
				BOOST_PP_REPEAT(BL, CNV_MOVE_A2B_SERIAL, ignore);
				b += RJ*BL;
			}
		}
		else {
			// simd copy with bounds-check
			for(; j+RJ < i; j+= RJ){
				BOOST_PP_REPEAT(BL, CNV_MOVE_A2B_CHECK, ignore);
				b += RJ*BL;
				a += RJ;
			}
			// naiive copy
			for(; j<i+BL; j+=RJ){
				int jjsz = (RJ<n-j?RJ:n-j);
				for(int ii=0; ii<isz; ++ii)
					for(int jj=0; jj<jjsz; ++jj)
						b[ii*RJ+jj] = aa[SAFE_INDEX(i+ii,j+jj,lda)];
				b += RJ*BL;
			}
		}
	}
}

void getcol(int c, double* x, double** bb, int nth)
{
	int ii = c/BL;
	int r = c%BL;
	int id = ii%nth;
#if TILE_SCHEME == 0
	int mynbl = (ii-id+nth-1)/nth;
	size_t t = mynbl*((2*id+2+nth*(mynbl-1))/2)*BL*BL;
#else
	size_t t = 0;
	for(int I=id, i=id*BL; I<ii; I+=nth, i+=nth*BL)
		t += (i+BL+RJ-1)/RJ*RJ*BL;
#endif
	double* b = bb[id] + t + r*RJ;
	for(int j=0; j<=c; j+=RJ){
		vstore(x, vload(b));
		x += RJ;
		b += RJ*BL;
	}
}


// MV, R2k の template
void NAME()
{
	double* b = bb[id];
	// load_odd_a
	for(int i=id*BL; i<n; i+=nth*BL){
		int j = 0;
		int isz = n -i;
		if(isz >= BL){
			// load U_[x,u,v], w
			for(; j < i; j+= 2*RJ){
				//BOOST_PP_REPEAT(BL, CNV_MOVE_A2B, ignore);
				//b += RJ*BL;
				//load_even_a_*
				// mvr2k for odd_a
				// store odd_a
				//load_odd_a_*
				// mvr2k for even_a
				// store_even_a
			}
			if(i-j==RJ){
				// mvr2k for odd_a
				// store_odd_a
				// load_odd_a !
			}
#if TILE_SCHEME != 0
			else {
				//
			}
#endif
			// diagonal block
#if TILE_SCHEME == 0
			BOOST_PP_REPEAT(DIV(BL,RL), DIAGONAL_KERNEL, ignore);
#else
			// very complicated...
#endif

		}
		else {
			for(; j+RJ < i; j+= RJ){
				BOOST_PP_REPEAT(BL, CNV_MOVE_A2B_CHECK, ignore);
				b += RJ*BL;
			}
		}
	}
}



int main()
{
	int n = 23;
	int lda = 30;
	int nth = 3;
	double* a = (double*)malloc(sizeof(double)*lda*n);
	double* a2 = (double*)malloc(sizeof(double)*lda*n);
	double** bb = (double**)malloc(sizeof(double*)*nth);
	double* b = 0;
	size_t bsz = address_tile_buf(bb, b, n, nth);
	b = (double*)malloc(sizeof(double)*(bsz));
	address_tile_buf(bb, b, n, nth);

	for(int i=0; i<n; ++i)
		for(int j=0; j<=i; ++j) a[i*lda+j] = (int)((double)std::rand()/RAND_MAX*100);
	std::printf("from:\n");
	for(int i=0; i<n; ++i){
		std::printf("   ");
		for(int j=0; j<=i; ++j) std::printf(" %3d", (int)a[i*lda+j]);
		std::printf("\n");
	}
	for(int id=0; id<nth; ++id)
		convert_to_tile(n, a, lda, bb, id, nth);
	for(int i=0; i<n; ++i)
		getcol(i, a2+i*lda, bb, nth);
	std::printf("\n\nto:\n");
	for(int i=0; i<n; ++i){
		std::printf("   ");
		for(int j=0; j<=i; ++j) std::printf(" %3d", (int)a2[i*lda+j]);
		std::printf("\n");
	}
	free(b); free(bb); free(a2); free(a);
	return 0;
}


#if 0
// MVの実装
// 行列Aは縦2*SIMD幅、横BLのタイルで分割された対称行列
// 対角タイルは対称な要素を重複して格納する。

int id, nt;
for(int i=id; i<n/BL; i+=id){
	//
	a_o[BL];
	for(int t=0; t<BL; ++t) a_o[t] = load(A);
	a_e[BL];
	for(int j=0; j <= i; j+= 2*RJ){
		y[j] += a
	}
}
#endif