#Kevd
Sparc64VIIIfx 用に最適化されたLAPACKのDSYEVDです。
小さな行列に対しても高い性能がでるよう、データレイアウトを工夫するなどを行っています。
基本的に、コードの読みやすさはまったく考慮せず、性能のみを優先したものとなっているため、
コード読解は苦痛なものとなることが予想されます。

LAPACKのコードを参考に作ったものが多いですが、その後、一から書き直したため、
LAPACKのソースコードが残っているものは少ないです。

Sparc64VIIIfxでは、$n \le 4000$ の範囲ではLAPACKの２倍以上の性能が出ます。
とくに、 $n=1000$ では約３倍、$n=100$ では約４倍の性能となっています。
$n > 4000$ での性能検証を行っていませんが、おそらく$n \le 6000$ 程度まではLAPACKよりも高性能
ではないかと予測しています。
それ以上の行列サイズでは、LAPACKを用いるか、十分な行列サイズですので、
分散並列の実装(ScaLAPACK, EigenK, ELPA)を用いることが適切です。
FX-10上では、$n=4000$の場合、36CPUを用いたScaLAPACKの方が高速です。

## 使い方
おおむね、基にしたLAPACKコードから機能削減したものとなっているため、対応するLAPACKのマニュアルを参照すると、
動作がわかるはずです。とくに、入力・出力データの形式には互換性があります。


## ドライバールーチン
dsyevdt_, dsyevdt, dsyevdeasy_ はドライバーインターフェースです。基本的にこれを使ってください。
これらの制約として、jobz="V", uplo="U"でしか動きません。また、どのCPUでも16byteアライメントを要求します。

## 三重対角化
dsytdc は三重対角化の実装となっています。おそらく唯一のDongarra-Wilkinson法^0の実装です。
オリジナルのコードはLAPACKのdsytd.fですので、FORTRANな人はそちらを参照するとコード読解の参考になると思います。
ただし、一から書き直しているため、元のコードに対応するものは一切残っていません。

プログラムの構造として、データレイアウトを抽象化しており、データレイアウト依存の処理はr24mv.cにある
BLASレベルの関数に追い出しています。そのため、dsytdcは教科書的なアルゴリズムに近く、
十分な最適化を行えていません。

r24mv.c は三重対角化で用いるBLASレベルの関数を実装したものです。
データレイアウトを変換することで、データアクセスを単純化し、内部ループでのアドレッシング計算も簡単となり、
メモリバンド幅を引き出せるようになりますが、プログラムの見た目はまったく直観に反するものとなります。
SSE2相当のSIMDをIntrinsicで使っているため、
古いCPUでは動かず、新しいCPUでは性能が出ないかもしれません。
また、Sparc64VIIIfxの持つ多数のSIMDレジスタを活用することを前提としているため、
ふつうのCPUだとレジスタスピルが発生してしまいます。

## 三重対角行列の固有対計算・分割統治法
この部分は、基本的には、LAPACKのコードにompのプラグマを１行追加したものとなっています。
C言語に変換したコードもありますが、性能上での利点はありません。

## 三重対角化の逆変換
外部発表はしていますが、まだ論文にしていないアルゴリズムが含まれています。

逆変換はcompact-WY表現を用いることで、簡単にLevel-3 BLASを用いることができる計算^1ですが、
行列サイズが小さい場合はBLASのオーバーヘッドが大きいため、性能がでません。
そのため、BLASを使うのではなく、これ自体をBLASレベルで実装することを方針にしました。

LAPACKが採用しているアルゴリズムとは、ループ順序から異なり、まったく独立に開発したコードとなっています。

キャッシュ上にパッキングした列ブロックを保存するため、行列サイズが一定以上となると、
性能劣化します。Sparc64VIIIfxだと、$n>8000$程度でこの問題が発生すると計算されますので、
このサイズとなった場合はLAPACKのものを用いてください。

主要なコードがマクロで実装されており、またデータレイアウト変換をしているため、
大変わかりにくいコードとなっています。（行列へのデータアクセスの位置がわかりますか？）

## TODO
*dsytdc: 行列サイズが小さい場合向けの、$n$回の同期へ変更
*dormtrb: レジスタ数を節約
*AVX, AVS512対応
*GPU? CUDA?
*Francis QRのLevel-2実装

## Changelog
2016/5/21:
MinGWへの対応
MinGWでOpenMPとBLAS, LAPACKを使うにあたって、多くのdllが必要になります。
libblas.dll, liblapack.dllを手に入れて、また、MinGWのbinディレクトリにあるdllを、実行ファイルのある
ディレクトリにコピーしてください。


C99
allocaを可変長配列に置き換えました。他にC99にしかないヘッダーファイルを読み込んでいます。

## 参考文献
*^0 S.Kudo et.al.: ``三重対角化に対するDongarra-Wilkinson法の実装と性能解析'', HPCS2015予稿集 (2015). (2017年まで有料)
*^1 G.H.Golub et.al.: ``Matrix Computations,'' J.H.Univ.Press (1996).
