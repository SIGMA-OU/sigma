/*!
* \file    Host_Matrix.cuh
* \brief   Sigma_Matrixに用いられているHost_Matrixを定義しています。
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef HOST_MATRIX_CUH_
#define HOST_MATRIX_CUH_
#undef NDEBUG

#include<cuda_runtime.h>
#include<cassert>
#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include<string>
#include<fstream>
#include"cublas_v2.h"
#include"DEFINES.cuh"
#include"Utilities.cuh"


using namespace std;
/*!
* \brief   CPU上での行列を扱うクラスになります
* \details GPU計算を行うために必要なCPU上でのメモリを確保しています。
*          全てのデータはfloat形の数値として扱われます。基本的にcol_majorです
*/
class Host_Matrix{
public:
	/*!
	* \brief   空の \c Host_Matrix オブジェクトを作成します。
	*/
	Host_Matrix(){};

	/*!
	* \brief   指定した行/列数の \c Host_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	*
	* \param[in]   rows        行数
	* \param[in]   cols        列数
	* \param[in]   init       初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	Host_Matrix(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	/*!
	* \brief   デストラクタ。メモリに格納されているデータを破棄します。
	*/
	~Host_Matrix(){ if (data_ != NULL) free(data_); };

	/*!
	* \brief   コピーコンストラクタ.行列のデータを含めてコピーします
	*/
	Host_Matrix(const Host_Matrix& obj);

	////////////////////////////////	getter		////////////////////////////////
	/*!
	* \brief   幅widht_を取得します
	* \return  幅
	*/
	inline unsigned int width(){ return width_; };

	/*!
	* \brief   高さheight_を取得します
	* \return  高さ
	*/
	inline unsigned int height(){ return height_; };

	/*!
	* \brief   チャネルchannel_を取得します
	* \return  チャネル数
	*/
	inline unsigned int channel(){ return channel_; };

	/*!
	* \brief   深さdepth_を取得します(３D情報の時に使います)
	* \return  深さ
	*/
	inline unsigned int depth(){ return depth_; };

	/*!
	* \brief   行数rows_を取得します
	* \return  行数
	*/
	inline unsigned int rows() { return rows_; };

	/*!
	* \brief   列数cols_を取得します
	* \return  列数
	*/
	inline unsigned int cols(){ return cols_; };

	/*!
	* \brief   格納しているデータのポインタを取得します。
	* \return  格納しているデータのポインタ
	* \warning	返されるポインタはデバイス上でのポインタです。
	*/
	inline  float* data(){ return data_; }

	/*!
	* \brief   データのサイズをsize_t型で取得します
	* \return  格納しているデータのサイズ
	*/
	inline size_t size(){ return size_; };

	/*!
	* \brief   CPU上に格納されているデータを取得します
	* \prama[in]	height	高さです(データが二次元の場合はrowを表します)
	* \prama[in]	width	幅です(データが二次元の場合はcolを表します)
	* \prama[in]	channel	チャネルです
	* \prama[in]	depth	深さです
	*/
	inline float at(unsigned int height, unsigned int width = 0, unsigned int channel = 0, unsigned int depth = 0){
		if (data_ == NULL){
			cout << "Error : Host_matrix : No data" << endl;
		}
		else if (width < width_ && height < height_ && channel < channel_ && depth < depth_){
			unsigned int idx = depth * (width_ * height_ * channel_) + channel*(width_ * height_) + width * height_ + height;
			return data_[idx];
		}
		else{
			cout << "error : Device Matrix : at : out of size" << endl;
			return -1;
		}
	};

	////////////////////////////	operator	///////////////////

	Host_Matrix& operator = (const Host_Matrix& obj);

	Host_Matrix& operator * (const Host_Matrix& obj);
	Host_Matrix& operator + (const Host_Matrix& obj);
	Host_Matrix& operator - (const Host_Matrix& obj);

	///////////////////////////////  print function    //////////////////////////////////////
	/*!
	* \brief   \c Host_Matrix の全てのデータを出力します
	* \prama[in]	All_status	trueにすることで、データも全て表示します
	*/
	void print(bool All_status = false);

private:
	//!widthです
	unsigned int width_;

	//!heightです
	unsigned int height_;

	//!channelです
	unsigned int channel_;

	//!depthです
	unsigned int depth_;

	//!行数です
	unsigned int rows_;

	//!列数です
	unsigned int cols_;

	//!データ配列(CPU上にメモリを確保します)
	float* data_;

	//!サイズです
	size_t size_;
};

#endif