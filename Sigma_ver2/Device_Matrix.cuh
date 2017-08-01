/*!
* \file    Device_Matrix.cuh
* \brief   Sigma_Matrixに用いられているDevice_Matrixを定義しています。
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef DEVICE_MATRIX_CUH_
#define DEVICE_MATRIX_CUH_
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
#include<random>
#include<iomanip>
#include"curand.h"
#include"Sigma_global_functions.cuh"
#include"DEFINES.cuh"
#include"Host_Matrix.cuh"

using namespace std;

/*!
* \brief   GPU上での行列を扱うクラスになります
* \details GPU計算を行うために必要なGPU上でのメモリを確保しています。
*          全てのデータはfloat形の数値として扱われます。基本的にcol_majorです
*/
class Device_Matrix{
public:
	/*!
	* \brief   空の \c Device_Matrix オブジェクトを作成します。
	*/
	Device_Matrix();

	/*!
	* \brief   Device_Matrixを用いて\c Device_Matrixオブジェクトを作成します。(コピーコンストラクタ)
	* \prama[in]  obj  Device_Matrix
	*/
	Device_Matrix(const Device_Matrix& obj);

	/*!
	* \brief   指定した行/列数の \c Device_Matrix オブジェクトを作成し，
	*          同時にデータの初期化も行います。
	* \param[in]   rows        行数
	* \param[in]   cols        列数
	* \param[in]   *host_data	  データが格納されたfloat型ポインタ(CPU上の列優先のデータとする)
	*/
	Device_Matrix(unsigned int rows, unsigned int cols, float* host_data);

	/*!
	* \brief   指定した行/列数の \c Device_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	* \param[in]   rows        行数
	* \param[in]   cols        列数
	* \param[in]   init       初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	Device_Matrix(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	/*!
	* \brief   指定した行/列数の \c Device_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	* \param[in]   height	高さ
	* \param[in]   width	幅
	* \param[in]   channel	チャネル数
	* \param[in]   depth	深さ
	* \param[in]   init		初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	Device_Matrix(unsigned int height, unsigned int width = 1, unsigned int channel = 1, unsigned int depth = 1, INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	/*!
	* \brief   デストラクタ。デバイス内に格納されているデータを破棄します。
	*/
	~Device_Matrix(){ if (data_ != NULL) cudaFree(data_); };

	/*!
	* \brief   Device_Matrixを用いて\c Device_Matrixオブジェクトを初期化します。
	* \prama[in]  obj  Device_Matrix
	*/
	void initialize(Device_Matrix& obj);

	/*!
	* \brief   指定した行/列数の \c Device_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	* \param[in]   rows        行数
	* \param[in]   cols        列数
	* \param[in]   init       初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	void initialize(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	/*!
	* \brief   指定した行/列数の \c Device_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	* \param[in]   height	高さ
	* \param[in]   width	幅
	* \param[in]   channel	チャネル数
	* \param[in]   depth	深さ
	* \param[in]   init		初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	void initialize(unsigned int height, unsigned int width = 1, unsigned int channel = 1, unsigned int depth = 1,INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	///////////////////////////////		read / write	funciton/////////////////////////////////////////////

	/*!
	* \brief	ファイルを読み込んで \c Device_Matrix を初期化します
	* \prama[in]	file_name	読み込むファイル名
	*/
	void read_data(string file_name);

	/*!
	* \brief	データを書き込みます
	* \prama[in]	file_name	書き込むファイル名を書き込みます
	*/
	void write_data(string file_name);

	///////////////////////////////		setter	    ///////////////////////////////
	/*!
	* \brief   GPU上に格納されているデータを入力します
	* \param[in]	val	入力する値です
	* \prama[in]	height	高さです(データが二次元の場合はrowを表します)
	* \prama[in]	width	幅です(データが二次元の場合はcolを表します)
	* \prama[in]	channel	チャネルです
	* \prama[in]	depth	深さです
	*/
	inline void set(float val,unsigned int height, unsigned int width = 0, unsigned int channel = 0, unsigned int depth = 0){
		if (width < width_ && height < height_ && channel < channel_ && depth < depth_){
			unsigned int idx = depth * (width_ * height_ * channel_) + channel*(width_ * height_) + width * height_ + height;
			CHECK(cudaMemcpy(&data_[idx], &val,sizeof(float), cudaMemcpyHostToDevice));
		}
		else{
			cout << "error : Device Matrix : at : out of size" << endl;
			return;
		}
	}

	/*!
	* \brief   データのポインタを変更します
	* \param[in]	data	GPU上でのデータポインタ
	* \warning	必ずGPU上でのポインタを引数にしてください。サイズの変更も行いません
	*/
	void set_data(float* data){ data_ = data; };

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
	* \warning	返されるポインタはデバイス上でのポインタです。data[i]などを書くとエラーをはきます。
	*/
	inline  float* data(){ return data_; }

	/*!
	* \brief   データのサイズをsize_t型で取得します
	* \return  格納しているデータのサイズ
	*/
	inline size_t size(){ return size_; };
	
	/*!
	* \brief   GPU上に格納されているデータを取得します
	* \prama[in]	height	高さです(データが二次元の場合はrowを表します)
	* \prama[in]	width	幅です(データが二次元の場合はcolを表します)
	* \prama[in]	channel	チャネルです
	* \prama[in]	depth	深さです
	*/
	inline float at(unsigned int height, unsigned int width = 0, unsigned int channel = 0, unsigned int depth = 0){
		if (width < width_ && height < height_ && channel < channel_ && depth < depth_){
			unsigned int idx = depth * (width_ * height_ * channel_) + channel*(width_ * height_) + width * height_ + height;
			float val = 0;
			CHECK(cudaMemcpy(&val, &data_[idx], sizeof(float), cudaMemcpyDeviceToHost));
			return val;
		}
		else{
			cout << "error : Device Matrix : at : out of size" << endl;
			return -1;
		}
	};

	/////////////////////////////	get_Matrix	////////////////////
	
	Host_Matrix& get_Matrix();

	////////////////////////////	operator	///////////////////
	/*!
	* \brief	代入演算子です
	*/
	Device_Matrix& operator = (const Device_Matrix& obj);
	
	//!未実装です
	Device_Matrix& operator * (const Device_Matrix& obj);
	//!未実装です
	Device_Matrix& operator + (const Device_Matrix& obj);
	//!未実装です
	Device_Matrix& operator - (const Device_Matrix& obj);

	/////////////////////////////    print /////////////////////////
	/*!
	* \brief   \c Device_Matrix の全てのデータを出力します
	* \prama[in]	All_status	trueにすることで、データも全て表示します
	*/
	void print(bool All_status = false);

private:
	//!CPU上にデータを転送する際に用いる \c Host_Matrix です
	//Host_Matrix host_matrix_;

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

	//!データ配列(GPU上にメモリを確保します)
	float* data_;

	//!サイズです
	size_t size_;
};

#endif