/*!
* \file    Utilities.cuh
* \brief   便利な関数郡です
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef UTILITIES_CUH_
#define UTILITIES_CUH_
#undef NDEBUG

#include<cuda_runtime.h>
#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include<vector>
#include"cublas_v2.h"

//![0, 1)の一様分布の乱数を出力する関数です。
float uniform();

//!平均mu、標準偏差sigmaの正規分布の乱数を生成する関数です
float gaussian(float mu, float sigma);

/*!
* \brief   GPU内で、データの列優先から行優先へデータ配置を換える関数
* \param[in]   d_data       移すデータ先のポインタ(デバイスポインタ)
* \param[in]   src_data		移すデータ元のポインタ(デバイスポインタ)
* \param[in]   rows		　　移すデータ元のrow
* \param[in]   cols		　 移すデータ元のcol
*/
__global__ void c2r_major_change2D(float* d_data, float* src_data, unsigned int rows, unsigned int cols);

/*!
* \brief   GPU内で、データの行優先から列優先へデータ配置を換える関数
* \param[in]   d_data       移すデータ先のポインタ(デバイスポインタ)
* \param[in]   src_data		移すデータ元のポインタ(デバイスポインタ)
* \param[in]   rows		　　移すデータ元のrow
* \param[in]   cols		　 移すデータ元のcol
*/
__global__ void r2c_major_change2D(float* d_data, float* src_data, unsigned int rows, unsigned int cols);

/*!
* \brief   二次元配列(列優先)の各列に一次元配列の値を加算します
* \param[in]	d_data		移すデータ先のポインタ(デバイスポインタ)
* \param[in]	rows		d_dataの行数
* \param[in]	cols		d_dataの列数
* \param[in]	src_data	足し合わせる列行列
*/
__global__ void plus_each_col2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data);

/*!
* \brief   二次元配列(列優先)の各行に一次元配列の値を加算します
* \param[in]	d_data		移すデータ先のポインタ(デバイスポインタ)
* \param[in]	rows		d_dataの行数
* \param[in]	cols		d_dataの列数
* \param[in]	src_data	足し合わせる行行列
*/
__global__ void plus_each_row2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data);

/*!
* \brief   二次元配列(列優先)の要素毎の掛け算(d_data[i][j] = d_data[i][j] ＊ src_data[i][j])
* \param[in]	d_data		移すデータ先のポインタ(デバイスポインタ)
* \param[in]	rows		d_dataの行数
* \param[in]	cols		d_dataの列数
* \param[in]	src_data	掛け合わせる
*/
__global__ void multiply2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data);

/*!
* \brief   src_dataからsubtruct_dataを引きます
* \param[in]	subtruct_data	引くためのMatrix
* \param[in]	src_data	元データMatrix
* \param[in]	rows	src_Matrixの行数
* \param[in]	cols	src_Matrixの列数
*/
__global__ void subtractArray2D(float *subtruct_data, float *src_data, unsigned int rows, unsigned int cols);

/*!
* \brief   src_dataからsubtruct_dataを引いたものをresult_dataへ移します
* \param[in]	subtruct_data	引くためのMatrix
* \param[in]	src_data	元データMatrix
* \param[in]	result_data	結果出力先
* \param[in]	rows	src_Matrixの行数
* \param[in]	cols	src_Matrixの列数
*/
__global__ void subtractArray2D(float *subtruct_data, float *src_data, float *result_data, unsigned int rows, unsigned int cols);

/*!
* \brief   1次元に変換された2次元情報に対して、Paddingを適用して畳み込み処理を行います。
* \param[in]	subtruct_data	引くためのMatrix
* \param[in]	src_data	元データMatrix
* \param[in]	result_data	結果出力先
* \param[in]	rows	src_Matrixの行数
* \param[in]	cols	src_Matrixの列数
*/
__global__ void Convolutuion2D_from1Darray_PadOn(float* Img_data,
	unsigned int Img_width, unsigned int Img_height, unsigned int Img_channel,
	float* filter_data,
	unsigned int Fil_width, unsigned int Fil_height, unsigned int Fil_channel,
	unsigned int stride,
	unsigned int Out_width, unsigned int Out_height
	);


/*!
* \brief   1次元に変換された2次元情報に対して、Paddingを適用して畳み込み処理を行います。
* \param[in]	subtruct_data	引くためのMatrix
* \param[in]	src_data	元データMatrix
* \param[in]	result_data	結果出力先
* \param[in]	rows	src_Matrixの行数
* \param[in]	cols	src_Matrixの列数
*/
__global__ void Convolutuion2D_from1Darray_PadOff(float* Img_data,
	unsigned int Img_width, unsigned int Img_height, unsigned int Img_channel,
	float* filter_data,
	unsigned int Fil_width, unsigned int Fil_height, unsigned int Fil_channel,
	unsigned int stride
	);


#endif
