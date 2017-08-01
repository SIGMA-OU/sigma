/*!
* \file    Sigma_global_functions.cuh
* \brief   Sigmaで用いるGlobal関数群です
* \date    2016/06/18
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef SIGMA_GLOBAL_FUNCTIONS_CUH_
#define SIGMA_GLOBAL_FUNCTIONS_CUH_
#undef NDEBUG

#include<cuda_runtime.h>
#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include"cublas_v2.h"
#include"curand.h"
#include"curand_kernel.h"

/*!
* \brief   二次元配列dataを値valに設定します。
* \param[in]	data	配列のポインタ
* \param[in]	rows	行数
* \param[in]	cals	列数
* \param[in]	val	値
*/
__global__ void set_array_2D(float* data,unsigned int rows, unsigned int cols,float val);

/*!
* \brief   二次元配列dataを単位行列に初期化します
* \param[in]	data	配列のポインタ
* \param[in]	rows	行数
* \param[in]	cals	列数
*/
__global__ void set_array_identity_2D(float* data, unsigned int rows, unsigned int cols);

/*!
* \brief   二次元配列dataをstepに初期化します
* \param[in]	data	配列のポインタ
* \param[in]	rows	行数
* \param[in]	cals	列数
*/
__global__ void set_array_step_2D(float* data, unsigned int rows, unsigned int cols);

/*!
* \brief   二次元配列dataの全ての要素にvalを加算します
* \param[in]	data	配列のポインタ
* \param[in]	rows	行数
* \param[in]	cals	列数
* \param[in]	val	加算する値
*/
__global__ void sum_constant_2D(float*data,unsigned int rows, unsigned int cols,float val);

/*!
* \brief   二次元配列を一様分布で初期化します。
* \param[in]	data	配列のポインタ
* \param[in]	rows	行数
* \param[in]	cals	列数
* \param[in]	x	下限値ｘ
* \param[in]	y	上限値y
*/
__global__ void initialize_uniform2D(float *data, unsigned int rows, unsigned int cols, float x, float y,curandState_t *state);

/*!
* \brief   二次元配列を正規分布で初期化します
* \param[in]	data	配列のポインタ
* \param[in]	rows	行数
* \param[in]	cals	列数
* \param[in]	mean	平均
* \param[in]	stddev	分散
*/
__global__ void initialize_normal2D(float *data, unsigned int rows, unsigned int cols, float mean, float stddev,curandState_t *state);

/*!
* \brief   乱数を生成するために、curandState_tを初期化します
* \param[in]	seed	乱数のseed値(ランダムな値を入れてください)
* \param[in]	rows	行数
* \param[in]	cals	列数
* \param[in]	*state	curandState_t
*/
__global__ void rand_init2D(unsigned long long seed,unsigned int rows, unsigned int cols, curandState_t *states);

/*!
* \brief	1次元配列のBlock処理を行います。\n(data[data_begin] -> src_data[0].....data[data_begin + src_data_num] -> src_data[src_data_num])
* \param[in]	data	データを出力する配列
* \param[in]	data_begin	転送する配列先の先頭要素
* \param[in]	src_data	転送するデータ配列
* \param[in]	src_data_num	転送するデータ配列の配列数
*/
__global__ void block1D(float* data, unsigned int data_begin, float* src_data, unsigned int src_data_num);

/*!
* \brief	二次元画像のデータを畳み込み処理しやすい形に配置を変えます.
* \param[in]	result_data	転送先のデータ配列
* \param[in]	src_data	二次元配列
* \param[in]	Image_width	二次元配列の幅
* \param[in]	Image_height	二次元配列の高さ
* \param[in]	Image_channle	二次元配列のチャネル数
* \param[in]	filter_width	フィルタの幅
* \param[in]	filter_height	フィルタの高さ
* \param[in]	stride	ストライド幅
* \param[in]	padding	パディングするかどうか(bool型です)
* \param[in]	result_rows	出力先行列の行数
* \param[in]	result_cols	出力先行列の列数
* \warning	起動するblock,gridはresult_dataのデータ用素数です。間違わないように!!
*/
__global__ void replace_array_for_CONV2D(float* result_data,float* src_data,
										unsigned int Image_width,unsigned int Image_height,unsigned Image_channel,
										unsigned int filter_width,unsigned int filter_height,unsigned int stride,
										bool padding,unsigned int result_rows, unsigned int result_cols);

/*!
* \brief	二次元畳み込みのデルタの逆伝播の際に使用します。
* \param[in]	prev_delta	前の層のデルタ行列
* \param[in]	src_data	replace_array
* \param[in]	Image_width	二次元配列の幅
* \param[in]	Image_height	二次元配列の高さ
* \param[in]	Image_channle	二次元配列のチャネル数
* \param[in]	filter_width	フィルタの幅
* \param[in]	filter_height	フィルタの高さ
* \param[in]	stride	ストライド幅
* \param[in]	padding	パディングするかどうか(bool型です)
* \param[in]	prev_rows	prev_deltaの行数
* \param[in]	prev_cols	prev_deltaの列数
* \param[in]	src_rows	src_dataの行数
* \param[in]	src_cols	src_dataの列数 
* warninig	起動するblock,gridはprev_deltaの各データの要素数です
*/
__global__ void replace_array_for_backward_CONV2D(float* prev_delta, float* src_data,
										unsigned int Image_width, unsigned int Image_height, unsigned Image_channel,
										unsigned int filter_width, unsigned int filter_height, unsigned int stride,
										bool padding, unsigned int prev_rows, unsigned int prev_cols,
										unsigned int src_rows,unsigned int src_cols);

/*!
* \brief	二次元マックスプーリングを行います
* \param[in]	output_data	転送先のデータ配列
* \param[in]	src_data	二次元配列
* \param[in]	idx_matrix	インデックスの情報が格納される配列です
* \param[in]	Image_width	二次元配列の幅
* \param[in]	Image_height	二次元配列の高さ
* \param[in]	Image_channle	二次元配列のチャネル数
* \param[in]	outpout_width	出力先のwidth
* \param[in]	output_height	出力先のheight
* \param[in]	pooling_width	フィルタの幅
* \param[in]	pooling_height	フィルタの高さ
* \param[in]	stride	ストライド幅
* \param[in]	padding	パディングするかどうか(bool型です)
* \param[in]	output_rows	出力先行列の行数
* \param[in]	output_cols	出力先行列の列数
* \warning	起動するblock,gridはresult_dataのデータ用素数です。間違わないように!!
*/
__global__ void max_Pooling2D(float* output_data, float* src_data,int* idx_matrix,
										unsigned int Image_width,unsigned int Image_height,unsigned int Image_channel,
										unsigned int output_width, unsigned int output_height,
										unsigned int pooling_width,unsigned int pooling_height,unsigned int stride,
										bool padding, unsigned int output_rows,unsigned int output_cols);

/*!
* \brief	二次元アベレージプーリングを行います
* \param[in]	output_data	転送先のデータ配列
* \param[in]	src_data	二次元配列
* \param[in]	Image_width	二次元配列の幅
* \param[in]	Image_height	二次元配列の高さ
* \param[in]	Image_channle	二次元配列のチャネル数
* \param[in]	outpout_width	出力先のwidth
* \param[in]	output_height	出力先のheight
* \param[in]	pooling_width	フィルタの幅
* \param[in]	pooling_height	フィルタの高さ
* \param[in]	stride	ストライド幅
* \param[in]	padding	パディングするかどうか(bool型です)
* \param[in]	output_rows	出力先行列の行数
* \param[in]	output_cols	出力先行列の列数
* \warning	起動するblock,gridはresult_dataのデータ用素数です。間違わないように!!
*/
__global__ void average_Pooling2D(float* output_data, float* src_data,
										unsigned int Image_width, unsigned int Image_height, unsigned int Image_channel,
										unsigned int output_width,unsigned int output_height,
										unsigned int pooling_width, unsigned int pooling_height, unsigned int stride,
										bool padding, unsigned int output_rows, unsigned int output_cols);

/*!
* \brief	二次元のマックスプーリングしたdeltaを逆伝播します
* \param[in]	prev_delta	前の層のデルタ
* \param[in]	src_data	自層のデルタ
* \param[in]	idx_matirx	インデックス情報が格納された配列
* \param[in]	prev_delta_width	前層のデルタの幅
* \param[in]	prev_delta_height	前層のデルタの高さ
* \param[in]	prev_delta_channle	前層のデルタのチャネル数
* \param[in]	src_width	自層のwidth
* \param[in]	src_height	自層のheight
* \param[in]	pooling_width	プーリングの幅
* \param[in]	pooling_height	プーリングの高さ
* \param[in]	stride	ストライド幅
* \param[in]	padding	パディングするかどうか(bool型です)
* \param[in]	prev_delta_rows	前の層のデルタの行数
* \param[in]	prev_delta_cols	前の層のデルタの列数
* \warning	起動するblock,gridはprev_delta_dataのデータ用素数です。間違わないように!!
*/
__global__ void backward_for_max_Pooling2D(float* prev_delta, float* src_data,int* idx_matrix,
						unsigned int prev_delta_width, unsigned int prev_delta_height, unsigned int prev_delta_channel,
						unsigned int src_width,unsigned int src_height,
						unsigned int pooling_width,unsigned int pooling_height,unsigned int stride,
						bool padding,unsigned int prev_delta_rows,unsigned int prev_delta_cols);

/*!
* \brief	二次元のマックスプーリングしたdeltaを逆伝播します
* \param[in]	prev_delta	前の層のデルタ
* \param[in]	src_data	自層のデルタ
* \param[in]	prev_delta_width	前層のデルタの幅
* \param[in]	prev_delta_height	前層のデルタの高さ
* \param[in]	prev_delta_channle	前層のデルタのチャネル数
* \param[in]	src_width	自層のwidth
* \param[in]	src_height	自層のheight
* \param[in]	pooling_width	プーリングの幅
* \param[in]	pooling_height	プーリングの高さ
* \param[in]	stride	ストライド幅
* \param[in]	padding	パディングするかどうか(bool型です)
* \param[in]	prev_delta_rows	前の層のデルタの行数
* \param[in]	prev_delta_cols	前の層のデルタの列数
* \warning	起動するblock,gridはprev_delta_dataのデータ用素数です。間違わないように!!
*/
__global__ void backward_for_average_Pooling2D(float* prev_delta, float* src_data,
						unsigned int prev_delta_width, unsigned int prev_delta_height, unsigned int prev_delta_channel,
						unsigned int src_width, unsigned int src_height,
						unsigned int pooling_width, unsigned int pooling_height, unsigned int stride,
						bool padding, unsigned int prev_delta_rows, unsigned int prev_delta_cols);

#endif