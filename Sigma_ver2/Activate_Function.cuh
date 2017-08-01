/*!
* \file    Activate_Function.cuh
* \brief   使用する活性化関数を定義しています。
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef ACTIVATE_FUNCTION_CUH_
#define ACTIVATE_FUNCTION_CUH_
#undef NDEBUG

#include<cuda_runtime.h>
#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include"DEFINES.cuh"
#include"Sigma_Matrix.cuh"
#include"cublas_v2.h"

__device__
float AC_sigmoid(float x);//!シグモイド関数です

__device__
float AC_differential_sigmoid(float x);//!シグモイド関数の導関数です

__device__
float AC_tanh(float x);//!双曲線関数です

__device__
float AC_differential_tanh(float x);//!双曲線関数の導関数です

__device__
float AC_identity(float x);//!恒等写像関数です

__device__
float AC_differential_identity(float x);//!恒等写像関数の導関数です

__device__
float AC_relu(float x);//!ランプ関数です

__device__
float AC_differential_relu(float x);//!ランプ関数の導関数です

__device__
float AC_plane(float x);//!好きな関数を定義できます

__device__
float AC_differential_plane(float x);//!何か関数を定義したときは、必ず導関数も定義してください

__global__
void apply_functions2D(ACTIVATE_FUNCTION func, float* src_data, unsigned int rows, unsigned int cols, float* result_data);//!行列に対して活性化関数を適用します

__global__
void apply_differnetial_functions2D(ACTIVATE_FUNCTION func, float* src_data, unsigned int rows, unsigned int cols, float* result_data);//!行列に対して活性化関数の導関数を適用します

/*!
* \brief   活性化関数をGPU上の指定された配列に適応します
* \param[in]	fucn		適用する活性化関数(hytan, sig, iden, relu, softmax, maxout,plane)
* \param[in]	src_matrix	適用するSigma_Matrix
* \param[in]	result_matrix		結果を出力するSigma_Matrix
*/
void apply_Acvtivate_function(ACTIVATE_FUNCTION func, Sigma_Matrix* src_matrix, Sigma_Matrix* result_matrix);

/*!
* \brief   活性化関数をGPU上の指定された配列に適応します
* \param[in]	fucn		適用する活性化関数(hytan, sig, iden, relu, softmax, maxout,plane)
* \param[in]	src_matrix	適用するSigma_Matrix
* \param[in]	result_matrix		結果を出力するSigma_Matrix
*/
void apply_d_Acvtivate_function(ACTIVATE_FUNCTION func, Sigma_Matrix* src_matrix, Sigma_Matrix* result_matrix);

#endif