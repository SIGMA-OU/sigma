/*!
* \file    Activate_Function.cuh
* \brief   �g�p���銈�����֐����`���Ă��܂��B
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
float AC_sigmoid(float x);//!�V�O���C�h�֐��ł�

__device__
float AC_differential_sigmoid(float x);//!�V�O���C�h�֐��̓��֐��ł�

__device__
float AC_tanh(float x);//!�o�Ȑ��֐��ł�

__device__
float AC_differential_tanh(float x);//!�o�Ȑ��֐��̓��֐��ł�

__device__
float AC_identity(float x);//!�P���ʑ��֐��ł�

__device__
float AC_differential_identity(float x);//!�P���ʑ��֐��̓��֐��ł�

__device__
float AC_relu(float x);//!�����v�֐��ł�

__device__
float AC_differential_relu(float x);//!�����v�֐��̓��֐��ł�

__device__
float AC_plane(float x);//!�D���Ȋ֐����`�ł��܂�

__device__
float AC_differential_plane(float x);//!�����֐����`�����Ƃ��́A�K�����֐�����`���Ă�������

__global__
void apply_functions2D(ACTIVATE_FUNCTION func, float* src_data, unsigned int rows, unsigned int cols, float* result_data);//!�s��ɑ΂��Ċ������֐���K�p���܂�

__global__
void apply_differnetial_functions2D(ACTIVATE_FUNCTION func, float* src_data, unsigned int rows, unsigned int cols, float* result_data);//!�s��ɑ΂��Ċ������֐��̓��֐���K�p���܂�

/*!
* \brief   �������֐���GPU��̎w�肳�ꂽ�z��ɓK�����܂�
* \param[in]	fucn		�K�p���銈�����֐�(hytan, sig, iden, relu, softmax, maxout,plane)
* \param[in]	src_matrix	�K�p����Sigma_Matrix
* \param[in]	result_matrix		���ʂ��o�͂���Sigma_Matrix
*/
void apply_Acvtivate_function(ACTIVATE_FUNCTION func, Sigma_Matrix* src_matrix, Sigma_Matrix* result_matrix);

/*!
* \brief   �������֐���GPU��̎w�肳�ꂽ�z��ɓK�����܂�
* \param[in]	fucn		�K�p���銈�����֐�(hytan, sig, iden, relu, softmax, maxout,plane)
* \param[in]	src_matrix	�K�p����Sigma_Matrix
* \param[in]	result_matrix		���ʂ��o�͂���Sigma_Matrix
*/
void apply_d_Acvtivate_function(ACTIVATE_FUNCTION func, Sigma_Matrix* src_matrix, Sigma_Matrix* result_matrix);

#endif