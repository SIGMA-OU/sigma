/*!
* \file    Utilities.cuh
* \brief   �֗��Ȋ֐��S�ł�
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

//![0, 1)�̈�l���z�̗������o�͂���֐��ł��B
float uniform();

//!����mu�A�W���΍�sigma�̐��K���z�̗����𐶐�����֐��ł�
float gaussian(float mu, float sigma);

/*!
* \brief   GPU���ŁA�f�[�^�̗�D�悩��s�D��փf�[�^�z�u��������֐�
* \param[in]   d_data       �ڂ��f�[�^��̃|�C���^(�f�o�C�X�|�C���^)
* \param[in]   src_data		�ڂ��f�[�^���̃|�C���^(�f�o�C�X�|�C���^)
* \param[in]   rows		�@�@�ڂ��f�[�^����row
* \param[in]   cols		�@ �ڂ��f�[�^����col
*/
__global__ void c2r_major_change2D(float* d_data, float* src_data, unsigned int rows, unsigned int cols);

/*!
* \brief   GPU���ŁA�f�[�^�̍s�D�悩���D��փf�[�^�z�u��������֐�
* \param[in]   d_data       �ڂ��f�[�^��̃|�C���^(�f�o�C�X�|�C���^)
* \param[in]   src_data		�ڂ��f�[�^���̃|�C���^(�f�o�C�X�|�C���^)
* \param[in]   rows		�@�@�ڂ��f�[�^����row
* \param[in]   cols		�@ �ڂ��f�[�^����col
*/
__global__ void r2c_major_change2D(float* d_data, float* src_data, unsigned int rows, unsigned int cols);

/*!
* \brief   �񎟌��z��(��D��)�̊e��Ɉꎟ���z��̒l�����Z���܂�
* \param[in]	d_data		�ڂ��f�[�^��̃|�C���^(�f�o�C�X�|�C���^)
* \param[in]	rows		d_data�̍s��
* \param[in]	cols		d_data�̗�
* \param[in]	src_data	�������킹���s��
*/
__global__ void plus_each_col2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data);

/*!
* \brief   �񎟌��z��(��D��)�̊e�s�Ɉꎟ���z��̒l�����Z���܂�
* \param[in]	d_data		�ڂ��f�[�^��̃|�C���^(�f�o�C�X�|�C���^)
* \param[in]	rows		d_data�̍s��
* \param[in]	cols		d_data�̗�
* \param[in]	src_data	�������킹��s�s��
*/
__global__ void plus_each_row2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data);

/*!
* \brief   �񎟌��z��(��D��)�̗v�f���̊|���Z(d_data[i][j] = d_data[i][j] �� src_data[i][j])
* \param[in]	d_data		�ڂ��f�[�^��̃|�C���^(�f�o�C�X�|�C���^)
* \param[in]	rows		d_data�̍s��
* \param[in]	cols		d_data�̗�
* \param[in]	src_data	�|�����킹��
*/
__global__ void multiply2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data);

/*!
* \brief   src_data����subtruct_data�������܂�
* \param[in]	subtruct_data	�������߂�Matrix
* \param[in]	src_data	���f�[�^Matrix
* \param[in]	rows	src_Matrix�̍s��
* \param[in]	cols	src_Matrix�̗�
*/
__global__ void subtractArray2D(float *subtruct_data, float *src_data, unsigned int rows, unsigned int cols);

/*!
* \brief   src_data����subtruct_data�����������̂�result_data�ֈڂ��܂�
* \param[in]	subtruct_data	�������߂�Matrix
* \param[in]	src_data	���f�[�^Matrix
* \param[in]	result_data	���ʏo�͐�
* \param[in]	rows	src_Matrix�̍s��
* \param[in]	cols	src_Matrix�̗�
*/
__global__ void subtractArray2D(float *subtruct_data, float *src_data, float *result_data, unsigned int rows, unsigned int cols);

/*!
* \brief   1�����ɕϊ����ꂽ2�������ɑ΂��āAPadding��K�p���ď�ݍ��ݏ������s���܂��B
* \param[in]	subtruct_data	�������߂�Matrix
* \param[in]	src_data	���f�[�^Matrix
* \param[in]	result_data	���ʏo�͐�
* \param[in]	rows	src_Matrix�̍s��
* \param[in]	cols	src_Matrix�̗�
*/
__global__ void Convolutuion2D_from1Darray_PadOn(float* Img_data,
	unsigned int Img_width, unsigned int Img_height, unsigned int Img_channel,
	float* filter_data,
	unsigned int Fil_width, unsigned int Fil_height, unsigned int Fil_channel,
	unsigned int stride,
	unsigned int Out_width, unsigned int Out_height
	);


/*!
* \brief   1�����ɕϊ����ꂽ2�������ɑ΂��āAPadding��K�p���ď�ݍ��ݏ������s���܂��B
* \param[in]	subtruct_data	�������߂�Matrix
* \param[in]	src_data	���f�[�^Matrix
* \param[in]	result_data	���ʏo�͐�
* \param[in]	rows	src_Matrix�̍s��
* \param[in]	cols	src_Matrix�̗�
*/
__global__ void Convolutuion2D_from1Darray_PadOff(float* Img_data,
	unsigned int Img_width, unsigned int Img_height, unsigned int Img_channel,
	float* filter_data,
	unsigned int Fil_width, unsigned int Fil_height, unsigned int Fil_channel,
	unsigned int stride
	);


#endif
