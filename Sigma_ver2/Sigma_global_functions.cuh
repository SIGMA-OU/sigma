/*!
* \file    Sigma_global_functions.cuh
* \brief   Sigma�ŗp����Global�֐��Q�ł�
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
* \brief   �񎟌��z��data��lval�ɐݒ肵�܂��B
* \param[in]	data	�z��̃|�C���^
* \param[in]	rows	�s��
* \param[in]	cals	��
* \param[in]	val	�l
*/
__global__ void set_array_2D(float* data,unsigned int rows, unsigned int cols,float val);

/*!
* \brief   �񎟌��z��data��P�ʍs��ɏ��������܂�
* \param[in]	data	�z��̃|�C���^
* \param[in]	rows	�s��
* \param[in]	cals	��
*/
__global__ void set_array_identity_2D(float* data, unsigned int rows, unsigned int cols);

/*!
* \brief   �񎟌��z��data��step�ɏ��������܂�
* \param[in]	data	�z��̃|�C���^
* \param[in]	rows	�s��
* \param[in]	cals	��
*/
__global__ void set_array_step_2D(float* data, unsigned int rows, unsigned int cols);

/*!
* \brief   �񎟌��z��data�̑S�Ă̗v�f��val�����Z���܂�
* \param[in]	data	�z��̃|�C���^
* \param[in]	rows	�s��
* \param[in]	cals	��
* \param[in]	val	���Z����l
*/
__global__ void sum_constant_2D(float*data,unsigned int rows, unsigned int cols,float val);

/*!
* \brief   �񎟌��z�����l���z�ŏ��������܂��B
* \param[in]	data	�z��̃|�C���^
* \param[in]	rows	�s��
* \param[in]	cals	��
* \param[in]	x	�����l��
* \param[in]	y	����ly
*/
__global__ void initialize_uniform2D(float *data, unsigned int rows, unsigned int cols, float x, float y,curandState_t *state);

/*!
* \brief   �񎟌��z��𐳋K���z�ŏ��������܂�
* \param[in]	data	�z��̃|�C���^
* \param[in]	rows	�s��
* \param[in]	cals	��
* \param[in]	mean	����
* \param[in]	stddev	���U
*/
__global__ void initialize_normal2D(float *data, unsigned int rows, unsigned int cols, float mean, float stddev,curandState_t *state);

/*!
* \brief   �����𐶐����邽�߂ɁAcurandState_t�����������܂�
* \param[in]	seed	������seed�l(�����_���Ȓl�����Ă�������)
* \param[in]	rows	�s��
* \param[in]	cals	��
* \param[in]	*state	curandState_t
*/
__global__ void rand_init2D(unsigned long long seed,unsigned int rows, unsigned int cols, curandState_t *states);

/*!
* \brief	1�����z���Block�������s���܂��B\n(data[data_begin] -> src_data[0].....data[data_begin + src_data_num] -> src_data[src_data_num])
* \param[in]	data	�f�[�^���o�͂���z��
* \param[in]	data_begin	�]������z���̐擪�v�f
* \param[in]	src_data	�]������f�[�^�z��
* \param[in]	src_data_num	�]������f�[�^�z��̔z��
*/
__global__ void block1D(float* data, unsigned int data_begin, float* src_data, unsigned int src_data_num);

/*!
* \brief	�񎟌��摜�̃f�[�^����ݍ��ݏ������₷���`�ɔz�u��ς��܂�.
* \param[in]	result_data	�]����̃f�[�^�z��
* \param[in]	src_data	�񎟌��z��
* \param[in]	Image_width	�񎟌��z��̕�
* \param[in]	Image_height	�񎟌��z��̍���
* \param[in]	Image_channle	�񎟌��z��̃`���l����
* \param[in]	filter_width	�t�B���^�̕�
* \param[in]	filter_height	�t�B���^�̍���
* \param[in]	stride	�X�g���C�h��
* \param[in]	padding	�p�f�B���O���邩�ǂ���(bool�^�ł�)
* \param[in]	result_rows	�o�͐�s��̍s��
* \param[in]	result_cols	�o�͐�s��̗�
* \warning	�N������block,grid��result_data�̃f�[�^�p�f���ł��B�Ԉ��Ȃ��悤��!!
*/
__global__ void replace_array_for_CONV2D(float* result_data,float* src_data,
										unsigned int Image_width,unsigned int Image_height,unsigned Image_channel,
										unsigned int filter_width,unsigned int filter_height,unsigned int stride,
										bool padding,unsigned int result_rows, unsigned int result_cols);

/*!
* \brief	�񎟌���ݍ��݂̃f���^�̋t�`�d�̍ۂɎg�p���܂��B
* \param[in]	prev_delta	�O�̑w�̃f���^�s��
* \param[in]	src_data	replace_array
* \param[in]	Image_width	�񎟌��z��̕�
* \param[in]	Image_height	�񎟌��z��̍���
* \param[in]	Image_channle	�񎟌��z��̃`���l����
* \param[in]	filter_width	�t�B���^�̕�
* \param[in]	filter_height	�t�B���^�̍���
* \param[in]	stride	�X�g���C�h��
* \param[in]	padding	�p�f�B���O���邩�ǂ���(bool�^�ł�)
* \param[in]	prev_rows	prev_delta�̍s��
* \param[in]	prev_cols	prev_delta�̗�
* \param[in]	src_rows	src_data�̍s��
* \param[in]	src_cols	src_data�̗� 
* warninig	�N������block,grid��prev_delta�̊e�f�[�^�̗v�f���ł�
*/
__global__ void replace_array_for_backward_CONV2D(float* prev_delta, float* src_data,
										unsigned int Image_width, unsigned int Image_height, unsigned Image_channel,
										unsigned int filter_width, unsigned int filter_height, unsigned int stride,
										bool padding, unsigned int prev_rows, unsigned int prev_cols,
										unsigned int src_rows,unsigned int src_cols);

/*!
* \brief	�񎟌��}�b�N�X�v�[�����O���s���܂�
* \param[in]	output_data	�]����̃f�[�^�z��
* \param[in]	src_data	�񎟌��z��
* \param[in]	idx_matrix	�C���f�b�N�X�̏�񂪊i�[�����z��ł�
* \param[in]	Image_width	�񎟌��z��̕�
* \param[in]	Image_height	�񎟌��z��̍���
* \param[in]	Image_channle	�񎟌��z��̃`���l����
* \param[in]	outpout_width	�o�͐��width
* \param[in]	output_height	�o�͐��height
* \param[in]	pooling_width	�t�B���^�̕�
* \param[in]	pooling_height	�t�B���^�̍���
* \param[in]	stride	�X�g���C�h��
* \param[in]	padding	�p�f�B���O���邩�ǂ���(bool�^�ł�)
* \param[in]	output_rows	�o�͐�s��̍s��
* \param[in]	output_cols	�o�͐�s��̗�
* \warning	�N������block,grid��result_data�̃f�[�^�p�f���ł��B�Ԉ��Ȃ��悤��!!
*/
__global__ void max_Pooling2D(float* output_data, float* src_data,int* idx_matrix,
										unsigned int Image_width,unsigned int Image_height,unsigned int Image_channel,
										unsigned int output_width, unsigned int output_height,
										unsigned int pooling_width,unsigned int pooling_height,unsigned int stride,
										bool padding, unsigned int output_rows,unsigned int output_cols);

/*!
* \brief	�񎟌��A�x���[�W�v�[�����O���s���܂�
* \param[in]	output_data	�]����̃f�[�^�z��
* \param[in]	src_data	�񎟌��z��
* \param[in]	Image_width	�񎟌��z��̕�
* \param[in]	Image_height	�񎟌��z��̍���
* \param[in]	Image_channle	�񎟌��z��̃`���l����
* \param[in]	outpout_width	�o�͐��width
* \param[in]	output_height	�o�͐��height
* \param[in]	pooling_width	�t�B���^�̕�
* \param[in]	pooling_height	�t�B���^�̍���
* \param[in]	stride	�X�g���C�h��
* \param[in]	padding	�p�f�B���O���邩�ǂ���(bool�^�ł�)
* \param[in]	output_rows	�o�͐�s��̍s��
* \param[in]	output_cols	�o�͐�s��̗�
* \warning	�N������block,grid��result_data�̃f�[�^�p�f���ł��B�Ԉ��Ȃ��悤��!!
*/
__global__ void average_Pooling2D(float* output_data, float* src_data,
										unsigned int Image_width, unsigned int Image_height, unsigned int Image_channel,
										unsigned int output_width,unsigned int output_height,
										unsigned int pooling_width, unsigned int pooling_height, unsigned int stride,
										bool padding, unsigned int output_rows, unsigned int output_cols);

/*!
* \brief	�񎟌��̃}�b�N�X�v�[�����O����delta���t�`�d���܂�
* \param[in]	prev_delta	�O�̑w�̃f���^
* \param[in]	src_data	���w�̃f���^
* \param[in]	idx_matirx	�C���f�b�N�X��񂪊i�[���ꂽ�z��
* \param[in]	prev_delta_width	�O�w�̃f���^�̕�
* \param[in]	prev_delta_height	�O�w�̃f���^�̍���
* \param[in]	prev_delta_channle	�O�w�̃f���^�̃`���l����
* \param[in]	src_width	���w��width
* \param[in]	src_height	���w��height
* \param[in]	pooling_width	�v�[�����O�̕�
* \param[in]	pooling_height	�v�[�����O�̍���
* \param[in]	stride	�X�g���C�h��
* \param[in]	padding	�p�f�B���O���邩�ǂ���(bool�^�ł�)
* \param[in]	prev_delta_rows	�O�̑w�̃f���^�̍s��
* \param[in]	prev_delta_cols	�O�̑w�̃f���^�̗�
* \warning	�N������block,grid��prev_delta_data�̃f�[�^�p�f���ł��B�Ԉ��Ȃ��悤��!!
*/
__global__ void backward_for_max_Pooling2D(float* prev_delta, float* src_data,int* idx_matrix,
						unsigned int prev_delta_width, unsigned int prev_delta_height, unsigned int prev_delta_channel,
						unsigned int src_width,unsigned int src_height,
						unsigned int pooling_width,unsigned int pooling_height,unsigned int stride,
						bool padding,unsigned int prev_delta_rows,unsigned int prev_delta_cols);

/*!
* \brief	�񎟌��̃}�b�N�X�v�[�����O����delta���t�`�d���܂�
* \param[in]	prev_delta	�O�̑w�̃f���^
* \param[in]	src_data	���w�̃f���^
* \param[in]	prev_delta_width	�O�w�̃f���^�̕�
* \param[in]	prev_delta_height	�O�w�̃f���^�̍���
* \param[in]	prev_delta_channle	�O�w�̃f���^�̃`���l����
* \param[in]	src_width	���w��width
* \param[in]	src_height	���w��height
* \param[in]	pooling_width	�v�[�����O�̕�
* \param[in]	pooling_height	�v�[�����O�̍���
* \param[in]	stride	�X�g���C�h��
* \param[in]	padding	�p�f�B���O���邩�ǂ���(bool�^�ł�)
* \param[in]	prev_delta_rows	�O�̑w�̃f���^�̍s��
* \param[in]	prev_delta_cols	�O�̑w�̃f���^�̗�
* \warning	�N������block,grid��prev_delta_data�̃f�[�^�p�f���ł��B�Ԉ��Ȃ��悤��!!
*/
__global__ void backward_for_average_Pooling2D(float* prev_delta, float* src_data,
						unsigned int prev_delta_width, unsigned int prev_delta_height, unsigned int prev_delta_channel,
						unsigned int src_width, unsigned int src_height,
						unsigned int pooling_width, unsigned int pooling_height, unsigned int stride,
						bool padding, unsigned int prev_delta_rows, unsigned int prev_delta_cols);

#endif