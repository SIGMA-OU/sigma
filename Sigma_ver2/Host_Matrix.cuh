/*!
* \file    Host_Matrix.cuh
* \brief   Sigma_Matrix�ɗp�����Ă���Host_Matrix���`���Ă��܂��B
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
* \brief   CPU��ł̍s��������N���X�ɂȂ�܂�
* \details GPU�v�Z���s�����߂ɕK�v��CPU��ł̃��������m�ۂ��Ă��܂��B
*          �S�Ẵf�[�^��float�`�̐��l�Ƃ��Ĉ����܂��B��{�I��col_major�ł�
*/
class Host_Matrix{
public:
	/*!
	* \brief   ��� \c Host_Matrix �I�u�W�F�N�g���쐬���܂��B
	*/
	Host_Matrix(){};

	/*!
	* \brief   �w�肵���s/�񐔂� \c Host_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	*
	* \param[in]   rows        �s��
	* \param[in]   cols        ��
	* \param[in]   init       �������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	Host_Matrix(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	/*!
	* \brief   �f�X�g���N�^�B�������Ɋi�[����Ă���f�[�^��j�����܂��B
	*/
	~Host_Matrix(){ if (data_ != NULL) free(data_); };

	/*!
	* \brief   �R�s�[�R���X�g���N�^.�s��̃f�[�^���܂߂ăR�s�[���܂�
	*/
	Host_Matrix(const Host_Matrix& obj);

	////////////////////////////////	getter		////////////////////////////////
	/*!
	* \brief   ��widht_���擾���܂�
	* \return  ��
	*/
	inline unsigned int width(){ return width_; };

	/*!
	* \brief   ����height_���擾���܂�
	* \return  ����
	*/
	inline unsigned int height(){ return height_; };

	/*!
	* \brief   �`���l��channel_���擾���܂�
	* \return  �`���l����
	*/
	inline unsigned int channel(){ return channel_; };

	/*!
	* \brief   �[��depth_���擾���܂�(�RD���̎��Ɏg���܂�)
	* \return  �[��
	*/
	inline unsigned int depth(){ return depth_; };

	/*!
	* \brief   �s��rows_���擾���܂�
	* \return  �s��
	*/
	inline unsigned int rows() { return rows_; };

	/*!
	* \brief   ��cols_���擾���܂�
	* \return  ��
	*/
	inline unsigned int cols(){ return cols_; };

	/*!
	* \brief   �i�[���Ă���f�[�^�̃|�C���^���擾���܂��B
	* \return  �i�[���Ă���f�[�^�̃|�C���^
	* \warning	�Ԃ����|�C���^�̓f�o�C�X��ł̃|�C���^�ł��B
	*/
	inline  float* data(){ return data_; }

	/*!
	* \brief   �f�[�^�̃T�C�Y��size_t�^�Ŏ擾���܂�
	* \return  �i�[���Ă���f�[�^�̃T�C�Y
	*/
	inline size_t size(){ return size_; };

	/*!
	* \brief   CPU��Ɋi�[����Ă���f�[�^���擾���܂�
	* \prama[in]	height	�����ł�(�f�[�^���񎟌��̏ꍇ��row��\���܂�)
	* \prama[in]	width	���ł�(�f�[�^���񎟌��̏ꍇ��col��\���܂�)
	* \prama[in]	channel	�`���l���ł�
	* \prama[in]	depth	�[���ł�
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
	* \brief   \c Host_Matrix �̑S�Ẵf�[�^���o�͂��܂�
	* \prama[in]	All_status	true�ɂ��邱�ƂŁA�f�[�^���S�ĕ\�����܂�
	*/
	void print(bool All_status = false);

private:
	//!width�ł�
	unsigned int width_;

	//!height�ł�
	unsigned int height_;

	//!channel�ł�
	unsigned int channel_;

	//!depth�ł�
	unsigned int depth_;

	//!�s���ł�
	unsigned int rows_;

	//!�񐔂ł�
	unsigned int cols_;

	//!�f�[�^�z��(CPU��Ƀ��������m�ۂ��܂�)
	float* data_;

	//!�T�C�Y�ł�
	size_t size_;
};

#endif