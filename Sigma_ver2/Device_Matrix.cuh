/*!
* \file    Device_Matrix.cuh
* \brief   Sigma_Matrix�ɗp�����Ă���Device_Matrix���`���Ă��܂��B
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
* \brief   GPU��ł̍s��������N���X�ɂȂ�܂�
* \details GPU�v�Z���s�����߂ɕK�v��GPU��ł̃��������m�ۂ��Ă��܂��B
*          �S�Ẵf�[�^��float�`�̐��l�Ƃ��Ĉ����܂��B��{�I��col_major�ł�
*/
class Device_Matrix{
public:
	/*!
	* \brief   ��� \c Device_Matrix �I�u�W�F�N�g���쐬���܂��B
	*/
	Device_Matrix();

	/*!
	* \brief   Device_Matrix��p����\c Device_Matrix�I�u�W�F�N�g���쐬���܂��B(�R�s�[�R���X�g���N�^)
	* \prama[in]  obj  Device_Matrix
	*/
	Device_Matrix(const Device_Matrix& obj);

	/*!
	* \brief   �w�肵���s/�񐔂� \c Device_Matrix �I�u�W�F�N�g���쐬���C
	*          �����Ƀf�[�^�̏��������s���܂��B
	* \param[in]   rows        �s��
	* \param[in]   cols        ��
	* \param[in]   *host_data	  �f�[�^���i�[���ꂽfloat�^�|�C���^(CPU��̗�D��̃f�[�^�Ƃ���)
	*/
	Device_Matrix(unsigned int rows, unsigned int cols, float* host_data);

	/*!
	* \brief   �w�肵���s/�񐔂� \c Device_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	* \param[in]   rows        �s��
	* \param[in]   cols        ��
	* \param[in]   init       �������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	Device_Matrix(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	/*!
	* \brief   �w�肵���s/�񐔂� \c Device_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	* \param[in]   height	����
	* \param[in]   width	��
	* \param[in]   channel	�`���l����
	* \param[in]   depth	�[��
	* \param[in]   init		�������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	Device_Matrix(unsigned int height, unsigned int width = 1, unsigned int channel = 1, unsigned int depth = 1, INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	/*!
	* \brief   �f�X�g���N�^�B�f�o�C�X���Ɋi�[����Ă���f�[�^��j�����܂��B
	*/
	~Device_Matrix(){ if (data_ != NULL) cudaFree(data_); };

	/*!
	* \brief   Device_Matrix��p����\c Device_Matrix�I�u�W�F�N�g�����������܂��B
	* \prama[in]  obj  Device_Matrix
	*/
	void initialize(Device_Matrix& obj);

	/*!
	* \brief   �w�肵���s/�񐔂� \c Device_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	* \param[in]   rows        �s��
	* \param[in]   cols        ��
	* \param[in]   init       �������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	void initialize(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	/*!
	* \brief   �w�肵���s/�񐔂� \c Device_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	* \param[in]   height	����
	* \param[in]   width	��
	* \param[in]   channel	�`���l����
	* \param[in]   depth	�[��
	* \param[in]   init		�������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	void initialize(unsigned int height, unsigned int width = 1, unsigned int channel = 1, unsigned int depth = 1,INITIALIZER init = ZERO, float a = 0, float b = 0.01f);

	///////////////////////////////		read / write	funciton/////////////////////////////////////////////

	/*!
	* \brief	�t�@�C����ǂݍ���� \c Device_Matrix �����������܂�
	* \prama[in]	file_name	�ǂݍ��ރt�@�C����
	*/
	void read_data(string file_name);

	/*!
	* \brief	�f�[�^���������݂܂�
	* \prama[in]	file_name	�������ރt�@�C�������������݂܂�
	*/
	void write_data(string file_name);

	///////////////////////////////		setter	    ///////////////////////////////
	/*!
	* \brief   GPU��Ɋi�[����Ă���f�[�^����͂��܂�
	* \param[in]	val	���͂���l�ł�
	* \prama[in]	height	�����ł�(�f�[�^���񎟌��̏ꍇ��row��\���܂�)
	* \prama[in]	width	���ł�(�f�[�^���񎟌��̏ꍇ��col��\���܂�)
	* \prama[in]	channel	�`���l���ł�
	* \prama[in]	depth	�[���ł�
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
	* \brief   �f�[�^�̃|�C���^��ύX���܂�
	* \param[in]	data	GPU��ł̃f�[�^�|�C���^
	* \warning	�K��GPU��ł̃|�C���^�������ɂ��Ă��������B�T�C�Y�̕ύX���s���܂���
	*/
	void set_data(float* data){ data_ = data; };

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
	* \warning	�Ԃ����|�C���^�̓f�o�C�X��ł̃|�C���^�ł��Bdata[i]�Ȃǂ������ƃG���[���͂��܂��B
	*/
	inline  float* data(){ return data_; }

	/*!
	* \brief   �f�[�^�̃T�C�Y��size_t�^�Ŏ擾���܂�
	* \return  �i�[���Ă���f�[�^�̃T�C�Y
	*/
	inline size_t size(){ return size_; };
	
	/*!
	* \brief   GPU��Ɋi�[����Ă���f�[�^���擾���܂�
	* \prama[in]	height	�����ł�(�f�[�^���񎟌��̏ꍇ��row��\���܂�)
	* \prama[in]	width	���ł�(�f�[�^���񎟌��̏ꍇ��col��\���܂�)
	* \prama[in]	channel	�`���l���ł�
	* \prama[in]	depth	�[���ł�
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
	* \brief	������Z�q�ł�
	*/
	Device_Matrix& operator = (const Device_Matrix& obj);
	
	//!�������ł�
	Device_Matrix& operator * (const Device_Matrix& obj);
	//!�������ł�
	Device_Matrix& operator + (const Device_Matrix& obj);
	//!�������ł�
	Device_Matrix& operator - (const Device_Matrix& obj);

	/////////////////////////////    print /////////////////////////
	/*!
	* \brief   \c Device_Matrix �̑S�Ẵf�[�^���o�͂��܂�
	* \prama[in]	All_status	true�ɂ��邱�ƂŁA�f�[�^���S�ĕ\�����܂�
	*/
	void print(bool All_status = false);

private:
	//!CPU��Ƀf�[�^��]������ۂɗp���� \c Host_Matrix �ł�
	//Host_Matrix host_matrix_;

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

	//!�f�[�^�z��(GPU��Ƀ��������m�ۂ��܂�)
	float* data_;

	//!�T�C�Y�ł�
	size_t size_;
};

#endif