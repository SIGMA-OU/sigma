/*!
* \file    Sigma_Matrix.cuh
* \brief   Sigma_Matrix���`���Ă��܂��B
* \date    2016/06/18
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef SIGMA_MATRIX_CUH_
#define SIGMA_MATRIX_CUH_
#undef NDEBUG

#include"Device_Matrix.cuh"

/*!
* \brief	������ \c Device_Matrix�@��ێ����邽�߂̃N���X�ł�
* \details	������ \c Device_Matrix ��ێ����邽�߂̃N���X�ł��B
*          �@\c Device_Matrix �́Aadd_Matrix()�Œǉ����邱�Ƃ��ł��܂��B
*			�ǉ�����ۂ́A�K�������T�C�Y��\c Device_Matrix��ǉ����Ă��������B
*			\c Device_Matrix ��Sigam�QMatrix�ɒǉ����邱�ƂŁA�f�[�^��S�ĂȂ��邱�Ƃ��ł�
*			�v�Z�����̍œK�����s���Ă��܂��B
*/
class Sigma_Matrix{
public:
	/*!
	* \brief   ��� \c Sigma_Matrix �I�u�W�F�N�g���쐬���܂��B
	*/
	Sigma_Matrix();

	/*!
	* \brief	�f�X�g���N�^�ł�
	*/
	~Sigma_Matrix(){
		if (data_ != NULL)CHECK(cudaFree(data_));
	}

	/*!
	* \brief   �R�s�[�R���X�g���N�^�ł�
	* \param[in]	obj	�R�s�[����Sigma_Matrix
	*/
	Sigma_Matrix(const Sigma_Matrix &obj);

	/*!
	* \brief   �w�肵���s/�񐔂� \c Sigma_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	* \param[in]   rows        �s��
	* \param[in]   cols        ��
	* \param[in]   init       �������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	Sigma_Matrix(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.1f);

	/*!
	* \brief   �w�肵���s/�񐔂� \c Sigma_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	* \param[in]	height	����
	* \param[in]	width	��
	* \param[in]
	* \param[in]   init       �������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	Sigma_Matrix(unsigned height, unsigned int width = 1, unsigned int channel = 1, unsigned int depth = 1, unsigned int batch_size = 1, INITIALIZER init = ZERO, float a = 0, float b = 0.1f);

	/*!
	* \brief   �w�肵���s/�񐔂� \c Sigma_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	* \param[in]	height	����
	* \param[in]	width	��
	* \param[in]
	* \param[in]   init       �������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	void initialize(unsigned height,unsigned int width = 1, unsigned int channel = 1, unsigned int depth = 1, unsigned int batch_size = 1,INITIALIZER init = ZERO , float a = 0, float b = 0.1f);
	
	/*!
	* \brief   �w�肵���s/�񐔂� \c Sigma_Matrix �I�u�W�F�N�g���쐬���A�����Ƀf�[�^�̏��������s���܂��B
	*          init��CONSTAN�̎��́A�������ɐݒ肵�����l������B
	*�@�@�@�@�@ init��GAUSSIAN�܂���UNIF_DISTRIB�̂Ƃ��͑������܂Œl�������
	*			���ꂼ��(������,������) = (����, ���U)or(x,y)�ƂȂ�
	* \param[in]   rows        �s��
	* \param[in]   cols        ��
	* \param[in]   init       �������q (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  �����l(CONSTANT) ����(GAUSSIAN)�@�l�悘(UNIF_DISTRIB)
	* \param[in]   b		�@���U(GAUSSIAN)�@�l��y(UNIF_DISTRIB)
	*/
	void initialize(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.1f);

	/*!
	* \brief	\c Device_Matrix��ǉ����܂�
	* \param[in]	Matrix	�f�o�C�X�}�g���b�N�X�ł�
	*/
	void add_Matrix(Device_Matrix& Matrix);

	/*!
	* \brief	Sigma_Matrix����\c Device_Matrix���擾���܂�
	* \param[in]	num	�擾������Matrix�̔ԍ�
	* \warning	�K��Batch_size_�����������ԍ������Ă��������Bindex�́@0�@����n�܂�܂�
	*/
	Device_Matrix& get_Matrix(unsigned int idx);

	/*!
	* \brief	Sigma_Matrix�̓�������\�����܂�
	* \param[in]	All_Status	���g�̔z����o�͂��邩�ǂ���
	*/
	void print(bool All_status = false);

	/////////////////////////////////// getter  /////////////////////
	/*!
	* \brief	height���擾���܂�
	* \return	height	
	*/
	inline unsigned int height(){ return height_; };

	/*!
	* \brief	width���擾���܂�
	* \return	width
	*/
	inline unsigned int width(){ return width_; };

	/*!
	* \brief	channel���擾���܂�
	* \return	channel
	*/
	inline unsigned int channel(){ return channel_; };

	/*!
	* \brief	depth���擾���܂�
	* \return	depth
	*/
	inline unsigned int depth(){ return depth_; };

	/*!
	* \brief	batch_size���擾���܂�
	* \return	batch_size
	*/
	inline unsigned int batch_size(){ return batch_size_; };

	/*!
	* \brief	rows���擾���܂�
	* \return	rows
	*/
	inline unsigned int rows(){ return rows_; };

	/*!
	* \brief	cols���擾���܂�
	* \return	cols
	*/
	inline unsigned int cols(){ return cols_; };

	/*!
	* \brief	size���擾���܂�
	* \return	size
	*/
	inline size_t size(){ return size_; };

	/*!
	* \brief	�z��̃|�C���^���擾���܂�
	* \return	data
	* \warning	GPU��̃|�C���^�ł���ACPU��������Ƀf�[�^��������̂ł͂���܂���
	*/
	inline float* data(){ return data_; };

	/*!
	* \brief	�i�[���ꂽ�~�j�o�b�`�̐擪�A�h���X��Ԃ��܂�
	* \param[in]	batch_num	�w��o�b�`�ԍ��ł�(0����n�܂�܂�)
	* \return	batch_num	�w�肵���o�b�`�ԍ��ł�(0����n�܂�_�ɒ���)
	* \warning	GPU��̃|�C���^�ł���ACPU��������Ƀf�[�^��������̂ł͂���܂���(��{�I�ɓ��������݂̂Ŏg���܂�)
	*/
	inline float* mini_batch_data(unsigned int batch_num){
		if (batch_num >= batch_size_){
			cout << "error : mini_batch_data : over size of batch_num";
			return data_;
		}
		else{
			return data_ + batch_num *(width_ * height_ * channel_ * depth_) ;
		}
	};

	/////////////////////setter////////////////////////////////
	/*!
	* \brief	�f�[�^�̒l��ݒ肵�܂�\n
	* \param[in]	row �s�ԍ��ł�
	* \param[in]	col	��ԍ��ł�
	* \param[in]	value	�ݒ肷��l�ł�
	* \warning	�z��̃T�C�Y�𒴂��Ȃ��悤�ɒ��ӂ��Ă��������B
	*/
	void set_value(unsigned int row, unsigned int col,float value);

	/*!
	* \brief	�f�[�^��z��f�[�^(Host��)�Őݒ肵�܂��B(��D��̔z��f�[�^�ɂ��Ă��������B)\n
	* \param[in]	data
	* \warning	data_size���R�s�[�g�p�Ƃ��܂��̂ŁAdata�́A�����T�C�Y���p�ӂ��Ă��������B
	*/
	void set_matrix(float* data);

	/*!
	* \brief	�f�[�^�̎w�肳�ꂽ�o�b�`�ԍ��̃f�[�^���A�z��f�[�^�Őݒ肵�܂�
	* \param[in]	data	�z��f�[�^
	* \param[in]	star_idx�@�o�b�`���̉��Ԗڂ�
	* \param[in]	data_num	����data�ɂ́A���f�[�^�������Ă��邩
	* \warning	data_num���f�[�^��ݒ肵�܂��̂ŁA�e�ʂ��I�[�o�[���Ȃ��悤�ɒ��ӂ��Ă��������B
	*/
	void set_matrix(float* data, unsigned int start_idx, unsigned int data_num = 1);

	/*!
	* \brief	�f�[�^�����o�C�i����CSV�ɏo�͂��܂�
	* \param[in]	file_path	�t�@�C�����ł�
	* \param[in]	binary_output	�o�C�i���[�f�[�^�œf���o������I�ׂ܂�(flase��CSV�o�͂���܂�)
	* \warning	.csv��file_path�ɕK�v����܂���\n��D��̍s�񂪓]�u���ꂽ��Ԃŕۑ�����܂��B
	*/
	void save(string file_path,bool binary_output = true);

	/*!
	* \brief	�o�C�i���t�@�C����ǂݍ��݂܂�
	* \param[in]	file_path	�t�@�C�����ł�(.dat�͂���܂���)
	*/
	void load(string file_path);

	/////////////////////////////////////////////////////
	Sigma_Matrix& operator = (const Sigma_Matrix& obj);

	Sigma_Matrix& operator /= (float x);

	Sigma_Matrix& operator *= (float x);

	Device_Matrix& operator [](unsigned int idx);

private:
	//!device_Matrix���擾�������Ƃ��ɗp���܂�
	Device_Matrix d_mx;

	//!�e \c Device_Matrix ��width�ł�
	unsigned int width_;

	//!�e \c Device_Matrix ��height�ł�
	unsigned int height_;

	//!�e \c Device_Matrix ��channel�ł�
	unsigned int channel_;

	//!�e \c Device_Matrix ��depth�ł�
	unsigned int depth_;

	//! \c Device_Matrix �̕ێ����ł��B�o�b�`�̑傫���ɂȂ�܂��B
	unsigned int batch_size_;

	//! \c Sigma_Matrix �̑S�Ẵf�[�^�z��̍s���ł�
	unsigned int rows_;

	//! \c Sigma_Matrix �̑S�Ẵf�[�^�z��̗񐔂ł�
	unsigned int cols_;

	//!�S�Ẵf�[�^�̃f�[�^��ł��B
	float *data_;

	//!�S�Ẵf�[�^�̃T�C�Y�ł�
	size_t size_;

};



#endif