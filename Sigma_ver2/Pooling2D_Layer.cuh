/*!
* \file    Pooling2D_Layer.cuh
* \brief	�񎟌��v�[�����O�w���`���Ă��܂��B
* \date    2016/07/10
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef POOLING2D_LAYER_CUH_
#define POOLING2D_LAYER_CUH_
#undef NDEBUG

#include"Abstract_Middle_Layer.cuh"
#include"Optimizer.cuh"
#include"Additional_Process.cuh"
#include"Activate_Function.cuh"
#include"Utilities.cuh"
#include"Sigma_global_functions.cuh"
#include<vector>
using namespace std;

class Pooling2D_Layer : public Abstract_Middle_Layer{
public:
	//!�v�[�����O�w���K�肷��p�����[�^�ł�
	struct Param{
		//!�v�[�����O���s�����ł�
		unsigned int width;
		//!�v�[�����O���s�������ł�
		unsigned int height;
		//!�X�g���C�h�ł�
		unsigned int stride;
		//!�p�f�B���O�̗L���ł�
		bool padding;
		//!�}�b�N�X�v�[�����O���A�x���[�W�v�[�����O����I�ׂ܂�
		POOLING_METHOD method;
	};

	/*!
	* \brief	�f�t�H���g�R���X�g���N�^�ł�
	*/
	Pooling2D_Layer();

	/*!
	* \brief	�f�t�H���g�R���X�g���N�^�ł�
	* \param[in]	param	�v�[�����O�w�̃p�����[�^
	*/
	Pooling2D_Layer(Pooling2D_Layer::Param &param);

	/*!
	* \brief	�f�t�H���g�R���X�g���N�^�ł�
	* \param[in]	width	�v�[�����O�w�̕�
	* \param[in]	height	�v�[�����O�w�̍���
	* \param[in]	stride	�v�[�����O�w�̃X�g���C�h
	* \param[in]	padding	�p�f�B���O�����邩�ǂ���
	* \param[in]	method	MAXpooling or AVERAGEpooling��
	*/
	Pooling2D_Layer(unsigned int width, unsigned int height,unsigned int stride, bool padding, POOLING_METHOD method);

	//!�f�t�H���g�f�X�g���N�^
	~Pooling2D_Layer(){ if (d_idx_matrix != NULL)CHECK(cudaFree(d_idx_matrix)); };

	/*!
	* \brief   ���`�d���܂�
	* \param[in]	input	���͍s��ł�
	* \param[in]	mode	�w�K�����A���_����
	* \return	output
	*/
	Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode);
	
	/*!
	* \brief   ���������܂�
	* \param[in]	prev_output	�O�̑w�̏o�͂ł�
	* \param[in]	output
	*/
	Sigma_Matrix* initialize(Sigma_Matrix* prev_output);

	/*!
	* \brief   �o�b�`�T�C�Y��ύX�����Ƃ��ɁA�e�w�̏o�͂̑傫����ύX���܂�
	* \param[in]	prev_output	�O�̑w�̏o�͂ł�
	* \return	output
	*/
	Sigma_Matrix* change_batchsize(Sigma_Matrix* prev_output);

	/*!
	* \brief   �o�͂���������l��Ԃ��܂�
	* \return	d_output
	*/
	Sigma_Matrix* get_d_output();

	/*!
	* \brief   delta���擾���܂�
	* \return	delta
	*/
	inline Sigma_Matrix* get_delta(){ return &delta_; };

	/*!
	* \brief	�w�K�ɕK�v��\c Optimizer �����������܂�
	* \param[in]	prev_output
	*/
	Sigma_Matrix* initialize_optimizer(Sigma_Matrix* prev_output);

	/*!
	* \brief   �t�`�d���܂�
	* \param[in]	prev_delta	�O�̑w�̃f���^�s��ł�
	* \param[in]	prev_d_output	�O�̑w�̔������ꂽ�o�͂ł�
	*/
	void back_ward(Sigma_Matrix* prev_delta, Sigma_Matrix* prev_d_output);

	/*!
	* \brief   Optimizer��p���āA�p�����[�^�̕ύX�ʂ��v�Z���܂�(�v�[�����O�w�́A�������܂���) 
	*/
	Sigma_Matrix* calc_update_param(Sigma_Matrix* prev_output){ return &output_; };

	/*!
	* \brief   �p�����[�^��ύX���܂�(�v�[�����O�w�͉������܂���) 
	*/
	void update_param(){ return; };

	/*!
	* \brief   Additional_Process��ǉ����܂��B
	* \param[in]	ap	Additional_Process
	*/
	void add_process(Additional_Process& ap);

	//!�������ł�
	void save(string file_path){ return; };

	//!�������ł�
	void load(string file_path){ return; };

	///////////////////////////  Getter   /  Setter   /////////////////////
	
	/*!
	* \brief	�p�����[�^��ݒ肵�܂�
	* \param[in]	param	�p�����[�^�ł�
	*/
	void set_param(Pooling2D_Layer::Param& param);

	/*!
	* \brief	Param���擾���܂�
	* \return	param
	*/
	inline Pooling2D_Layer::Param& param(){ return param_; };

	/*!
	* \brief	delta���擾���܂�
	* \return	delta
	*/
	Sigma_Matrix& delta(){ return delta_; };

	/*!
	* \brief	�������ꂽ�o�͂��擾���܂�
	* \return	d_output
	*/
	Sigma_Matrix& d_output(){ return d_output_; }

	int* idx_matrix(){ return d_idx_matrix; };

	void set_optimizer(Optimizer& opt){ return; };

private:
	//!�p�����[�^�ł�
	Pooling2D_Layer::Param param_;

	//!���`�d�p
	vector<Additional_Process*> Process;

	//!�f���^�s��ł�
	Sigma_Matrix delta_;

	//!*�O�̑w�̏o�͂̔����l���v�Z����̈���m�ۂ��܂�
	Sigma_Matrix d_output_;

	//!�v�[�����O�����C���f�b�N�X��ۑ����邽�߂̔z��(int�^�z��ł�)
	int* d_idx_matrix;
};


#endif