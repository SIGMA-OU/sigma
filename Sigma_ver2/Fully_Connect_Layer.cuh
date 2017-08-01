/*!
* \file    Fully_Connect_Layer.cuh
* \brief	�S�����w���`���Ă��܂��B
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef FULLY_CONNECT_LAYER_CUH_
#define FULLY_CONNECT_LAYER_CUH_
#undef NDEBUG

#include"Abstract_Middle_Layer.cuh"
#include"Optimizer.cuh"
#include"Additional_Process.cuh"
#include"Activate_Function.cuh"
#include"Utilities.cuh"
#include<vector>
using namespace std;

class Fully_Connect_Layer : public Abstract_Middle_Layer{
public:

	//�S�����w���K�肷��p�����[�^�ł�
	struct Param{
		//!�m�[�h���ł�
		unsigned int nodes;
		//!�������֐��ł�
		ACTIVATE_FUNCTION func;

		//!����������Ƃ��̏d�݂̕��ϒl�ł�
		float mu_;

		//!����������Ƃ��̏d�݂̕��U�ł�
		float stddev_;

		//!����������Ƃ��̃o�C�A�X�̏����l
		float init_bias_;
	};

	/*!
	* \brief	�f�t�H���g�R���X�g���N�^�ł�
	*/
	Fully_Connect_Layer();

	/*!
	* \brief	�R���X�g���N�^�ł�
	* \param[in]	���͑w�̃p�����[�^�ł�
	*/
	Fully_Connect_Layer(Fully_Connect_Layer::Param &param);

	/*!
	* \brief	�R���X�g���N�^�ł�
	* \param[in]	node_num	���͑w�̃p�����[�^�ł�
	* \param[in]	func	�������֐�
	* \param[in]	mu	����������Ƃ��̕��ϒl
	* \param[in]	stddev	����������Ƃ��̕��U
	*/
	Fully_Connect_Layer(int node_num,ACTIVATE_FUNCTION func, float mu = 0, float stddev = 0.1f,float init_bias = 0);

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
	Sigma_Matrix* get_delta(){ return &delta_; };

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
	* \brief   Optimizer��p���āA�p�����[�^�̕ύX�ʂ��v�Z���܂�
	*/
	Sigma_Matrix* calc_update_param(Sigma_Matrix* prev_output);

	/*!
	* \brief   �p�����[�^��ύX���܂�
	*/
	void update_param();

	/*!
	* \brief   Additional_Process��ǉ����܂��B
	* \param[in]	ap	Additional_Process
	*/
	void add_process(Additional_Process& ap);

	//!�������ł�
	void save(string file_path){ return; };

	//!�������ł�
	void load(string file_path){ return; };

	//////////////////////// Getter / Setter  //////////////////////////

	/*!
	* \brief	�d�ݍs���ݒ肵�܂�
	* \param[in]	weight	�d�ݍs��ł�
	*/
	void set_weight(Sigma_Matrix& weight){ weight_ = weight; };

	/*!
	* \brief	�o�C�A�X��ݒ肵�܂�
	* \param[in]	bias	�o�C�A�X�s��ł�
	*/
	void set_bias(Sigma_Matrix& bias){ bias_ = bias; };

	/*!
	* \brief	Optimizer��ݒ肵�܂�
	* \param[in]	optimimzer	optimizer�ł�
	*/
	void set_optimizer(Optimizer& optimizer){ optimizer_ = optimizer; };

	/*!
	* \brief	Optimizer��ݒ肵�܂�
	* \param[in]	param	�p�����[�^�ł�
	*/
	void set_param(Fully_Connect_Layer::Param param){ param_ = param; };

	/*!
	* \brief	�d�ݍs����擾���܂�
	* \return	�d�ݍs��
	*/
	Sigma_Matrix& weight(){ return weight_; };

	/*!
	* \brief	�o�C�A�X���擾���܂�
	* \return	�o�C�A�X
	*/
	Sigma_Matrix& bias(){ return bias_; };

	/*!
	* \brief	������Ԃ��擾���܂�
	* \return	�������
	*/
	Sigma_Matrix& net(){ return net_; };

	/*!
	* \brief	Param���擾���܂�
	* \return	param
	*/
	Fully_Connect_Layer::Param& param(){ return param_; };

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

	/*!
	* \brief	�d�݂̍X�V�s����擾���܂�
	* \return	�d�݂̍X�V�s��
	*/
	Sigma_Matrix& update_weight(){ return update_weight_; };

	/*!
	* \brief	�o�C�A�X�̍X�V�s����擾���܂�
	* \return	�o�C�A�X�̍X�V�s��
	*/
	Sigma_Matrix& update_bias(){ return update_bias_; };

	/*!
	* \brief	Optimizer���擾���܂�
	* \return	optimimzer_
	*/
	Optimizer& optimizer(){ return optimizer_; };

private:
	//!�p�����[�^�ł�
	Fully_Connect_Layer::Param param_;

	//!Optimizer�ł�
	Optimizer optimizer_;

	//!���`�d�p
	vector<Additional_Process*> Process;

	//!�o�C�A�X�s��ł�
	Sigma_Matrix bias_;

	//!�d�ݍs��ł�
	Sigma_Matrix weight_;

	//!������Ԃł�
	Sigma_Matrix net_;

	//////////////////////////////////�p�����[�^�X�V�p

	//!�f���^�s��ł�
	Sigma_Matrix delta_;

	//!*�O�̑w�̏o�͂̔����l���v�Z����̈���m�ۂ��܂�
	Sigma_Matrix d_output_;

	//!�d�݂̍X�V��
	Sigma_Matrix update_weight_;

	//!�o�C�A�X�̍X�V��
	Sigma_Matrix update_bias_;

	//!�o�C�A�X�̌v�Z�ɗp����P�ʗ�s��ł�
	Sigma_Matrix CONSTANT_1_;
};

#endif