/*!
* \file    Convolution2D_Layer.cuh
* \brief	2�����̏�ݍ��ݑw���`���Ă��܂�
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/
#ifndef CONVOLUTION2D_LAYER_CUH_
#define CONVOLUTION2D_LAYER_CUH_
#undef NDEBUG

#include"Abstract_Middle_Layer.cuh"
#include"Optimizer.cuh"
#include"Additional_Process.cuh"
#include"Activate_Function.cuh"
#include"Utilities.cuh"
#include<vector>
using namespace std;

class Convolution2D_Layer : public Abstract_Middle_Layer{
public:
	//!��ݍ��ݑw���K�肷��p�����[�^�ł�
	struct Param{
		//!��ݍ��݃t�B���^�̕�,����,�t�B���^��,�X�g���C�h�ł�
		unsigned int width, height, filter_num, stride;
		//!�������֐��ł�
		ACTIVATE_FUNCTION func;
		//!�p�f�B���O�����邩�ǂ����ł�
		bool padding;

		//!�������̂Ƃ��ɗp����t�B���^�̕��ϒl�ł�
		float mu_;

		//!�������̎��ɗp����t�B���^�̕��U�l�ł�
		float stddev_;

		//!�������̎��ɗp����o�C�A�X�̒l�ł�
		float init_bias_;
	};
	//!�f�t�H���g�R���X�g���N�^�ł�
	Convolution2D_Layer();

	Convolution2D_Layer(Convolution2D_Layer::Param& param);

	Convolution2D_Layer(unsigned int width,unsigned int height,unsigned int filter_num,unsigned int stride, bool padding,ACTIVATE_FUNCTION func = relu,float mu = 0,float stddev = 0.05f,float init_bias = 0);

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
	* \brief	�w�K�ɕK�v��\c Optimizer �����������܂�
	* \param[in]	prev_output
	* \return	*output
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
	* \brief	delta���擾���܂�
	* \return	delta
	*/
	Sigma_Matrix* get_delta(){ return &delta_; };

	/*!
	* \brief   output�̃|�C���^��Ԃ��܂�
	* \return	&output
	*/
	Sigma_Matrix* get_output(){ return &output_; };

	void save(std::string file_path){ return; };

	void load(std::string file_path){ return; };

	/////////////////// setter  /////////////////
	/*!
	* \brief	Optimizer��ݒ肵�܂�
	* \param[in]	optimimzer	optimizer�ł�
	*/
	inline void set_optimizer(Optimizer& opt){ optimizer_ = opt; };

	////////////////////geter  /////////////////////////
	/*!
	* \brief	�p�����[�^�[���擾���܂�
	* \return	param_
	*/
	inline Convolution2D_Layer::Param& param(){ return param_; };

	/*!
	* \brief	optimizer���擾���܂�
	* \return	param_
	*/
	inline Optimizer optimezer(){ return optimizer_; };

	/*!
	* \brief	�o�C�A�X���擾���܂�
	* \return	bais_
	*/
	inline Sigma_Matrix& bias(){ return bias_; };

	/*!
	* \brief	�t�B���^�[���擾���܂�
	* \return	filter_
	*/
	inline Sigma_Matrix& filter(){ return filter_; };

	/*!
	* \brief	net�l���擾���܂�
	* \return	net_
	*/
	inline Sigma_Matrix& net(){ return net_; };

	/*!
	* \brief	���̓f�[�^����בւ����f�[�^
	* \return	transe_input_
	*/
	inline Sigma_Matrix& transe_input(){ return transe_input_; };

	/*!
	* \brief	�f���^�s������擾���܂�
	* \return	delta_
	*/
	inline Sigma_Matrix& delta(){ return delta_; };

	/*!
	* \brief	transe_delta_���擾���܂�
	* \return	transe_delta_
	*/
	inline Sigma_Matrix& transe_delta(){ return transe_delta_; };

	/*!
	* \brief	�����o�͂��擾���܂�
	* \return	d_outpout_
	*/
	inline Sigma_Matrix& d_output(){ return d_output_; };

	/*!
	* \brief	�t�B���^�̍X�V�s����擾���܂�
	* \return	update_filter_
	*/
	inline Sigma_Matrix& update_filter(){ return update_filter_; };

	/*!
	* \brief	�o�C�A�X�̍X�V�s����擾���܂�
	* \return	update_bias_
	*/
	inline Sigma_Matrix& update_bias(){ return update_bias_; };


private:
	//!�p�����[�^�ł�
	Convolution2D_Layer::Param param_;

	//!Optimizer�ł�
	Optimizer optimizer_;

	//!���`�d�p
	vector<Additional_Process*> Process;

	//!�o�C�A�X�s��ł�
	Sigma_Matrix bias_;

	//!�t�B���^�[�s��ł�
	Sigma_Matrix filter_;

	//!������Ԃł�
	Sigma_Matrix net_;

	//!���̓f�[�^����בւ����f�[�^�ł�
	Sigma_Matrix transe_input_;

	//////////////////////////////////////

	//!�f���^�s��ł�
	Sigma_Matrix delta_;

	//�t�`�d�̎��Ɏg���܂��B
	Sigma_Matrix transe_delta_;

	//!*�O�̑w�̏o�͂̔����l���v�Z����̈���m�ۂ��܂�
	Sigma_Matrix d_output_;

	//!filter�̍X�V�ʂł�
	Sigma_Matrix update_filter_;

	//!�o�C�A�X�̍X�V��
	Sigma_Matrix update_bias_;

	//!�o�C�A�X�̌v�Z�ɗp����P�ʗ�s��ł�
	Sigma_Matrix CONSTANT_1_;
};

#endif