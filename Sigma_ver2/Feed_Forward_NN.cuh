/*!
* \file		Feed_Forward_NN.cuh
* \brief	Sigma�ŗp����AFeed_Forward_NN���`���Ă��܂�
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef FEED_FORWARD_NN_CUH_
#define FEED_FORWARD_NN_CUH_
#undef NDEBUG

#include"Input_Layer.cuh"
#include"Fully_Connect_Layer.cuh"
#include"Convolution2D_Layer.cuh"
#include"Optimizer.cuh"
#include"time.h"
#include<vector>
using namespace std;


/*!
* \brief	Sigma�ŗp���鏇�`�d�j���[�����l�b�g���[�N���K�肵�Ă��܂��B
* \details	Sigma�ŗp���鏇�`�d�j���[�����l�b�g���[�N���K�肵�Ă��܂��B\n
*			�j���[�����l�b�g���[�N���f���́A\c Abstract_Layer�@��\n
*			�����Ă��܂��Blearn�֐��Ŋw�K���s���܂��B
*/
class Feed_Forward_NN{
public:
	/*!
	* \brief	�f�t�H���g�R���X�g���N�^�ł�
	*/
	Feed_Forward_NN();

	/*!
	* \brief	�R���X�g���N�^�ł�
	* \param[in]	first_layer	���͑w�ŏ������ł��܂�
	*/
	Feed_Forward_NN(Input_Layer& first_layer);

	/*!
	* \brief	�w��ǉ����܂�
	* \param[in]	Layer	�ǉ�����w
	*/
	void add_layer(Abstract_Middle_Layer& Layer);

	/*!
	* \brief	�w�K���s���܂��B
	* \param[in]	input_data	���̓f�[�^�f�[�^�ł�.
	* \param[in]	teacher_data	���t�f�[�^�ł�.
	* \return	�����֐��̒l�ł�
	*/
	float learn(Sigma_Matrix& input_data, Sigma_Matrix& teacher_data);

	/*!
	* \brief	���`�d���s���܂�(�w�K���̏��`�d�ł�)
	* \param[in]	input_data	���̓f�[�^�f�[�^�ł�.
	*/
	void forward(Sigma_Matrix* input_data);

	/*!
	* \brief	delta�̋t�`�d���s���܂�
	*/
	void backward();

	/*!
	* \brief	�d�݂̍X�V�ʂ��v�Z���܂�.
	*/
	void calc_update_param();

	/*!
	* \brief	�p�����[�^���A�b�v�f�[�g���܂�
	*/
	void update();

	/*!
	* \brief	���_�����܂�(���`�d��������)�B
	* \param[in]	input_data	���̓f�[�^�f�[�^�ł�.
	* \return	�ŏI�w�̏o�͂ł�
	*/
	Sigma_Matrix& infer(Sigma_Matrix& input_data);

	/*!
	* \brief	���_�����܂�(���`�d��������)�B
	* \param[in]	input_data	���̓f�[�^�f�[�^�ł�.
	* \param[in]	teacher_data	���t�f�[�^�ł�.
	* \return	�����֐��̒l�ł�
	*/
	float infer(Sigma_Matrix& input_data, Sigma_Matrix& teacher_data);

	/*!
	* \brief	�����֐���ݒ肵�܂�
	* \param[in]	loss	LOSS_FUNCTION
	*/
	void set_Loss_function(LOSS_FUNCTION loss){ loss_ = loss; };

	/*!
	* \brief	Optimizer��ݒ肵�܂�
	* \param[in]	opt	Optimizer
	*/
	void set_optimizer(Optimizer& opt);

	unsigned int training_num(){ return training_num_; };
	NN_MODE mode(){ return mode_; };
	LOSS_FUNCTION loss(){ loss_; };

	/*!
	* \brief	���f����ۑ����܂�(�������E�E)
	* \param[in]	dir_path	�ۑ�����f�B���N�g�����w�肵�܂�
	* \param[in]	Output_All_Data	�S�Ẵp�����[�^��ۑ����邩
	*/
	void save(string dir_path, bool Output_All_Data = true){ return; };

private:
	//!�w�K��
	unsigned int training_num_;

	//!�w�K�� or�@���_
	NN_MODE mode_;

	//!�����֐�
	LOSS_FUNCTION loss_;

	//!�œK����@
	OPTIMIZER opt_;

	//!���͑w�ł�.
	Input_Layer* input_;

	//!���͑w�ȊO�̑w�ł�
	vector<Abstract_Middle_Layer*> layers_;

	//!���t�f�[�^�̃|�C���^
	Sigma_Matrix* teacher_data_;
};

#endif