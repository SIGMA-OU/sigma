/*!
* \file    Abstract_Middle_Layer.cuh
* \brief   ���ۃN���XAbstract_Middle_Layer���`���Ă��܂�
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef ABSTRACT_MIDDLE_LAYER_CUH_
#define ABSTRACT_MIDDLE_LAYER_CUH_
#undef NDEBUG


#include"Abstract_Layer.cuh"
#include"Optimizer.cuh"
#include<string>

/*!
* \brief	Sigma�ŗp������Input_Layer�ȊO�̒��ۃN���X�ł�
* \details	Sigma�ŗp������Input_Layer�ȊO�̒��ۃN���X�ł�\n
*/

class Abstract_Middle_Layer : public Abstract_Layer{
public:

	/*!
	* \brief   ���`�d���܂�
	* \param[in]	input	���͍s��ł�
	* \param[in]	mode	�w�K�����A���_����
	* \return	output
	*/
	virtual Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode) = 0;

	/*!
	* \brief   ���������܂�
	* \param[in]	prev_output	�O�̑w�̏o�͂ł�
	* \param[in]	output
	*/
	virtual Sigma_Matrix* initialize(Sigma_Matrix* prev_output) = 0;

	/*!
	* \brief   �o�b�`�T�C�Y��ύX�����Ƃ��ɁA�e�w�̏o�͂̑傫����ύX���܂�
	* \param[in]	prev_output	�O�̑w�̏o�͂ł�
	* \return	output
	*/
	virtual Sigma_Matrix* change_batchsize(Sigma_Matrix* prev_output) = 0;

	/*!
	* \brief   �o�͂���������l��Ԃ��܂�
	* \return	d_output
	*/
	virtual Sigma_Matrix* get_d_output() = 0;

	/*!
	* \brief	�w�K�ɕK�v��\c Optimizer �����������܂�
	* \param[in]	prev_output
	* \return	*output
	*/
	virtual Sigma_Matrix* initialize_optimizer(Sigma_Matrix* prev_output) = 0;

	/*!
	* \brief   �t�`�d���܂�
	* \param[in]	prev_delta	�O�̑w�̃f���^�s��ł�
	* \param[in]	prev_d_output	�O�̑w�̔������ꂽ�o�͂ł�
	*/
	virtual void back_ward(Sigma_Matrix* prev_delta, Sigma_Matrix* prev_d_output) = 0;

	/*!
	* \brief   Optimizer��p���āA�p�����[�^�̕ύX�ʂ��v�Z���܂�
	*/
	virtual Sigma_Matrix* calc_update_param(Sigma_Matrix* prev_output) = 0;

	/*!
	* \brief   �p�����[�^��ύX���܂�
	*/
	virtual void update_param() = 0;

	/*!
	* \brief	delta���擾���܂�
	* \return	delta
	*/
	virtual Sigma_Matrix* get_delta() = 0;

	/*!
	* \brief   Optimizer���Z�b�g���܂�
	* \param	opt	Optimizer�ł�
	*/
	virtual void  set_optimizer(Optimizer& opt) = 0;

	/*!
	* \brief   output�̃|�C���^��Ԃ��܂�
	* \return	&output
	*/
	Sigma_Matrix* get_output(){ return &output_; };

};

#endif