/*!
* \file   Dropout_Process.cuh
* \brief   �h���b�v�A�E�g�������s�����߂̃N���XDropout����`����Ă��܂��B(�܂�������)
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef DROPOUT_PROCESS_CUH_
#define DROPOUT_PROCESS_CUH_
#undef NDEBUG

#include"Additional_Process.cuh"

/*!
* \brief	Dropout�����s���邽�߂̃N���XDropout���K�肳��Ă��܂��B
* \details	Dropout�����s���邽�߂̃N���XDropout���K�肳��Ă��܂��B\n
*			���z�N���X�ł��� \c Additional_Process�@���p�����Ă���A\n
*			\c Sigma_Layer�@�ɒǉ����邱�ƂŁA���s���邱�Ƃ��ł��܂��B\n
*/
class Dropout : public Additional_Process{
public:
	//!�f�t�H���g�R���X�g���N�^�ł�
	Dropout(){ rate_ = 0.0f; };

	//!�R�s�[�R���X�g���N�^�ł�
	Dropout(const Dropout& obj){
		rate_ = obj.rate_;
	}

	/*!
	* \brief   ���͂��ꂽ�s��ɑ΂���Dropout�}�X�N���|�����킹�܂�
	* \param[in]	input	���͍s��̃|�C���^
	* \param[in]	mode	�w�K�������_����(TRAIN,INFER)
	* \return  �s��
	*/
	Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode = TRAIN);

	/*!
	* \brief   �h���b�v�A�E�g���[�g��ݒ肵�܂�
	* \param[in]	val	�h���b�v�A�E�g���[�g�ł�
	* \warning	�h���b�v�A�E�g���[�g�́A�K��0�`100�̊Ԃɂ��Ă��������B
	* \return  �s��
	*/
	void set_rate(float val){
		if (val < 0 || val >100){
			std::cout << "error : Dropout : input rate is wrong" << std::endl;
			return;
		}
		rate_ = val;
		return;
	};

	/*!
	* \brief   �h���b�v�A�E�g���[�g���擾���܂�
	* \return  �h���b�v�A�E�g���[�g
	*/
	float rate(){ return rate_; };

private:
	//!�h���b�v�A�E�g���[�g
	float rate_;
};

#endif