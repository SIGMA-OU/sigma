/*!
* \file   Noising_Process.cuh
* \brief   �m�C�Y�������s�����߂̃N���XNoising����`����Ă��܂��B(�܂�������)
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef NOISING_PROCESS_CUH_
#define NOISING_PROCESS_CUH_
#undef NDEBUG

#include"Additional_Process.cuh"

/*!
* \brief	�m�C�Y���������s���邽�߂̃N���XNoising���K�肳��Ă��܂��B
* \details	�m�C�Y���������s���邽�߂̃N���XNoising���K�肳��Ă��܂��B\n
*			���z�N���X�ł��� \c Additional_Process�@���p�����Ă���A\n
*			\c Sigma_Layer�@�ɒǉ����邱�ƂŁA���s���邱�Ƃ��ł��܂��B\n
*/
class Noising : public Additional_Process{
public:
	//!�f�t�H���g�R���X�g���N�^�ł�
	Noising(){ rate_ = 0.0f; };

	//!�R�s�[�R���X�g���N�^�ł�
	Noising(const Noising& obj){
		rate_ = obj.rate_;
	};

	enum mode{
		//!�K�E�X���z�ɏ]�����m�C�Y
		GAUSSIAN_NOISE,

		//�}�X�L���O�m�C�Y
		MASKING_NOISE,

		//�Ӗ����m�C�Y
		SALTANDPEPPER
	};

	/*!
	* \brief   ���͂��ꂽ�s��ɑ΂��ăm�C�Y�������܂�
	* \param[in]	input	���͍s��̃|�C���^
	* \param[in]	mode	�w�K�������_����(TRAIN,INFER)
	* \return  �s��
	*/
	Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode = TRAIN);

	/*!
	* \brief   �f�[�^�ɑ΂��ăm�C�Y���|���銄����ݒ肵�܂�
	* \param[in]	val	�m�C�Y���|���銄���ł�
	* \warning	�����͂́A�K��0�`100�̊Ԃɂ��Ă��������B
	* \return  �s��
	*/
	void set_rate(float val){
		if (val < 0 || val >100){
			std::cout << "error : Noising : input rate is wrong" << std::endl;
			return;
		}
		rate_ = val;
		return;
	};

	/*!
	* \brief   �m�C�Y���|���銄�����擾���܂�
	* \return  �m�C�Y���|���銄��
	*/
	float rate(){ return rate_; };

private:
	//!�m�C�Y���[�g
	float rate_;

	//���[�h
	Noising::mode mode_;
};

#endif