/*!
* \file    Additional_Process.cuh
* \brief   �w�̏��`�d���ɒǉ��ł��鉉�Z�����ł��B\n�h���b�v�A�E�g�␳�K���Ȃǂ��s���܂�
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef ADDITIONAL_PROCESS_CUH_
#define ADDITIONAL_PROCESS_CUH_
#undef NDEBUG

#include"Sigma_Matrix.cuh"

/*!
* \brief   �w�̏��`�d���ɒǉ��ł��鉉�Z�������K�肵�Ă鉼�z�N���X�ł��B
* \details �w�̏��`�d���s������̍s�񏈗����s���������ɗp����N���X�ł��B\n
*			�f�[�^�ɑ΂��ăm�C�Y����������A�m�[�h���h���b�v�A�E�g����������ł��܂��B\n
*			�ǂ̂悤�ȏ������s�����ɂ��ẮAADDITIONAL_PROCESS��p���Ďw�肵�܂��B\n
*			�w��additional_process��add���邱�ƂŁA�������s���܂��B\n
*		    �o�͂����s��̑傫���́A�ς��܂���B\n
*/
class Additional_Process{
public:
	/*!
	* \brief   ���͂��ꂽ�s��ɑ΂��ď������s���܂�
	* \param[in]	input	���̓f�[�^
	* \param[in]	mode	�w�K�������_����(TRAIN,INFER)
	* \return  Sigma_matrix�̃|�C���^
	*/
	virtual Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode) = 0;
};

#endif