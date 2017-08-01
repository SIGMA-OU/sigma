/*!
* \file    Sigma_Evaluation_Function.cuh
* \brief   Sigma_Matrix��p�����֗��Ȋ֐��S�ł�
* \date    2016/07/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef SIGMA_EVALUATION_FUNCTION_CUH_
#define SIGMA_EVALUATION_FUNCTION_CUH_
#undef NDEBUG

#include<vector>
#include"Sigma_Matrix.cuh"
#include"cublas_v2.h"

using namespace std;

/*!
* \brief   �e�s�̍ő�l��index���i�[���܂�
* \param[in]	matrix	Sigma_Matrix�ł�
* \return �ő�l��\��idx��vector<int>�̔z��ŕԂ��܂��B
*/
vector<int> argmax_idx(Sigma_Matrix& matrix);

/*!
* \brief   ��̔z��̒��ŁA���������̌���Ԃ��܂��B���N���X���ޖ��ł悭�g���܂��B
* \param[in]	label	int�^�z��
* \param[in]	correct int�^�z��
* \return ���������̌�
*/
int equall(vector<int>& label,vector<int>& correct);

/*!
* \brief   ���x����񂩂�AONE_HOT�Ȕz����쐬���܂��B
* \param[in]	label	int�^�z��
* \param[in]	class_num	�S�N���X�̐��ł�
* \return�@Onehot�ȍs��(����łЂƂ���1������Ă��܂��B) 
*/
Sigma_Matrix make_onehot_matrix(vector<int>& label_data, unsigned int class_num);

/*!
* \brief	Sigma_Matrix���̃f�[�^���������ɂ��܂��~�j�o�b�`���쐬���܂��B
* \param[in]	size	�~�j�o�b�`�̃T�C�Y�ł�
* \param[in]	idx	�~�j�o�b�`�̃C���f�b�N�X�ł��B(data��size * idx�Ԃ���~�j�o�b�`�T�C�Y���f�[�^�����o���܂��B)
* \param[in]	src_matrix	���}�g���b�N�X
* \param[in]	minibatch_matrix �쐬����}�g���b�N�X
* \return�@Onehot�ȍs��(����łЂƂ���1������Ă��܂��B)
*/
void next_batch(unsigned int size,unsigned int idx,Sigma_Matrix& src_matrix,Sigma_Matrix& mini_batch_matrix);

/*!
* \brief   �~�j�o�b�`���쐬���܂��B
* \param[in]	src_matrix	���ɂȂ�s��ł��B
* \param[in]	minibatch_size	�~�j�o�b�`�̑傫���ł�
* \param[in]	minibatch_num	�~�j�o�b�`�̌��ł�
* \param[in]	RANDOM_CLIP	�~�j�o�b�`�̍쐬���@�ł�
* \return�@Onehot�ȍs��(����łЂƂ���1������Ă��܂��B)
*/
//vector<Sigma_Matrix> make_mini_batch(Sigma_Matrix& src_matrix,unsigned int minibatch_size,unsigned int minibatch_num,bool RANDOM_CLIP = true);

/*!
* \brief   �N���X���ɕ�����ꂽSigmaMatrix�z���p���ă~�j�o�b�`���쐬���܂��B
* \param[in]	src_matrix	���ɂȂ�}�g���b�N�X�S�ł��B
* \param[in]	minibatch_size	�~�j�o�b�`�̑傫���ł�
* \param[in]	minibatch_num	�~�j�o�b�`�̌��ł�
* \param[in]	RANDOM_CLIP	�~�j�o�b�`�̍쐬���@�ł�
* \return�@Onehot�ȍs��(����łЂƂ���1������Ă��܂��B)
*/
//vector<Sigma_Matrix> make_mini_batch(vector<Sigma_Matrix>& src_matrixs, unsigned int minibatch_size, unsigned int minibatch_num,bool RANDOM_CLIP = true);

#endif
