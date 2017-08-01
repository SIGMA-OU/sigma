/*!
* \file    Data_Set.cuh
* \brief	�f�[�^���i�[����N���X�ł�
* \date    2016/07/15
* \author  Asatani Satoshi(@exaintelligence)
*/

#ifndef DATA_SET_CUH_
#define DATA_SET_CUH_
#undef NDEBUG

#include"Sigma_Matrix.cuh"

/*!
* \brief	�f�[�^���ȒP�Ɉ������߂̃N���XData_Set�N���X���`���Ă��܂�
* \details	�f�[�^���ȒP�Ɉ������߂̃N���XData_Set�N���X���`���Ă��܂�\n
*			�f�[�^�S�Ă�data_�Ɋi�[����Ă���\n
*			�~�j�o�b�`�������I�ɍ쐬���邱�Ƃ��ł��܂��B\n
*			�f�[�^��add�Œǉ����邱�Ƃ��ł��܂��B\n
*			�����I�ɋ��t�f�[�^�̍쐬���\�ł�\n
*/
class SL_Data_Set{
public:
	//!�f�t�H���g�R���X�g���N�^�ł�
	SL_Data_Set();

	/*!
	* \brief	�R���X�g���N�^�ł�
	* \param[in]	data	�f�o�C�X�}�g���b�N�X�ŏ������ł��܂�
	* \warning	�o�b�`�T�C�Y�ȊO�̃f�[�^�T�C�Y������������āA�Ȍ�A�����T�C�Y�̃f�[�^����������Ȃ��Ȃ�܂��B
	*/
	SL_Data_Set(Device_Matrix& data);

	/*!
	* \brief	�R���X�g���N�^�ł�
	* \param[in]	data	�f�o�C�X�}�g���b�N�X�ŏ������ł��܂�
	* \param[in]	teacher	data�ɑΉ�����teacher�ł��B
	* \warning	�o�b�`�T�C�Y�ȊO�̃f�[�^�T�C�Y������������āA�Ȍ�A�����T�C�Y�̃f�[�^����������Ȃ��Ȃ�܂��B
	*/
	SL_Data_Set(Device_Matrix& data,Device_Matrix& teacher);

	/*!
	* \brief	�R���X�g���N�^�ł��B�@\c Sigma_Matrix �̃o�b�`�T�C�Y�������f�[�^���������܂�
	* \param[in]	data	Sigma_Matrix�ł��B
	* \warning	�o�b�`�T�C�Y�ȊO�̃f�[�^�T�C�Y������������āA�Ȍ�A�����T�C�Y�̃f�[�^����������Ȃ��Ȃ�܂��B
	*/
	SL_Data_Set(Sigma_Matrix& data);

	/*!
	* \brief	�R���X�g���N�^�ł��B�@\c Sigma_Matrix �̃o�b�`�T�C�Y�������f�[�^���������܂�
	* \param[in]	data	Sigma_Matrix�ł��B
	* \param[in]	teacher	data�ɑΉ�����teacher�ł�
	* \warning	�o�b�`�T�C�Y�ȊO�̃f�[�^�T�C�Y������������āA�Ȍ�A�����T�C�Y�̃f�[�^����������Ȃ��Ȃ�܂��B
	*/
	SL_Data_Set(Sigma_Matrix& data,Sigma_Matrix& teacher);

	Sigma_Matrix& make_teacher();

	Sigma_Matrix& next(unsigned int data_num);

	Sigma_Matrix& next_teacher(unsigned int data_num);

private:
	//!�f�[�^�̔ԍ����w�肵�Ă���C���f�b�N�X�ł��Bnext�֐��Ŏg�p�����܂�
	unsigned int idx;

	//!�P���������̓e�X�g�f�[�^�������Ă��܂�
	Sigma_Matrix data_;

	//!���t�������͐����f�[�^�������Ă��܂�(�ʂɎg��Ȃ����Ƃ�����܂��B)
	Sigma_Matrix teacher_;

	//!�O���ɃA�N�Z�X�ł���}�g���b�N�X��
	Sigma_Matrix divide_data_;

	//training_data��vector�^�ŃX�g�b�N���܂�
	vector<Sigma_Matrix> training_set_;

	//teacher_data��vector�^�ŃX�g�b�N���܂��B
	vector<Sigma_Matrix> teacher_set_;

	//�N���X�ɑΉ��������O��ݒ肷�邱�Ƃ��ł��܂��B
	vector<string> tag_set;

	//�e�f�[�^�ɑ΂���label��ێ��ł��܂��B
	vector<vector<int>> label_set;
};



#endif