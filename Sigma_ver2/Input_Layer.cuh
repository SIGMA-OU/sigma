/*!
* \file    Input_Layer.cuh
* \brief	���͑w�ł���Input_Layer���`���Ă��܂�
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef INPUT_LAYER_CUH_
#define INPUT_LAYER_CUH_
#undef NDEBUG

#include<vector>
#include"Abstract_Layer.cuh"
#include"Additional_Process.cuh"
using namespace std;
/*!
* \brief	Sigma�ŗp��������͑w�N���X�ł�
* \details	Sigma�ŗp��������͑w�N���X�ł�\n
*			\c Abstract_Layer ���p�����Ă��܂��B\n
*			Input_Layer�͏��̎󂯎����Ƃ��ċ@�\���Ă��܂�\n
*			Input_Layer�͓��͂��ꂽ�����R�s�[���܂��B
*/
class Input_Layer : public Abstract_Layer{
public:
	//!�f�t�H���g�R���X�g���N�^�ł�
	Input_Layer();

	/*!
	* \brief   Layer�̏��������o���܂�(������)
	* \param[in]	file_patch	�����o��file_name�ł�
	*/
	void save(string file_path);

	/*!
	* \brief   Layer�̏���ǂݍ��݂܂�(������)
	* \param[in]	file_patch	�ǂݍ���file_name�ł�
	*/
	void load(string file_path);

	/*!
	* \brief   ���͑w�����������܂�
	* \param[in]	input	���̓f�[�^
	* \return  &output_
	*/
	Sigma_Matrix* initialize(Sigma_Matrix* input);

	/*!
	* \brief   �o�b�`�T�C�Y���ω������Ƃ��ɁA�e�w�̏o�͍s��̃T�C�Y��ύX���܂�
	* \param[in]	input	���̓f�[�^
	* \return  &output_
	*/
	Sigma_Matrix* change_batchsize(Sigma_Matrix* input);

	/*!
	* \brief   ���`�d���܂�
	* \param[in]	input	���͂���z��̃|�C���^�ł�
	*/
	Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode = TRAIN);

	/*!
	* \brief   output�̃|�C���^���擾���܂�
	* \return	&output
	*/
	Sigma_Matrix* get_output(){ return &output_; };

	/*!
	* \brief   Additional_Process��ǉ����܂��B
	* \param[in]	ap	Additinal�QProcess�ł��B
	*/
	void add_process(Additional_Process& ap);

private:
	//!�ǉ�����鉉�Z�����ł�
	vector<Additional_Process*> Process;
};

#endif