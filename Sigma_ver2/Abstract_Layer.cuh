/*!
* \file    Abstract_Layer.cuh
* \brief   ���ۃN���XAbstract_Layer���`���Ă��܂�
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef ABSTRACT_LAYER_CUH_
#define ABSTRACT_LAYER_CUH_
#undef NDEBUG

#include"Sigma_Matrix.cuh"
#include<string>

/*!
* \brief	Sigma�ŗp������Layer�̒��ۃN���X�ł�
* \details	Sigma�ŗp������Layer�̒��ۃN���X�ł�\n
*			Sigma�ŗp������Layer�͑S��Abstract_Layer���p�����Ă��܂��B\n
*/
class Abstract_Layer{
public:
	/*!
	* \brief   Layer�̏��������o���܂�
	* \param[in]	file_patch	�����o��file_name�ł�
	*/
	virtual void save(std::string file_path) = 0;

	/*!
	* \brief   Layer�̏���ǂݍ��݂܂�
	* \param[in]	file_patch	�ǂݍ���file_name�ł�
	*/
	virtual void load(std::string file_path) = 0;

	/*!
	* \brief   outoput�̎Q�Ƃ��擾���܂�
	* \return	output
	*/
	Sigma_Matrix& output(){ return output_; };

	/*!
	* \brief	�w�̎�ނ��擾���܂�
	* \return	kind_
	*/
	LAYER kind(){ return kind_; };

protected:
	//!�w�̏o�͂ł�
	Sigma_Matrix output_;

	//!�w�̎�ނł�
	LAYER kind_;
};

#endif