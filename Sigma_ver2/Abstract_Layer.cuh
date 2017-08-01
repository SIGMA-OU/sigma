/*!
* \file    Abstract_Layer.cuh
* \brief   抽象クラスAbstract_Layerを定義しています
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef ABSTRACT_LAYER_CUH_
#define ABSTRACT_LAYER_CUH_
#undef NDEBUG

#include"Sigma_Matrix.cuh"
#include<string>

/*!
* \brief	Sigmaで用いられるLayerの抽象クラスです
* \details	Sigmaで用いられるLayerの抽象クラスです\n
*			Sigmaで用いられるLayerは全てAbstract_Layerを継承しています。\n
*/
class Abstract_Layer{
public:
	/*!
	* \brief   Layerの情報を書き出します
	* \param[in]	file_patch	書き出すfile_nameです
	*/
	virtual void save(std::string file_path) = 0;

	/*!
	* \brief   Layerの情報を読み込みます
	* \param[in]	file_patch	読み込むfile_nameです
	*/
	virtual void load(std::string file_path) = 0;

	/*!
	* \brief   outoputの参照を取得します
	* \return	output
	*/
	Sigma_Matrix& output(){ return output_; };

	/*!
	* \brief	層の種類を取得します
	* \return	kind_
	*/
	LAYER kind(){ return kind_; };

protected:
	//!層の出力です
	Sigma_Matrix output_;

	//!層の種類です
	LAYER kind_;
};

#endif