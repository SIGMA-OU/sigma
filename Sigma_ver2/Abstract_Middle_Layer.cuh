/*!
* \file    Abstract_Middle_Layer.cuh
* \brief   抽象クラスAbstract_Middle_Layerを定義しています
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
* \brief	Sigmaで用いられるInput_Layer以外の抽象クラスです
* \details	Sigmaで用いられるInput_Layer以外の抽象クラスです\n
*/

class Abstract_Middle_Layer : public Abstract_Layer{
public:

	/*!
	* \brief   順伝播します
	* \param[in]	input	入力行列です
	* \param[in]	mode	学習中か、推論中か
	* \return	output
	*/
	virtual Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode) = 0;

	/*!
	* \brief   初期化します
	* \param[in]	prev_output	前の層の出力です
	* \param[in]	output
	*/
	virtual Sigma_Matrix* initialize(Sigma_Matrix* prev_output) = 0;

	/*!
	* \brief   バッチサイズを変更したときに、各層の出力の大きさを変更します
	* \param[in]	prev_output	前の層の出力です
	* \return	output
	*/
	virtual Sigma_Matrix* change_batchsize(Sigma_Matrix* prev_output) = 0;

	/*!
	* \brief   出力を微分した値を返します
	* \return	d_output
	*/
	virtual Sigma_Matrix* get_d_output() = 0;

	/*!
	* \brief	学習に必要な\c Optimizer を初期化します
	* \param[in]	prev_output
	* \return	*output
	*/
	virtual Sigma_Matrix* initialize_optimizer(Sigma_Matrix* prev_output) = 0;

	/*!
	* \brief   逆伝播します
	* \param[in]	prev_delta	前の層のデルタ行列です
	* \param[in]	prev_d_output	前の層の微分された出力です
	*/
	virtual void back_ward(Sigma_Matrix* prev_delta, Sigma_Matrix* prev_d_output) = 0;

	/*!
	* \brief   Optimizerを用いて、パラメータの変更量を計算します
	*/
	virtual Sigma_Matrix* calc_update_param(Sigma_Matrix* prev_output) = 0;

	/*!
	* \brief   パラメータを変更します
	*/
	virtual void update_param() = 0;

	/*!
	* \brief	deltaを取得します
	* \return	delta
	*/
	virtual Sigma_Matrix* get_delta() = 0;

	/*!
	* \brief   Optimizerをセットします
	* \param	opt	Optimizerです
	*/
	virtual void  set_optimizer(Optimizer& opt) = 0;

	/*!
	* \brief   outputのポインタを返します
	* \return	&output
	*/
	Sigma_Matrix* get_output(){ return &output_; };

};

#endif