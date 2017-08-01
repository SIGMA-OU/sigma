/*!
* \file    Additional_Process.cuh
* \brief   層の順伝播時に追加できる演算処理です。\nドロップアウトや正規化などを行えます
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef ADDITIONAL_PROCESS_CUH_
#define ADDITIONAL_PROCESS_CUH_
#undef NDEBUG

#include"Sigma_Matrix.cuh"

/*!
* \brief   層の順伝播時に追加できる演算処理を規定してる仮想クラスです。
* \details 層の順伝播を行った後の行列処理を行いたい時に用いるクラスです。\n
*			データに対してノイズをかけたり、ノードをドロップアウトさせたたりできます。\n
*			どのような処理を行うかについては、ADDITIONAL_PROCESSを用いて指定します。\n
*			層のadditional_processにaddすることで、処理を行えます。\n
*		    出力される行列の大きさは、変わりません。\n
*/
class Additional_Process{
public:
	/*!
	* \brief   入力された行列に対して処理を行います
	* \param[in]	input	入力データ
	* \param[in]	mode	学習中か推論中か(TRAIN,INFER)
	* \return  Sigma_matrixのポインタ
	*/
	virtual Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode) = 0;
};

#endif