/*!
* \file    Sigma_Evaluation_Function.cuh
* \brief   Sigma_Matrixを用いた便利な関数郡です
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
* \brief   各行の最大値のindexを格納します
* \param[in]	matrix	Sigma_Matrixです
* \return 最大値を表すidxをvector<int>の配列で返します。
*/
vector<int> argmax_idx(Sigma_Matrix& matrix);

/*!
* \brief   二つの配列の中で、同じ数字の個数を返します。多クラス分類問題でよく使います。
* \param[in]	label	int型配列
* \param[in]	correct int型配列
* \return 同じ数字の個数
*/
int equall(vector<int>& label,vector<int>& correct);

/*!
* \brief   ラベル情報から、ONE_HOTな配列を作成します。
* \param[in]	label	int型配列
* \param[in]	class_num	全クラスの数です
* \return　Onehotな行列(列内でひとつだけ1を取っています。) 
*/
Sigma_Matrix make_onehot_matrix(vector<int>& label_data, unsigned int class_num);

/*!
* \brief	Sigma_Matrix内のデータを小分けにしまたミニバッチを作成します。
* \param[in]	size	ミニバッチのサイズです
* \param[in]	idx	ミニバッチのインデックスです。(dataのsize * idx番からミニバッチサイズ分データを取り出します。)
* \param[in]	src_matrix	元マトリックス
* \param[in]	minibatch_matrix 作成するマトリックス
* \return　Onehotな行列(列内でひとつだけ1を取っています。)
*/
void next_batch(unsigned int size,unsigned int idx,Sigma_Matrix& src_matrix,Sigma_Matrix& mini_batch_matrix);

/*!
* \brief   ミニバッチを作成します。
* \param[in]	src_matrix	元になる行列です。
* \param[in]	minibatch_size	ミニバッチの大きさです
* \param[in]	minibatch_num	ミニバッチの個数です
* \param[in]	RANDOM_CLIP	ミニバッチの作成方法です
* \return　Onehotな行列(列内でひとつだけ1を取っています。)
*/
//vector<Sigma_Matrix> make_mini_batch(Sigma_Matrix& src_matrix,unsigned int minibatch_size,unsigned int minibatch_num,bool RANDOM_CLIP = true);

/*!
* \brief   クラス毎に分けられたSigmaMatrix配列を用いてミニバッチを作成します。
* \param[in]	src_matrix	元になるマトリックス郡です。
* \param[in]	minibatch_size	ミニバッチの大きさです
* \param[in]	minibatch_num	ミニバッチの個数です
* \param[in]	RANDOM_CLIP	ミニバッチの作成方法です
* \return　Onehotな行列(列内でひとつだけ1を取っています。)
*/
//vector<Sigma_Matrix> make_mini_batch(vector<Sigma_Matrix>& src_matrixs, unsigned int minibatch_size, unsigned int minibatch_num,bool RANDOM_CLIP = true);

#endif
