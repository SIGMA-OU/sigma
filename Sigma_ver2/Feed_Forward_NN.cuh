/*!
* \file		Feed_Forward_NN.cuh
* \brief	Sigmaで用いる、Feed_Forward_NNを定義しています
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef FEED_FORWARD_NN_CUH_
#define FEED_FORWARD_NN_CUH_
#undef NDEBUG

#include"Input_Layer.cuh"
#include"Fully_Connect_Layer.cuh"
#include"Convolution2D_Layer.cuh"
#include"Optimizer.cuh"
#include"time.h"
#include<vector>
using namespace std;


/*!
* \brief	Sigmaで用いる順伝播ニューラルネットワークを規定しています。
* \details	Sigmaで用いる順伝播ニューラルネットワークを規定しています。\n
*			ニューラルネットワークモデルは、\c Abstract_Layer　を\n
*			持っています。learn関数で学習を行えます。
*/
class Feed_Forward_NN{
public:
	/*!
	* \brief	デフォルトコンストラクタです
	*/
	Feed_Forward_NN();

	/*!
	* \brief	コンストラクタです
	* \param[in]	first_layer	入力層で初期化できます
	*/
	Feed_Forward_NN(Input_Layer& first_layer);

	/*!
	* \brief	層を追加します
	* \param[in]	Layer	追加する層
	*/
	void add_layer(Abstract_Middle_Layer& Layer);

	/*!
	* \brief	学習を行います。
	* \param[in]	input_data	入力データデータです.
	* \param[in]	teacher_data	教師データです.
	* \return	損失関数の値です
	*/
	float learn(Sigma_Matrix& input_data, Sigma_Matrix& teacher_data);

	/*!
	* \brief	順伝播を行います(学習時の順伝播です)
	* \param[in]	input_data	入力データデータです.
	*/
	void forward(Sigma_Matrix* input_data);

	/*!
	* \brief	deltaの逆伝播を行います
	*/
	void backward();

	/*!
	* \brief	重みの更新量を計算します.
	*/
	void calc_update_param();

	/*!
	* \brief	パラメータをアップデートします
	*/
	void update();

	/*!
	* \brief	推論をします(順伝播だけする)。
	* \param[in]	input_data	入力データデータです.
	* \return	最終層の出力です
	*/
	Sigma_Matrix& infer(Sigma_Matrix& input_data);

	/*!
	* \brief	推論をします(順伝播だけする)。
	* \param[in]	input_data	入力データデータです.
	* \param[in]	teacher_data	教師データです.
	* \return	損失関数の値です
	*/
	float infer(Sigma_Matrix& input_data, Sigma_Matrix& teacher_data);

	/*!
	* \brief	損失関数を設定します
	* \param[in]	loss	LOSS_FUNCTION
	*/
	void set_Loss_function(LOSS_FUNCTION loss){ loss_ = loss; };

	/*!
	* \brief	Optimizerを設定します
	* \param[in]	opt	Optimizer
	*/
	void set_optimizer(Optimizer& opt);

	unsigned int training_num(){ return training_num_; };
	NN_MODE mode(){ return mode_; };
	LOSS_FUNCTION loss(){ loss_; };

	/*!
	* \brief	モデルを保存します(未実装・・)
	* \param[in]	dir_path	保存するディレクトリを指定します
	* \param[in]	Output_All_Data	全てのパラメータを保存するか
	*/
	void save(string dir_path, bool Output_All_Data = true){ return; };

private:
	//!学習回数
	unsigned int training_num_;

	//!学習中 or　推論
	NN_MODE mode_;

	//!損失関数
	LOSS_FUNCTION loss_;

	//!最適化手法
	OPTIMIZER opt_;

	//!入力層です.
	Input_Layer* input_;

	//!入力層以外の層です
	vector<Abstract_Middle_Layer*> layers_;

	//!教師データのポインタ
	Sigma_Matrix* teacher_data_;
};

#endif