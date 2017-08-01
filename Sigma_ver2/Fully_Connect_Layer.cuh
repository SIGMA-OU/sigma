/*!
* \file    Fully_Connect_Layer.cuh
* \brief	全結合層を定義しています。
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef FULLY_CONNECT_LAYER_CUH_
#define FULLY_CONNECT_LAYER_CUH_
#undef NDEBUG

#include"Abstract_Middle_Layer.cuh"
#include"Optimizer.cuh"
#include"Additional_Process.cuh"
#include"Activate_Function.cuh"
#include"Utilities.cuh"
#include<vector>
using namespace std;

class Fully_Connect_Layer : public Abstract_Middle_Layer{
public:

	//全結合層を規定するパラメータです
	struct Param{
		//!ノード数です
		unsigned int nodes;
		//!活性化関数です
		ACTIVATE_FUNCTION func;

		//!初期化するときの重みの平均値です
		float mu_;

		//!初期化するときの重みの分散です
		float stddev_;

		//!初期化するときのバイアスの初期値
		float init_bias_;
	};

	/*!
	* \brief	デフォルトコンストラクタです
	*/
	Fully_Connect_Layer();

	/*!
	* \brief	コンストラクタです
	* \param[in]	入力層のパラメータです
	*/
	Fully_Connect_Layer(Fully_Connect_Layer::Param &param);

	/*!
	* \brief	コンストラクタです
	* \param[in]	node_num	入力層のパラメータです
	* \param[in]	func	活性化関数
	* \param[in]	mu	初期化するときの平均値
	* \param[in]	stddev	初期化するときの分散
	*/
	Fully_Connect_Layer(int node_num,ACTIVATE_FUNCTION func, float mu = 0, float stddev = 0.1f,float init_bias = 0);

	/*!
	* \brief   順伝播します
	* \param[in]	input	入力行列です
	* \param[in]	mode	学習中か、推論中か
	* \return	output
	*/
	Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode);

	/*!
	* \brief   初期化します
	* \param[in]	prev_output	前の層の出力です
	* \param[in]	output
	*/
	Sigma_Matrix* initialize(Sigma_Matrix* prev_output);

	/*!
	* \brief   バッチサイズを変更したときに、各層の出力の大きさを変更します
	* \param[in]	prev_output	前の層の出力です
	* \return	output
	*/
	Sigma_Matrix* change_batchsize(Sigma_Matrix* prev_output);

	/*!
	* \brief   出力を微分した値を返します
	* \return	d_output
	*/
	Sigma_Matrix* get_d_output();

	/*!
	* \brief   deltaを取得します
	* \return	delta
	*/
	Sigma_Matrix* get_delta(){ return &delta_; };

	/*!
	* \brief	学習に必要な\c Optimizer を初期化します
	* \param[in]	prev_output
	*/
	Sigma_Matrix* initialize_optimizer(Sigma_Matrix* prev_output);

	/*!
	* \brief   逆伝播します
	* \param[in]	prev_delta	前の層のデルタ行列です
	* \param[in]	prev_d_output	前の層の微分された出力です
	*/
	void back_ward(Sigma_Matrix* prev_delta, Sigma_Matrix* prev_d_output);

	/*!
	* \brief   Optimizerを用いて、パラメータの変更量を計算します
	*/
	Sigma_Matrix* calc_update_param(Sigma_Matrix* prev_output);

	/*!
	* \brief   パラメータを変更します
	*/
	void update_param();

	/*!
	* \brief   Additional_Processを追加します。
	* \param[in]	ap	Additional_Process
	*/
	void add_process(Additional_Process& ap);

	//!未実装です
	void save(string file_path){ return; };

	//!未実装です
	void load(string file_path){ return; };

	//////////////////////// Getter / Setter  //////////////////////////

	/*!
	* \brief	重み行列を設定します
	* \param[in]	weight	重み行列です
	*/
	void set_weight(Sigma_Matrix& weight){ weight_ = weight; };

	/*!
	* \brief	バイアスを設定します
	* \param[in]	bias	バイアス行列です
	*/
	void set_bias(Sigma_Matrix& bias){ bias_ = bias; };

	/*!
	* \brief	Optimizerを設定します
	* \param[in]	optimimzer	optimizerです
	*/
	void set_optimizer(Optimizer& optimizer){ optimizer_ = optimizer; };

	/*!
	* \brief	Optimizerを設定します
	* \param[in]	param	パラメータです
	*/
	void set_param(Fully_Connect_Layer::Param param){ param_ = param; };

	/*!
	* \brief	重み行列を取得します
	* \return	重み行列
	*/
	Sigma_Matrix& weight(){ return weight_; };

	/*!
	* \brief	バイアスを取得します
	* \return	バイアス
	*/
	Sigma_Matrix& bias(){ return bias_; };

	/*!
	* \brief	内部状態を取得します
	* \return	内部状態
	*/
	Sigma_Matrix& net(){ return net_; };

	/*!
	* \brief	Paramを取得します
	* \return	param
	*/
	Fully_Connect_Layer::Param& param(){ return param_; };

	/*!
	* \brief	deltaを取得します
	* \return	delta
	*/
	Sigma_Matrix& delta(){ return delta_; };

	/*!
	* \brief	微分された出力を取得します
	* \return	d_output
	*/
	Sigma_Matrix& d_output(){ return d_output_; }

	/*!
	* \brief	重みの更新行列を取得します
	* \return	重みの更新行列
	*/
	Sigma_Matrix& update_weight(){ return update_weight_; };

	/*!
	* \brief	バイアスの更新行列を取得します
	* \return	バイアスの更新行列
	*/
	Sigma_Matrix& update_bias(){ return update_bias_; };

	/*!
	* \brief	Optimizerを取得します
	* \return	optimimzer_
	*/
	Optimizer& optimizer(){ return optimizer_; };

private:
	//!パラメータです
	Fully_Connect_Layer::Param param_;

	//!Optimizerです
	Optimizer optimizer_;

	//!順伝播用
	vector<Additional_Process*> Process;

	//!バイアス行列です
	Sigma_Matrix bias_;

	//!重み行列です
	Sigma_Matrix weight_;

	//!内部状態です
	Sigma_Matrix net_;

	//////////////////////////////////パラメータ更新用

	//!デルタ行列です
	Sigma_Matrix delta_;

	//!*前の層の出力の微分値を計算する領域を確保します
	Sigma_Matrix d_output_;

	//!重みの更新量
	Sigma_Matrix update_weight_;

	//!バイアスの更新量
	Sigma_Matrix update_bias_;

	//!バイアスの計算に用いる単位列行列です
	Sigma_Matrix CONSTANT_1_;
};

#endif