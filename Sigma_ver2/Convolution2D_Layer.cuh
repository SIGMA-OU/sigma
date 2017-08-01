/*!
* \file    Convolution2D_Layer.cuh
* \brief	2次元の畳み込み層を定義しています
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/
#ifndef CONVOLUTION2D_LAYER_CUH_
#define CONVOLUTION2D_LAYER_CUH_
#undef NDEBUG

#include"Abstract_Middle_Layer.cuh"
#include"Optimizer.cuh"
#include"Additional_Process.cuh"
#include"Activate_Function.cuh"
#include"Utilities.cuh"
#include<vector>
using namespace std;

class Convolution2D_Layer : public Abstract_Middle_Layer{
public:
	//!畳み込み層を規定するパラメータです
	struct Param{
		//!畳み込みフィルタの幅,高さ,フィルタ数,ストライドです
		unsigned int width, height, filter_num, stride;
		//!活性化関数です
		ACTIVATE_FUNCTION func;
		//!パディングをするかどうかです
		bool padding;

		//!初期化のときに用いるフィルタの平均値です
		float mu_;

		//!初期化の時に用いるフィルタの分散値です
		float stddev_;

		//!初期化の時に用いるバイアスの値です
		float init_bias_;
	};
	//!デフォルトコンストラクタです
	Convolution2D_Layer();

	Convolution2D_Layer(Convolution2D_Layer::Param& param);

	Convolution2D_Layer(unsigned int width,unsigned int height,unsigned int filter_num,unsigned int stride, bool padding,ACTIVATE_FUNCTION func = relu,float mu = 0,float stddev = 0.05f,float init_bias = 0);

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
	* \brief	学習に必要な\c Optimizer を初期化します
	* \param[in]	prev_output
	* \return	*output
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
	* \brief	deltaを取得します
	* \return	delta
	*/
	Sigma_Matrix* get_delta(){ return &delta_; };

	/*!
	* \brief   outputのポインタを返します
	* \return	&output
	*/
	Sigma_Matrix* get_output(){ return &output_; };

	void save(std::string file_path){ return; };

	void load(std::string file_path){ return; };

	/////////////////// setter  /////////////////
	/*!
	* \brief	Optimizerを設定します
	* \param[in]	optimimzer	optimizerです
	*/
	inline void set_optimizer(Optimizer& opt){ optimizer_ = opt; };

	////////////////////geter  /////////////////////////
	/*!
	* \brief	パラメーターを取得します
	* \return	param_
	*/
	inline Convolution2D_Layer::Param& param(){ return param_; };

	/*!
	* \brief	optimizerを取得します
	* \return	param_
	*/
	inline Optimizer optimezer(){ return optimizer_; };

	/*!
	* \brief	バイアスを取得します
	* \return	bais_
	*/
	inline Sigma_Matrix& bias(){ return bias_; };

	/*!
	* \brief	フィルターを取得します
	* \return	filter_
	*/
	inline Sigma_Matrix& filter(){ return filter_; };

	/*!
	* \brief	net値を取得します
	* \return	net_
	*/
	inline Sigma_Matrix& net(){ return net_; };

	/*!
	* \brief	入力データを並べ替えたデータ
	* \return	transe_input_
	*/
	inline Sigma_Matrix& transe_input(){ return transe_input_; };

	/*!
	* \brief	デルタ行列をを取得します
	* \return	delta_
	*/
	inline Sigma_Matrix& delta(){ return delta_; };

	/*!
	* \brief	transe_delta_を取得します
	* \return	transe_delta_
	*/
	inline Sigma_Matrix& transe_delta(){ return transe_delta_; };

	/*!
	* \brief	微分出力を取得します
	* \return	d_outpout_
	*/
	inline Sigma_Matrix& d_output(){ return d_output_; };

	/*!
	* \brief	フィルタの更新行列を取得します
	* \return	update_filter_
	*/
	inline Sigma_Matrix& update_filter(){ return update_filter_; };

	/*!
	* \brief	バイアスの更新行列を取得します
	* \return	update_bias_
	*/
	inline Sigma_Matrix& update_bias(){ return update_bias_; };


private:
	//!パラメータです
	Convolution2D_Layer::Param param_;

	//!Optimizerです
	Optimizer optimizer_;

	//!順伝播用
	vector<Additional_Process*> Process;

	//!バイアス行列です
	Sigma_Matrix bias_;

	//!フィルター行列です
	Sigma_Matrix filter_;

	//!内部状態です
	Sigma_Matrix net_;

	//!入力データを並べ替えたデータです
	Sigma_Matrix transe_input_;

	//////////////////////////////////////

	//!デルタ行列です
	Sigma_Matrix delta_;

	//逆伝播の時に使います。
	Sigma_Matrix transe_delta_;

	//!*前の層の出力の微分値を計算する領域を確保します
	Sigma_Matrix d_output_;

	//!filterの更新量です
	Sigma_Matrix update_filter_;

	//!バイアスの更新量
	Sigma_Matrix update_bias_;

	//!バイアスの計算に用いる単位列行列です
	Sigma_Matrix CONSTANT_1_;
};

#endif