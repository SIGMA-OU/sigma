/*!
* \file   Noising_Process.cuh
* \brief   ノイズ処理を行うためのクラスNoisingが定義されています。(まだ未実装)
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef NOISING_PROCESS_CUH_
#define NOISING_PROCESS_CUH_
#undef NDEBUG

#include"Additional_Process.cuh"

/*!
* \brief	ノイズ処理を実行するためのクラスNoisingが規定されています。
* \details	ノイズ処理を実行するためのクラスNoisingが規定されています。\n
*			仮想クラスである \c Additional_Process　を継承しており、\n
*			\c Sigma_Layer　に追加することで、実行することができます。\n
*/
class Noising : public Additional_Process{
public:
	//!デフォルトコンストラクタです
	Noising(){ rate_ = 0.0f; };

	//!コピーコンストラクタです
	Noising(const Noising& obj){
		rate_ = obj.rate_;
	};

	enum mode{
		//!ガウス分布に従ったノイズ
		GAUSSIAN_NOISE,

		//マスキングノイズ
		MASKING_NOISE,

		//胡麻塩ノイズ
		SALTANDPEPPER
	};

	/*!
	* \brief   入力された行列に対してノイズをかけます
	* \param[in]	input	入力行列のポインタ
	* \param[in]	mode	学習中か推論中か(TRAIN,INFER)
	* \return  行数
	*/
	Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode = TRAIN);

	/*!
	* \brief   データに対してノイズを掛ける割合を設定します
	* \param[in]	val	ノイズを掛ける割合です
	* \warning	割合はは、必ず0～100の間にしてください。
	* \return  行数
	*/
	void set_rate(float val){
		if (val < 0 || val >100){
			std::cout << "error : Noising : input rate is wrong" << std::endl;
			return;
		}
		rate_ = val;
		return;
	};

	/*!
	* \brief   ノイズを掛ける割合を取得します
	* \return  ノイズを掛ける割合
	*/
	float rate(){ return rate_; };

private:
	//!ノイズレート
	float rate_;

	//モード
	Noising::mode mode_;
};

#endif