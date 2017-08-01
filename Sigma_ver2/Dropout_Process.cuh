/*!
* \file   Dropout_Process.cuh
* \brief   ドロップアウト処理を行うためのクラスDropoutが定義されています。(まだ未実装)
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef DROPOUT_PROCESS_CUH_
#define DROPOUT_PROCESS_CUH_
#undef NDEBUG

#include"Additional_Process.cuh"

/*!
* \brief	Dropoutを実行するためのクラスDropoutが規定されています。
* \details	Dropoutを実行するためのクラスDropoutが規定されています。\n
*			仮想クラスである \c Additional_Process　を継承しており、\n
*			\c Sigma_Layer　に追加することで、実行することができます。\n
*/
class Dropout : public Additional_Process{
public:
	//!デフォルトコンストラクタです
	Dropout(){ rate_ = 0.0f; };

	//!コピーコンストラクタです
	Dropout(const Dropout& obj){
		rate_ = obj.rate_;
	}

	/*!
	* \brief   入力された行列に対してDropoutマスクを掛け合わせます
	* \param[in]	input	入力行列のポインタ
	* \param[in]	mode	学習中か推論中か(TRAIN,INFER)
	* \return  行数
	*/
	Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode = TRAIN);

	/*!
	* \brief   ドロップアウトレートを設定します
	* \param[in]	val	ドロップアウトレートです
	* \warning	ドロップアウトレートは、必ず0～100の間にしてください。
	* \return  行数
	*/
	void set_rate(float val){
		if (val < 0 || val >100){
			std::cout << "error : Dropout : input rate is wrong" << std::endl;
			return;
		}
		rate_ = val;
		return;
	};

	/*!
	* \brief   ドロップアウトレートを取得します
	* \return  ドロップアウトレート
	*/
	float rate(){ return rate_; };

private:
	//!ドロップアウトレート
	float rate_;
};

#endif