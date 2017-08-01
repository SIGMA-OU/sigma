/*!
* \file    Pooling2D_Layer.cuh
* \brief	二次元プーリング層を定義しています。
* \date    2016/07/10
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef POOLING2D_LAYER_CUH_
#define POOLING2D_LAYER_CUH_
#undef NDEBUG

#include"Abstract_Middle_Layer.cuh"
#include"Optimizer.cuh"
#include"Additional_Process.cuh"
#include"Activate_Function.cuh"
#include"Utilities.cuh"
#include"Sigma_global_functions.cuh"
#include<vector>
using namespace std;

class Pooling2D_Layer : public Abstract_Middle_Layer{
public:
	//!プーリング層を規定するパラメータです
	struct Param{
		//!プーリングを行う幅です
		unsigned int width;
		//!プーリングを行う高さです
		unsigned int height;
		//!ストライドです
		unsigned int stride;
		//!パディングの有無です
		bool padding;
		//!マックスプーリングかアベレージプーリングかを選べます
		POOLING_METHOD method;
	};

	/*!
	* \brief	デフォルトコンストラクタです
	*/
	Pooling2D_Layer();

	/*!
	* \brief	デフォルトコンストラクタです
	* \param[in]	param	プーリング層のパラメータ
	*/
	Pooling2D_Layer(Pooling2D_Layer::Param &param);

	/*!
	* \brief	デフォルトコンストラクタです
	* \param[in]	width	プーリング層の幅
	* \param[in]	height	プーリング層の高さ
	* \param[in]	stride	プーリング層のストライド
	* \param[in]	padding	パディングをするかどうか
	* \param[in]	method	MAXpooling or AVERAGEpoolingか
	*/
	Pooling2D_Layer(unsigned int width, unsigned int height,unsigned int stride, bool padding, POOLING_METHOD method);

	//!デフォルトデストラクタ
	~Pooling2D_Layer(){ if (d_idx_matrix != NULL)CHECK(cudaFree(d_idx_matrix)); };

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
	inline Sigma_Matrix* get_delta(){ return &delta_; };

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
	* \brief   Optimizerを用いて、パラメータの変更量を計算します(プーリング層は、何もしません) 
	*/
	Sigma_Matrix* calc_update_param(Sigma_Matrix* prev_output){ return &output_; };

	/*!
	* \brief   パラメータを変更します(プーリング層は何もしません) 
	*/
	void update_param(){ return; };

	/*!
	* \brief   Additional_Processを追加します。
	* \param[in]	ap	Additional_Process
	*/
	void add_process(Additional_Process& ap);

	//!未実装です
	void save(string file_path){ return; };

	//!未実装です
	void load(string file_path){ return; };

	///////////////////////////  Getter   /  Setter   /////////////////////
	
	/*!
	* \brief	パラメータを設定します
	* \param[in]	param	パラメータです
	*/
	void set_param(Pooling2D_Layer::Param& param);

	/*!
	* \brief	Paramを取得します
	* \return	param
	*/
	inline Pooling2D_Layer::Param& param(){ return param_; };

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

	int* idx_matrix(){ return d_idx_matrix; };

	void set_optimizer(Optimizer& opt){ return; };

private:
	//!パラメータです
	Pooling2D_Layer::Param param_;

	//!順伝播用
	vector<Additional_Process*> Process;

	//!デルタ行列です
	Sigma_Matrix delta_;

	//!*前の層の出力の微分値を計算する領域を確保します
	Sigma_Matrix d_output_;

	//!プーリングしたインデックスを保存するための配列(int型配列です)
	int* d_idx_matrix;
};


#endif