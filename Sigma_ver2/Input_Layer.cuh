/*!
* \file    Input_Layer.cuh
* \brief	入力層であるInput_Layerを定義しています
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef INPUT_LAYER_CUH_
#define INPUT_LAYER_CUH_
#undef NDEBUG

#include<vector>
#include"Abstract_Layer.cuh"
#include"Additional_Process.cuh"
using namespace std;
/*!
* \brief	Sigmaで用いられる入力層クラスです
* \details	Sigmaで用いられる入力層クラスです\n
*			\c Abstract_Layer を継承しています。\n
*			Input_Layerは情報の受け取り口として機能しています\n
*			Input_Layerは入力された情報をコピーします。
*/
class Input_Layer : public Abstract_Layer{
public:
	//!デフォルトコンストラクタです
	Input_Layer();

	/*!
	* \brief   Layerの情報を書き出します(未実装)
	* \param[in]	file_patch	書き出すfile_nameです
	*/
	void save(string file_path);

	/*!
	* \brief   Layerの情報を読み込みます(未実装)
	* \param[in]	file_patch	読み込むfile_nameです
	*/
	void load(string file_path);

	/*!
	* \brief   入力層を初期化します
	* \param[in]	input	入力データ
	* \return  &output_
	*/
	Sigma_Matrix* initialize(Sigma_Matrix* input);

	/*!
	* \brief   バッチサイズが変化したときに、各層の出力行列のサイズを変更します
	* \param[in]	input	入力データ
	* \return  &output_
	*/
	Sigma_Matrix* change_batchsize(Sigma_Matrix* input);

	/*!
	* \brief   順伝播します
	* \param[in]	input	入力する配列のポインタです
	*/
	Sigma_Matrix* forward(Sigma_Matrix* input, NN_MODE mode = TRAIN);

	/*!
	* \brief   outputのポインタを取得します
	* \return	&output
	*/
	Sigma_Matrix* get_output(){ return &output_; };

	/*!
	* \brief   Additional_Processを追加します。
	* \param[in]	ap	Additinal＿Processです。
	*/
	void add_process(Additional_Process& ap);

private:
	//!追加される演算処理です
	vector<Additional_Process*> Process;
};

#endif