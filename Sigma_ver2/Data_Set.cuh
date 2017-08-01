/*!
* \file    Data_Set.cuh
* \brief	データを格納するクラスです
* \date    2016/07/15
* \author  Asatani Satoshi(@exaintelligence)
*/

#ifndef DATA_SET_CUH_
#define DATA_SET_CUH_
#undef NDEBUG

#include"Sigma_Matrix.cuh"

/*!
* \brief	データを簡単に扱うためのクラスData_Setクラスを定義しています
* \details	データを簡単に扱うためのクラスData_Setクラスを定義しています\n
*			データ全てはdata_に格納されており\n
*			ミニバッチを自動的に作成することができます。\n
*			データはaddで追加することができます。\n
*			自動的に教師データの作成も可能です\n
*/
class SL_Data_Set{
public:
	//!デフォルトコンストラクタです
	SL_Data_Set();

	/*!
	* \brief	コンストラクタです
	* \param[in]	data	デバイスマトリックスで初期化できます
	* \warning	バッチサイズ以外のデータサイズが初期化されて、以後、同じサイズのデータしか入れられなくなります。
	*/
	SL_Data_Set(Device_Matrix& data);

	/*!
	* \brief	コンストラクタです
	* \param[in]	data	デバイスマトリックスで初期化できます
	* \param[in]	teacher	dataに対応したteacherです。
	* \warning	バッチサイズ以外のデータサイズが初期化されて、以後、同じサイズのデータしか入れられなくなります。
	*/
	SL_Data_Set(Device_Matrix& data,Device_Matrix& teacher);

	/*!
	* \brief	コンストラクタです。　\c Sigma_Matrix のバッチサイズ分だけデータ数が増えます
	* \param[in]	data	Sigma_Matrixです。
	* \warning	バッチサイズ以外のデータサイズが初期化されて、以後、同じサイズのデータしか入れられなくなります。
	*/
	SL_Data_Set(Sigma_Matrix& data);

	/*!
	* \brief	コンストラクタです。　\c Sigma_Matrix のバッチサイズ分だけデータ数が増えます
	* \param[in]	data	Sigma_Matrixです。
	* \param[in]	teacher	dataに対応したteacherです
	* \warning	バッチサイズ以外のデータサイズが初期化されて、以後、同じサイズのデータしか入れられなくなります。
	*/
	SL_Data_Set(Sigma_Matrix& data,Sigma_Matrix& teacher);

	Sigma_Matrix& make_teacher();

	Sigma_Matrix& next(unsigned int data_num);

	Sigma_Matrix& next_teacher(unsigned int data_num);

private:
	//!データの番号を指定しているインデックスです。next関数で使用されれます
	unsigned int idx;

	//!訓練もしくはテストデータが入っています
	Sigma_Matrix data_;

	//!教師もしくは正解データが入っています(別に使わないこともあります。)
	Sigma_Matrix teacher_;

	//!外部にアクセスできるマトリックスを
	Sigma_Matrix divide_data_;

	//training_dataをvector型でストックします
	vector<Sigma_Matrix> training_set_;

	//teacher_dataをvector型でストックします。
	vector<Sigma_Matrix> teacher_set_;

	//クラスに対応した名前を設定することができます。
	vector<string> tag_set;

	//各データに対するlabelを保持できます。
	vector<vector<int>> label_set;
};



#endif