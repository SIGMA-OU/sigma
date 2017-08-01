/*!
* \file    Sigma_Matrix.cuh
* \brief   Sigma_Matrixを定義しています。
* \date    2016/06/18
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef SIGMA_MATRIX_CUH_
#define SIGMA_MATRIX_CUH_
#undef NDEBUG

#include"Device_Matrix.cuh"

/*!
* \brief	複数の \c Device_Matrix　を保持するためのクラスです
* \details	複数の \c Device_Matrix を保持するためのクラスです。
*          　\c Device_Matrix は、add_Matrix()で追加することができます。
*			追加する際は、必ず同じサイズの\c Device_Matrixを追加してください。
*			\c Device_Matrix をSigam＿Matrixに追加することで、データを全てつなげることができ
*			計算処理の最適化を行っています。
*/
class Sigma_Matrix{
public:
	/*!
	* \brief   空の \c Sigma_Matrix オブジェクトを作成します。
	*/
	Sigma_Matrix();

	/*!
	* \brief	デストラクタです
	*/
	~Sigma_Matrix(){
		if (data_ != NULL)CHECK(cudaFree(data_));
	}

	/*!
	* \brief   コピーコンストラクタです
	* \param[in]	obj	コピーするSigma_Matrix
	*/
	Sigma_Matrix(const Sigma_Matrix &obj);

	/*!
	* \brief   指定した行/列数の \c Sigma_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	* \param[in]   rows        行数
	* \param[in]   cols        列数
	* \param[in]   init       初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	Sigma_Matrix(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.1f);

	/*!
	* \brief   指定した行/列数の \c Sigma_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	* \param[in]	height	高さ
	* \param[in]	width	幅
	* \param[in]
	* \param[in]   init       初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	Sigma_Matrix(unsigned height, unsigned int width = 1, unsigned int channel = 1, unsigned int depth = 1, unsigned int batch_size = 1, INITIALIZER init = ZERO, float a = 0, float b = 0.1f);

	/*!
	* \brief   指定した行/列数の \c Sigma_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	* \param[in]	height	高さ
	* \param[in]	width	幅
	* \param[in]
	* \param[in]   init       初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	void initialize(unsigned height,unsigned int width = 1, unsigned int channel = 1, unsigned int depth = 1, unsigned int batch_size = 1,INITIALIZER init = ZERO , float a = 0, float b = 0.1f);
	
	/*!
	* \brief   指定した行/列数の \c Sigma_Matrix オブジェクトを作成し、同時にデータの初期化も行います。
	*          initがCONSTANの時は、第一引数に設定したい値を入れる。
	*　　　　　 initがGAUSSIANまたはUNIF_DISTRIBのときは第二引数まで値をいれる
	*			それぞれ(第一引数,第二引数) = (平均, 分散)or(x,y)となる
	* \param[in]   rows        行数
	* \param[in]   cols        列数
	* \param[in]   init       初期化子 (ZERO, CONSTANT, GAUSSIAN, UNIF_DISTRIB, IDENTITY, RANDOM,STEP,POISSON)
	* \param[in]   a		  初期値(CONSTANT) 平均(GAUSSIAN)　値域ｘ(UNIF_DISTRIB)
	* \param[in]   b		　分散(GAUSSIAN)　値域y(UNIF_DISTRIB)
	*/
	void initialize(unsigned int rows, unsigned int cols, INITIALIZER init = ZERO, float a = 0, float b = 0.1f);

	/*!
	* \brief	\c Device_Matrixを追加します
	* \param[in]	Matrix	デバイスマトリックスです
	*/
	void add_Matrix(Device_Matrix& Matrix);

	/*!
	* \brief	Sigma_Matrix内の\c Device_Matrixを取得します
	* \param[in]	num	取得したいMatrixの番号
	* \warning	必ずBatch_size_よりも小さい番号を入れてください。indexは　0　から始まります
	*/
	Device_Matrix& get_Matrix(unsigned int idx);

	/*!
	* \brief	Sigma_Matrixの内部情報を表示します
	* \param[in]	All_Status	中身の配列も出力するかどうか
	*/
	void print(bool All_status = false);

	/////////////////////////////////// getter  /////////////////////
	/*!
	* \brief	heightを取得します
	* \return	height	
	*/
	inline unsigned int height(){ return height_; };

	/*!
	* \brief	widthを取得します
	* \return	width
	*/
	inline unsigned int width(){ return width_; };

	/*!
	* \brief	channelを取得します
	* \return	channel
	*/
	inline unsigned int channel(){ return channel_; };

	/*!
	* \brief	depthを取得します
	* \return	depth
	*/
	inline unsigned int depth(){ return depth_; };

	/*!
	* \brief	batch_sizeを取得します
	* \return	batch_size
	*/
	inline unsigned int batch_size(){ return batch_size_; };

	/*!
	* \brief	rowsを取得します
	* \return	rows
	*/
	inline unsigned int rows(){ return rows_; };

	/*!
	* \brief	colsを取得します
	* \return	cols
	*/
	inline unsigned int cols(){ return cols_; };

	/*!
	* \brief	sizeを取得します
	* \return	size
	*/
	inline size_t size(){ return size_; };

	/*!
	* \brief	配列のポインタを取得します
	* \return	data
	* \warning	GPU上のポインタであり、CPUメモリ上にデータがあるものではありません
	*/
	inline float* data(){ return data_; };

	/*!
	* \brief	格納されたミニバッチの先頭アドレスを返します
	* \param[in]	batch_num	指定バッチ番号です(0から始まります)
	* \return	batch_num	指定したバッチ番号です(0から始まる点に注意)
	* \warning	GPU上のポインタであり、CPUメモリ上にデータがあるものではありません(基本的に内部処理のみで使います)
	*/
	inline float* mini_batch_data(unsigned int batch_num){
		if (batch_num >= batch_size_){
			cout << "error : mini_batch_data : over size of batch_num";
			return data_;
		}
		else{
			return data_ + batch_num *(width_ * height_ * channel_ * depth_) ;
		}
	};

	/////////////////////setter////////////////////////////////
	/*!
	* \brief	データの値を設定します\n
	* \param[in]	row 行番号です
	* \param[in]	col	列番号です
	* \param[in]	value	設定する値です
	* \warning	配列のサイズを超えないように注意してください。
	*/
	void set_value(unsigned int row, unsigned int col,float value);

	/*!
	* \brief	データを配列データ(Host側)で設定します。(列優先の配列データにしてください。)\n
	* \param[in]	data
	* \warning	data_size分コピー使用としますので、dataは、同じサイズ分用意してください。
	*/
	void set_matrix(float* data);

	/*!
	* \brief	データの指定されたバッチ番号のデータを、配列データで設定します
	* \param[in]	data	配列データ
	* \param[in]	star_idx　バッチ内の何番目か
	* \param[in]	data_num	そのdataには、何データ分入っているか
	* \warning	data_num分データを設定しますので、容量をオーバーしないように注意してください。
	*/
	void set_matrix(float* data, unsigned int start_idx, unsigned int data_num = 1);

	/*!
	* \brief	データ情報をバイナリかCSVに出力します
	* \param[in]	file_path	ファイル名です
	* \param[in]	binary_output	バイナリーデータで吐き出すかを選べます(flaseでCSV出力されます)
	* \warning	.csvはfile_pathに必要ありません\n列優先の行列が転置された状態で保存されます。
	*/
	void save(string file_path,bool binary_output = true);

	/*!
	* \brief	バイナリファイルを読み込みます
	* \param[in]	file_path	ファイル名です(.datはいりません)
	*/
	void load(string file_path);

	/////////////////////////////////////////////////////
	Sigma_Matrix& operator = (const Sigma_Matrix& obj);

	Sigma_Matrix& operator /= (float x);

	Sigma_Matrix& operator *= (float x);

	Device_Matrix& operator [](unsigned int idx);

private:
	//!device_Matrixを取得したいときに用います
	Device_Matrix d_mx;

	//!各 \c Device_Matrix のwidthです
	unsigned int width_;

	//!各 \c Device_Matrix のheightです
	unsigned int height_;

	//!各 \c Device_Matrix のchannelです
	unsigned int channel_;

	//!各 \c Device_Matrix のdepthです
	unsigned int depth_;

	//! \c Device_Matrix の保持数です。バッチの大きさになります。
	unsigned int batch_size_;

	//! \c Sigma_Matrix の全てのデータ配列の行数です
	unsigned int rows_;

	//! \c Sigma_Matrix の全てのデータ配列の列数です
	unsigned int cols_;

	//!全てのデータのデータ列です。
	float *data_;

	//!全てのデータのサイズです
	size_t size_;

};



#endif