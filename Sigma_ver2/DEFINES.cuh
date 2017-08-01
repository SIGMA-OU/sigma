/*!
* \file    DEFINES.cuh
* \brief   Sigma_GPUverで使う列挙体と定数を指定しています。
* \date    2016/05/30
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef DEFINES_CUH_
#define DEFINES_CUH_
#undef NDEBUG

//!block_dim_xを規定しています
#define BLOCK_X	32
//!block_dim_yを規定しています
#define BLOCK_Y	32

/*! @brief CUDAにおいてエラーが起きていないかをチェックするマクロです */
#define CHECK(call)											\
{															\
	const cudaError_t error = call;							\
	if(error != cudaSuccess){								\
		printf("Error: %s: %d, ",__FILE__, __LINE__);		\
		printf("code: %d, reason: %s \n", error,			\
		       	cudaGetErrorString(error));	getchar();		\
		exit(1);											\
											}								\
}

/*! @brief cuBLASにおいてエラーが起きていないかをチェックするマクロです */
#define BLAS_CH(call)										\
{															\
	const cublasStatus_t blas_error = call;					\
	if(blas_error != cudaSuccess){							\
		printf("cuBLAS_Error: %s: %d, ",__FILE__, __LINE__);\
		printf("press any key!\n");getchar();				\
		exit(1);											\
									}										\
}

/*! @brief cuRANDにおいてエラーが起きていないかをチェックするマクロです */
#define RAND_CH(call)										\
{															\
	const curandStatus_t blas_error = call;					\
	if(blas_error != CURAND_STATUS_SUCCESS){							\
		printf("cuRAND_Error: %s: %d, ",__FILE__, __LINE__);\
		printf("press any key!\n");getchar();				\
		exit(1);											\
										}										\
}

/*! @brief 行優先のインデックスを、列優先に変換するマクロです */
#define R2C(r, c, nrows) ((c) * (nrows) + (r))

#define CURAND_LIMIT_NUM	100000

/**
* @enum INITIALIZER
* 配列を初期化するための列挙体.初期化子です
*/
enum INITIALIZER{
	//!ゼロで初期化します
	ZERO,

	//!定数で初期化します
	CONSTANT,

	//!単位行列で初期化します
	IDENTITY,

	//!乱数（0〜1の範囲）で初期化します
	RANDOM,

	//!ガウス分布で初期化します
	GAUSSIAN,

	//!値域[x,y]の一様分布で初期化します
	UNIF_DISTRIB,

	//![1,2,3,4,5...]で初期化します
	STEP,

	//!自由に設定した初期化を行うためのものです
	PLANE
};

/**
* @enum RC_MAJOR
* 行優先/列優先を表す列挙体です
*/
enum RC_MAJOR{
	//!行優先です
	ROW_MAJOR,

	//!列優先です
	COL_MAJOR
};

/*!	@enum ACTIVATE_FUNCTION
* \brief   活性関数を記述するための列挙体です
*/
typedef enum ACTIVATE_FUNCTION{
	//!双曲線関数です
	hytan,

	//!シグモイド関数です
	sig,

	//!恒等写像関数です
	iden,

	//!ランプ関数(ReLU)です
	relu,

	//!ソフトマックス関数です
	softmax,

	//!マックスアウト関数です
	maxout,

	//!自由に関数を作る事ができます
	plane
}AF;

/*!	@enum OPTIMIZER
* \brief   学習の最適化を行うメソッドを表す列挙体です。
*/
typedef enum OPTIMIZER{
	//!確率的勾配降下法を用います
	SGD,

	//!確率的勾配降下法とモメンタムを用います
	SGD_MOMENTUM,

	//!ADAMを用います
	ADAM
}OP;

/*!	@enum REGULARIZATION
* \brief   重みの正則化を表す列挙体です。
*/
typedef enum REGULARIZATION{
	//!重み減衰を行います
	WEIGHT_DECAT,

	//!重み上限を設けます
	WEIGHT_UPPER_LIMIT,

	//!スパース正則化を行います
	SPARSE_REGULARIZATION
}RE;

/*!	@enum ADDITIONAL_PROCESS
* \brief   順伝播の際に、出力に対して行う演算処理を表した列挙体です
*/
typedef enum ADDITIONAL_PROCESS{

	//!DropOut マスクをかけます
	DROP_OUT,

	//!一定の雑音を付与します
	NOISE
}AP;

/*!	@enum LAYER
* \brief   Sigmaで基本的に用意されている層を表す列挙体です
*/
typedef enum LAYER{
	//!入力層です
	INPUT_LAYER,

	//!全結合層です
	FULLY_CONNECT_LAYER,

	//!畳み込み層です
	CONVOLUTION_LAYER,

	//!プーリング層です
	POOLING_LAYER,

	//!正規化層です
	NORMALIZATION_LAYER
}LR;

/*!	@enum LOSS_FUNCTION
* \brief   Sigmaで基本的に用意されている損失関数を表す列挙体です
*/
typedef enum LOSS_FUNCTION{
	//!交差エントロピー関数です
	CROSS_ENTROPY,

	//!二乗誤差です
	MEAN_SQURE_ERROR,

	//!損失関数を変更することができます。
	PLANE_LOSS
}LF;

/*!	@enum NN_MODE
* \brief   ニューラルネットワークが学習中か、推論中かを表す列挙体です
*/
enum NN_MODE{ TRAIN, INFER };

/*!	@enum POOLING_METHOD
* \brief	プーリング層のプーリングメソッドを表す列挙体です
*/
enum POOLING_METHOD{
	//!マックスプーリングです
	MAX_POOLING,

	//!アベレージプーリングです
	AVERAGE_POOLING
};

#endif