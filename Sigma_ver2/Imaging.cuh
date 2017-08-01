/*!
* \file    Imaging.cuh
* \brief	\c Sigma_Matrix �������� \c Device_Matrix���摜�����邱�Ƃ��ł��܂��B(�܂�GPU�œK�������Ă��܂���)
* \date    2016/07/29
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef IMAGING_CUH_
#define IMAGING_CUH_
#undef NDEBUG

#include"Sigma_Matrix.cuh"

#include<opencv2/opencv.hpp>
#include<vector>
#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_highgui249.lib")

using namespace std;
using namespace cv;

enum IMAGE_CONVERT_MODE{
	//!�o�͂̒l�����̂܂܉摜�ɓ���܂��B()
	USING_RAW_VALU,

	//!
	ZERO_CENTER_NORMALIZATION,

	//!
	MIN_FROM_MAX_NORMALIZATION
};

vector<Mat> Matrix2Image(Device_Matrix& matrix, bool RGB_MODE = false, IMAGE_CONVERT_MODE mode = ZERO_CENTER_NORMALIZATION);
vector<vector<Mat>> Matrix2Image(Sigma_Matrix& matrix, bool RGB_MODE = false, IMAGE_CONVERT_MODE mode = ZERO_CENTER_NORMALIZATION);


#endif

