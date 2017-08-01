#include"Imaging.cuh"

vector<Mat> Matrix2Image(Device_Matrix& matrix, bool RGB_MODE, IMAGE_CONVERT_MODE mode){
	vector<Mat> images;
	if (matrix.channel() != 3 && RGB_MODE == true){
		cout << "Matrix2Image error : CHANNEL must be 3 by using RGB_MODE" << endl;
		return images;
	}

	Mat image;
	if (RGB_MODE){
		image.create(matrix.height(),matrix.width(),CV_8UC3);
	}
	else{
		image.create(matrix.height(), matrix.width(), CV_8UC1);
	}
	float* data;
	data = (float*)malloc(matrix.size());
	CHECK(cudaMemcpy(data,matrix.data(),matrix.size(),cudaMemcpyDeviceToHost));

	switch (mode)
	{
	case USING_RAW_VALU:
		if (RGB_MODE){
			for (int ch = 0; ch < 3; ch++){
				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){

						int row_idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						if ((int)data[row_idx] < 0)image.data[3 *( x * matrix.height() + y) + ch] = 0;
						else if ((int)data[row_idx] > 255)image.data[3 * (x * matrix.height() + y) + ch] = 255;
						else image.data[3 * (x * matrix.height() + y) + ch] = (int)data[row_idx];

					}
				}
			}
			images.push_back(image.clone());
		}
		else{
			for (int ch = 0; ch < matrix.channel(); ch++){
				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						if ((int)data[idx] < 0)image.data[x * matrix.height() + y] = 0;
						else if ((int)data[idx] > 255)image.data[x * matrix.height() + y] = 255;
						else image.data[x * matrix.height() + y] = (int)data[idx];
					}
				}
				images.push_back(image.clone());
			}
		}
		break;
	case ZERO_CENTER_NORMALIZATION:
	{
		if (RGB_MODE){
			for (int ch = 0; ch < 3; ch++){

				float min = FLT_MAX;
				float max = -FLT_MAX;

				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int row_idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						if (data[row_idx] < min) min = data[row_idx];
						if (data[row_idx] > max) max = data[row_idx];
					}
				}
				if (min == 0) min = 1;
				if (max == 0) max = 1;

				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int row_idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						if (data[row_idx] < 0){
							image.data[3 * (x * matrix.height() + y) + ch] = (int)((data[row_idx]) / min * 127);
						}
						else{
							image.data[3 * (x * matrix.height() + y) + ch] = 128 + (int)((data[row_idx]) / max * 127);
						}
					}
				}
			}
			images.push_back(image.clone());
		}
		else{
			for (int ch = 0; ch < matrix.channel(); ch++){
				float min = FLT_MAX;
				float max = -FLT_MAX;

				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int row_idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						if (data[row_idx] < min) min = data[row_idx];
						if (data[row_idx] > max) max = data[row_idx];
					}
				}

				if (min == 0) min = 1;
				if (max == 0) max = 1;

				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						if (data[idx] < 0){
							image.data[x * matrix.height() + y] = (int)((data[idx]) / min * 127);
						}
						else{
							image.data[x * matrix.height() + y] = 128 + (int)((data[idx]) / max* 127);
						}
					}
				}
				images.push_back(image.clone());
			}
		}
	}
		break;
	case MIN_FROM_MAX_NORMALIZATION:
	{
		if (RGB_MODE){
			for (int ch = 0; ch < 3; ch++){

				float min = FLT_MAX;
				float max = -FLT_MAX;

				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int row_idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						if (data[row_idx] < min) min = data[row_idx];
						if (data[row_idx] > max) max = data[row_idx];
					}
				}
				float val = max - min;
				if (val == 0)val = 1;

				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int row_idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						image.data[3 * (x * matrix.height() + y) + ch] = (int)((data[row_idx] - min) / val * 255);
					}
				}
			}
			images.push_back(image.clone());
		}
		else{
			for (int ch = 0; ch < matrix.channel(); ch++){
				float min = FLT_MAX;
				float max = -FLT_MAX;

				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int row_idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						if (data[row_idx] < min) min = data[row_idx];
						if (data[row_idx] > max) max = data[row_idx];
					}
				}
				float val = max - min;
				if (val == 0)val = 1;

				for (int x = 0; x < matrix.width(); x++){
					for (int y = 0; y < matrix.height(); y++){
						int idx = ch * matrix.width() * matrix.height() + x * matrix.height() + y;
						image.data[x * matrix.height() + y] = (int)((data[idx] - min) / val * 255);
					}
				}
				images.push_back(image.clone());
			}
		}
	}
		break;
	default:
		break;
	}

	free(data);
	return images;
}


vector<vector<Mat>> Matrix2Image(Sigma_Matrix& matrix, bool RGB_MODE, IMAGE_CONVERT_MODE mode){
	vector<vector<Mat>> images;
	for (int i = 0; i < matrix.batch_size(); i++){
		vector<Mat> image = Matrix2Image(matrix.get_Matrix(i), RGB_MODE, mode);
		images.push_back(image);
	}
	return images;
}