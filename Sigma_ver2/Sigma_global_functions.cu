#include"Sigma_global_functions.cuh"

__global__ void set_array_2D(float* data, unsigned int rows, unsigned int cols, float val){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		data[idx] = val;
	}
}

__global__ void set_array_identity_2D(float* data, unsigned int rows, unsigned int cols){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		data[idx] = 1;
		if (ix == iy)data[idx] = 0;
	}
}

__global__ void set_array_step_2D(float* data, unsigned int rows, unsigned int cols){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		data[idx] = idx;
	}
}

__global__ void sum_constant_2D(float*data, unsigned int rows, unsigned int cols, float val){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		data[idx] += val;
	}
}

__global__ void initialize_uniform2D(float *data, unsigned int rows, unsigned int cols, float x, float y, curandState_t *state){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		data[idx] = x + (y - x) * curand_uniform(&state[idx]);
	}
}

__global__ void initialize_normal2D(float *data, unsigned int rows, unsigned int cols, float mean, float stddev,curandState_t *state){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		data[idx] = mean + stddev * curand_normal(&state[idx]);
	}
}

__global__ void rand_init2D(unsigned long long seed,unsigned int rows, unsigned int cols, curandState_t *states){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		curand_init(idx +seed,0,0,&states[idx]);
	}
}

__global__ void block1D(float* data,unsigned int data_begin,float* src_data, unsigned int src_data_num){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	if (ix < src_data_num){
		data[ix + data_begin] = src_data[ix];
	}
}

__global__ void replace_array_for_CONV2D(float* result_data, float* src_data,
	unsigned int Image_width, unsigned int Image_height, unsigned Image_channel,
	unsigned int filter_width, unsigned int filter_height, unsigned int stride,
	bool padding, unsigned int result_rows, unsigned int result_cols){

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * result_rows + iy;

	if (ix >= result_cols || iy >= result_rows)return;

	if (padding){
		//畳み込みを行う範囲のピクセル数を計算する
		unsigned int filter_size = filter_height * filter_width;
		//元画像のチャネル数
		unsigned int i_channel = iy / filter_size;
		//元画像のチャネルごとの畳み込みを行う範囲
		unsigned int rectangle = iy % filter_size;
		//元画像のwidth
		unsigned int i_width = rectangle / filter_height;
		//元画像のheight
		unsigned int i_height = rectangle % filter_height;

		//元画像の
		unsigned int height_num = (Image_height - 1) / stride + 1;
		//元画像の幅のインデックス
		unsigned int width_idx = ix / height_num * stride + i_width - filter_width / 2;
		//元画像の高さのインデックス
		unsigned int height_idx = ix % height_num * stride + i_height - filter_height / 2 ;

		//元画像中のインデックス
		if (width_idx >= 0 && width_idx < Image_width && height_idx >= 0 && height_idx < Image_height){
			unsigned int src_idx = i_channel * (Image_height * Image_width) + width_idx * Image_height + height_idx;
			result_data[idx] = src_data[src_idx];
		}
		else{
			result_data[idx] = 0;
		}

		return;
	}
	else{
		//畳み込みを行う範囲のピクセル数を計算する
		unsigned int filter_size = filter_height * filter_width;
		//元画像のチャネル数
		unsigned int i_channel = iy / filter_size;
		//元画像のチャネルごとの畳み込みを行う範囲
		unsigned int rectangle = iy % filter_size;
		//元画像のwidth
		unsigned int i_width = rectangle / filter_height;
		//元画像のheight
		unsigned int i_height = rectangle % filter_height;
		
		//元画像の
		unsigned int height_num = (Image_height - filter_height) / stride + 1 ;
		//元画像の幅のインデックス
		unsigned int width_idx = (ix / height_num) * stride + i_width;
		//元画像の高さのインデックス
		unsigned int height_idx = (ix % height_num) * stride + i_height;

		//元画像中のインデックス
		unsigned int src_idx = i_channel * (Image_height * Image_width) + width_idx * Image_height+ height_idx;

		//printf("ix : %d , iy : %d, width : %d , height : %d ,idx : %d,  src_idx : %d , data : %f\n",ix, iy, width_idx, height_idx,idx,src_idx,src_data[src_idx]);

		//代入
		result_data[idx] = src_data[src_idx];

		return;
	}
}

__global__ void replace_array_for_backward_CONV2D(float* prev_delta, float* src_data,
	unsigned int Image_width, unsigned int Image_height, unsigned Image_channel,
	unsigned int filter_width, unsigned int filter_height, unsigned int stride,
	bool padding, unsigned int prev_rows, unsigned int prev_cols, unsigned int src_rows, unsigned int src_cols){

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * prev_rows + iy;

	if (iy >= Image_height)return;
	if (ix >= Image_height * Image_channel)return;

	//スライドさせたときに、もっとも最初に畳み込みされる場所を計算する
	int idx_x = ((int)ix % (int)Image_width) - (int)filter_width + 1 ;
	int idx_y = (int)iy - (int)filter_height + 1;

	//画像のチャネル数を求める(列優先で画像が入っているので、Imageの幅で割ればOK)
	unsigned int image_channel_num = ix / Image_width;

	if (padding){
		if (idx_x < -(int)filter_width / 2)idx_x = -(int)filter_width / 2;
		if (idx_y < -(int)filter_height / 2)idx_y = -(int)filter_height / 2;

		unsigned int height_num = (Image_height - 1) / stride + 1;
		unsigned int width_num = (Image_width - 1) / stride + 1;

		prev_delta[idx] = 0;
		for (int i = idx_x; i <= (int)(ix % Image_width); i++){
			//srcにおけるidxを計算する
			unsigned int src_col_num  = (i + (int)filter_width / 2) / stride;
			//printf("(ix,iy) = (%d,%d),cont point: %d  src_col : %d\n", ix, iy, (i + (int)filter_width / 2) % stride, src_col_num);
			if (src_col_num >= width_num)break;
						if ((i + (int)filter_width / 2) % stride != 0)continue;
			
			for (int j = idx_y; j <= (int)iy; j++){
				int temp_y = j + filter_height / 2;
				if (temp_y % stride != 0)continue;
				if (temp_y / stride >= height_num)break;
				//src_dataにおける列番号を計算する
				unsigned int src_cols_idx = height_num * src_col_num + temp_y / stride;
				//src_dataにおける行番号を計算する
				unsigned int src_rows_idx = filter_height * (ix % Image_width - i) + (iy - j) + image_channel_num * (filter_width * filter_height);
				//src_dataにおけるidxを計算する
				unsigned int src_idx = src_cols_idx * src_rows + src_rows_idx;
				prev_delta[idx] += src_data[src_idx];
			}
		}
	}
	else{
		if (idx_x < 0) idx_x = 0;
		if (idx_y < 0) idx_y = 0;

		unsigned int height_num = (Image_height - filter_height) / stride + 1;
		unsigned int width_num = (Image_width - filter_width) / stride + 1;
		
		prev_delta[idx] = 0;

		//フィルターの開始位置の探索
		for (int i = idx_x; i <= ix % Image_width;i++){
			//srcにおけるidxを計算する
			unsigned int src_col_num = i / stride;
			
			//範囲外だと抜ける
			if (src_col_num >= width_num)break;

			//幅が範囲外だと、ループを抜ける
			//if (src_col_num * stride + filter_width >= Image_width)break;
			//起点がストライド幅に合致していなければスルー
			
			if (i % stride != 0)continue;

			//高さに関してループ
			for (int j = idx_y; j <= iy; j++){
				
				//起点がstrideに対応していなければ、スルー
				if (j % stride != 0)continue;
				if (j / stride >= height_num)break;
				//src_dataにおける列番号を計算する
				unsigned int src_cols_idx = height_num * src_col_num + j / stride;

				//src_dataにおける行番号を計算する
				unsigned int src_rows_idx = filter_height * (ix % Image_width - i) + (iy - j) + image_channel_num * (filter_width * filter_height);

				//src_dataにおけるidxを計算する
				unsigned int src_idx = src_cols_idx * src_rows + src_rows_idx;

				prev_delta[idx] += src_data[src_idx];
			}
		}
	}
	return;
}

__global__ void max_Pooling2D(float* output_data, float* src_data, int* idx_matrix,
	unsigned int Image_width, unsigned int Image_height, unsigned int Image_channel,
	unsigned int output_width, unsigned int output_height,
	unsigned int pooling_width, unsigned int pooling_height, unsigned int stride,
	bool padding, unsigned int output_rows, unsigned int output_cols){
	
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * output_rows + iy;
	
	if (ix >= output_width * Image_channel)return;
	if (iy >= output_height)return;

	//パディングによって、出力行列が変化する
	if (padding){
		unsigned int src_channel = ix / output_width;
		int src_idx_ix = (int)((ix % output_width) * stride) - (int)pooling_width / 2;
		int src_idx_iy = (int)(iy * stride) - (int)pooling_height / 2;
		unsigned int src_idx = 0;
		
		output_data[idx] = -FLT_MAX;

		for (int i = src_idx_ix; i < src_idx_ix + (int)pooling_width; i++){
			if (i >= (int)Image_width)break;
			else if (i < 0)continue;
			for (int j = src_idx_iy; j < src_idx_iy + (int)pooling_height; j++){
				if (j >= (int)Image_height)break;
				else if (j < 0)continue;
				src_idx = src_channel * Image_width * Image_height + i* Image_height + j;
				if (output_data[idx] <= src_data[src_idx]){ 
					output_data[idx] = src_data[src_idx];
					idx_matrix[idx] = (int)src_idx;
				}
			}
		}
	}
	else{
		
		unsigned int src_channel = ix / output_width;
		unsigned int src_idx_ix = (ix % output_width) * stride;
		unsigned int src_idx_iy = iy * stride;

		unsigned int src_idx = src_channel * Image_width * Image_height +  src_idx_ix * Image_height + src_idx_iy;
		output_data[idx] = src_data[src_idx];
		for (int i = (int)src_idx_ix; i < (int)src_idx_ix + (int)pooling_width; i++){
			if (i >= (int)Image_width)break;

			for (int j = (int)src_idx_iy; j < (int)src_idx_iy + (int)pooling_height; j++){
				if (j >= (int)Image_height) break;
				src_idx = src_channel * Image_width * Image_height + i* Image_height + j;
				if (output_data[idx] <= src_data[src_idx]){ 
					output_data[idx] = src_data[src_idx];
					idx_matrix[idx] = (int)src_idx;
				}
			
			}
		}

	}

	return;
}

__global__ void average_Pooling2D(float* output_data, float* src_data,
	unsigned int Image_width, unsigned int Image_height, unsigned int Image_channel,
	unsigned int output_width, unsigned int output_height,
	unsigned int pooling_width, unsigned int pooling_height, unsigned int stride,
	bool padding, unsigned int output_rows, unsigned int output_cols){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * output_rows + iy;

	if (ix >= output_width * Image_channel)return;
	if (iy >= output_height)return;

	//パディングによって、出力行列が変化する
	if (padding){
		unsigned int src_channel = ix / output_width;
		int src_idx_ix = (int)((ix % output_width) * stride) - (int)pooling_width / 2;
		int src_idx_iy = (int)(iy * stride) - (int)pooling_height / 2;
		unsigned int src_idx = 0;

		float val = 0;
		for (int i = src_idx_ix; i < src_idx_ix + (int)pooling_width; i++){
			if (i >= (int)Image_width)break;
			else if (i < 0)continue;
			for (int j = src_idx_iy; j < src_idx_iy + (int)pooling_height; j++){
				if (j >= (int)Image_height)break;
				else if (j < 0)continue;

				src_idx = src_channel * Image_width * Image_height + i* Image_height + j;
				val += src_data[src_idx];
			}
		}
		output_data[idx] = val / (float)(pooling_height * pooling_width);
	}
	else{

		unsigned int src_channel = ix / output_width;
		unsigned int src_idx_ix = (ix % output_width) * stride;
		unsigned int src_idx_iy = iy * stride;

		float val = 0;

		unsigned int src_idx = src_channel * Image_width * Image_height + src_idx_ix * Image_height + src_idx_iy;
		output_data[idx] = src_data[src_idx];
		for (int i = src_idx_ix; i < src_idx_ix + (int)pooling_width; i++){
			if (i >= (int)Image_width)break;
			for (int j = src_idx_iy; j < src_idx_iy + (int)pooling_height; j++){
				if (j >= (int)Image_height) break;
				src_idx = src_channel * Image_width * Image_height + i* Image_height + j;
				val += src_data[src_idx];
			}
		}
		output_data[idx] = val / (float)(pooling_height * pooling_width);
	}

	return;
}

__global__ void backward_for_max_Pooling2D(float* prev_delta, float* src_data, int* idx_matrix,
	unsigned int prev_delta_width, unsigned int prev_delta_height, unsigned int prev_delta_channel,
	unsigned int src_width, unsigned int src_height,
	unsigned int pooling_width, unsigned int pooling_height, unsigned int stride,
	bool padding, unsigned int prev_delta_rows, unsigned int prev_delta_cols){
	
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * prev_delta_rows + iy;

	if (ix >= prev_delta_width * prev_delta_channel)return;
	if (iy >= prev_delta_height)return;

	//スライドさせたときに、もっとも最初に畳み込みされる場所を計算する
	int idx_x = ((int)ix % (int)prev_delta_width) - (int)pooling_width + 1;
	int idx_y = (int)iy - (int)pooling_height + 1;

	//画像のチャネル数を求める(列優先で画像が入っているので、Imageの幅で割ればOK)
	int image_channel_num = ix / prev_delta_width;

	prev_delta[idx] = 0;

	if (padding){
		if (idx_x < -(int)pooling_width / 2)idx_x = -(int)pooling_width / 2;
		if (idx_y < -(int)pooling_height / 2)idx_y = -(int)pooling_height / 2;

		unsigned int height_num = (pooling_height - 1) / stride + 1;
		unsigned int width_num = (pooling_width - 1) / stride + 1;

		for (int i = idx_x; i <= (int)(ix % prev_delta_width); i++){
			//srcにおけるidxを計算する
			unsigned int src_col_num = (i + (int)pooling_width / 2) / stride;
			//printf("(ix,iy) = (%d,%d),cont point: %d  src_col : %d\n", ix, iy, (i + (int)filter_width / 2) % stride, src_col_num);
			if (src_col_num >= width_num)break;
			if ((i + (int)pooling_width / 2) % stride != 0)continue;

			for (int j = idx_y; j <= (int)iy; j++){
				int temp_y = j + pooling_height / 2;
				if (temp_y % stride != 0)continue;
				if (temp_y / stride >= height_num)break;
				//src_dataにおける列番号を計算する
				unsigned int src_cols_idx = temp_y / stride;
				//src_dataにおける行番号を計算する
				//src_dataにおけるidxを計算する
				unsigned int src_idx =  image_channel_num * src_width* src_height + src_col_num * src_height + src_cols_idx;
				if (idx_matrix[src_idx] == idx)prev_delta[idx] += src_data[src_idx];
			}
		}

	}
	else{
		if (idx_x < 0) idx_x = 0;
		if (idx_y < 0) idx_y = 0;
		unsigned int height_num = (prev_delta_height - pooling_height) / stride + 1;
		unsigned int width_num = (prev_delta_width - pooling_width) / stride + 1;

		for (int i = idx_x; i < ix % prev_delta_width; i++){
			//srcにおけるidxを計算する
			unsigned int src_col_num = i / stride;

			//範囲外だと抜ける
			if (src_col_num >= width_num)break;

			//幅が範囲外だと、ループを抜ける
			//起点がストライド幅に合致していなければスルー

			if (i % stride != 0)continue;

			//高さに関してループ
			for (int j = idx_y; j <= iy; j++){

				//起点がstrideに対応していなければ、スルー
				if (j % stride != 0)continue;
				if (j / stride >= height_num)break;
				//src_dataにおける列番号を計算する
				unsigned int src_cols_idx = j / stride;

				//src_dataにおけるidxを計算する
				unsigned int src_idx =  image_channel_num * src_width * src_height +  src_col_num * src_height + src_cols_idx;

				if(idx_matrix[src_idx] == idx)prev_delta[idx] += src_data[src_idx] / (float)(pooling_height * pooling_width);
			}
		}
	}

	return;
}

__global__ void backward_for_average_Pooling2D(float* prev_delta, float* src_data,
	unsigned int prev_delta_width, unsigned int prev_delta_height, unsigned int prev_delta_channel,
	unsigned int src_width, unsigned int src_height,
	unsigned int pooling_width, unsigned int pooling_height, unsigned int stride,
	bool padding, unsigned int prev_delta_rows, unsigned int prev_delta_cols){

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * prev_delta_rows + iy;

	if (ix >= prev_delta_width * prev_delta_channel)return;
	if (iy >= prev_delta_height)return;

	//スライドさせたときに、もっとも最初に畳み込みされる場所を計算する
	int idx_x = ((int)ix % (int)prev_delta_width) - (int)pooling_width + 1;
	int idx_y = (int)iy - (int)pooling_height + 1;

	//画像のチャネル数を求める(列優先で画像が入っているので、Imageの幅で割ればOK)
	int image_channel_num = ix / prev_delta_width;

	prev_delta[idx] = 0;
	float size = (float)(pooling_height * pooling_width);
	if (padding){
		if (idx_x < -(int)pooling_width / 2)idx_x = -(int)pooling_width / 2;
		if (idx_y < -(int)pooling_height / 2)idx_y = -(int)pooling_height / 2;

		unsigned int height_num = (pooling_height - 1) / stride + 1;
		unsigned int width_num = (pooling_width - 1) / stride + 1;

		for (int i = idx_x; i <= (int)(ix % prev_delta_width); i++){
			//srcにおけるidxを計算する
			unsigned int src_col_num = (i + (int)pooling_width / 2) / stride;
			//printf("(ix,iy) = (%d,%d),cont point: %d  src_col : %d\n", ix, iy, (i + (int)filter_width / 2) % stride, src_col_num);
			if (src_col_num >= width_num)break;
			if ((i + (int)pooling_width / 2) % stride != 0)continue;

			for (int j = idx_y; j <= (int)iy; j++){
				int temp_y = j + pooling_height / 2;
				if (temp_y % stride != 0)continue;
				if (temp_y / stride >= height_num)break;
				//src_dataにおける列番号を計算する
				unsigned int src_cols_idx = temp_y / stride;
				//src_dataにおける行番号を計算する
				//src_dataにおけるidxを計算する
				unsigned int src_idx = image_channel_num * src_width* src_height + src_col_num * src_height + src_cols_idx;
				prev_delta[idx] += src_data[src_idx] / size;
			}
		}

	}
	else{
		if (idx_x < 0) idx_x = 0;
		if (idx_y < 0) idx_y = 0;
		unsigned int height_num = (prev_delta_height - pooling_height) / stride + 1;
		unsigned int width_num = (prev_delta_width - pooling_width) / stride + 1;

		for (int i = idx_x; i < ix % prev_delta_width; i++){
			//srcにおけるidxを計算する
			unsigned int src_col_num = i / stride;

			//範囲外だと抜ける
			if (src_col_num >= width_num)break;

			//幅が範囲外だと、ループを抜ける
			//起点がストライド幅に合致していなければスルー

			if (i % stride != 0)continue;

			//高さに関してループ
			for (int j = idx_y; j <= iy; j++){

				//起点がstrideに対応していなければ、スルー
				if (j % stride != 0)continue;
				if (j / stride >= height_num)break;
				//src_dataにおける列番号を計算する
				unsigned int src_cols_idx = j / stride;

				//src_dataにおけるidxを計算する
				unsigned int src_idx = image_channel_num * src_width * src_height + src_col_num * src_height + src_cols_idx;

				prev_delta[idx] += src_data[src_idx] / size;
			}
		}
	}


	return;
}

