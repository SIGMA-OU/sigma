#include"Utilities.cuh"

#include"Utilities.cuh"

float uniform(){
	float ret = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
	return ret;
}

float gaussian(float mu, float sigma){
	float  z = sqrt(-2.0f * log(uniform())) *sin(2.0 * 3.1415 * uniform());
	return mu + sigma * z;
}


__global__ void c2r_major_change2D(float* d_data, float* src_data, unsigned int rows, unsigned int cols){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int col_idx = ix * rows + iy;
	unsigned int row_idx = iy * cols + ix;
	if (ix < cols && iy < rows){
		d_data[row_idx] = src_data[col_idx];
	}
}

__global__ void r2c_major_change2D(float* d_data, float* src_data, unsigned int rows, unsigned int cols){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int col_idx = ix * rows + iy;
	unsigned int row_idx = iy * cols + ix;
	if (ix < cols && iy < rows){
		d_data[col_idx] = src_data[row_idx];
	}
}

__global__ void plus_each_col2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		d_data[idx] += src_data[iy];
	}
}

__global__ void plus_each_row2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		d_data[idx] += src_data[ix];
	}
}


__global__ void multiply2D(float* d_data, unsigned int rows, unsigned int cols, float* src_data){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		d_data[idx] *= src_data[idx];
	}
}

__global__ void subtractArray2D(float *subtruct_data, float *src_data, unsigned int rows, unsigned int cols){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;

	if (ix < cols && iy < rows){
		src_data[idx] = src_data[idx] - subtruct_data[idx];
	}
}

__global__ void subtractArray2D(float *subtruct_data, float *src_data, float *result_data, unsigned int rows, unsigned int cols){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;

	if (ix < cols && iy < rows){
		result_data[idx] = src_data[idx] - subtruct_data[idx];
	}
}
