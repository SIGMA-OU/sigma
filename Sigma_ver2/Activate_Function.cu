#include"Activate_Function.cuh"

__device__
float AC_sigmoid(float x){
	return (1.0f / (1.0f + exp(-x)));
}

__device__
float AC_softmax_for_exp(float x){
	return exp(x);
}

__device__
float AC_differential_sigmoid(float x){
	float a = AC_sigmoid(x);
	return (1.0f - a) * a;
}

__device__
float AC_tanh(float x){
	return tanh(x);
}

__device__
float AC_differential_tanh(float x){
	float a = tanh(x);
	return 1 - a*a;
}

__device__
float AC_identity(float x){
	return x;
}

__device__
float AC_differential_identity(float x){
	return 1.0f;
}

__device__
float AC_relu(float x){
	if (x >= 0) return x;
	else return 0;
}

__device__
float AC_differential_relu(float x){
	if (x >= 0) return 1;
	else return 0;
}

__device__
float AC_plane(float x){
	if (x >= 0) return x;
	else return 0.1 * x;
}

__device__
float AC_differential_plane(float x){
	if (x >= 0) return 1;
	else return 0.1;
}

__global__
void apply_functions2D(ACTIVATE_FUNCTION func, float* src_data, unsigned int rows, unsigned int cols, float* result_data){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		switch (func)
		{
		case hytan:
			result_data[idx] = AC_tanh(src_data[idx]);
			return;
			break;
		case sig:
			result_data[idx] = AC_sigmoid(src_data[idx]);
			return;
			break;
		case iden:
			result_data[idx] = AC_identity(src_data[idx]);
			return;
			break;
		case relu:
			result_data[idx] = AC_relu(src_data[idx]);
			break;
		case softmax:
			result_data[idx] = AC_softmax_for_exp(src_data[idx]);
			break;
		case plane:
			result_data[idx] = AC_plane(src_data[idx]);
			break;
		default:
			break;
		}
	}
	return;
}

__global__
void apply_differnetial_functions2D(ACTIVATE_FUNCTION func, float* src_data, unsigned int rows, unsigned int cols, float* result_data){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = ix * rows + iy;
	if (ix < cols && iy < rows){
		switch (func)
		{
		case hytan:
			result_data[idx] = AC_differential_tanh(src_data[idx]);
			return;
			break;
		case sig:
			result_data[idx] = AC_differential_sigmoid(src_data[idx]);
			return;
			break;
		case iden:
			result_data[idx] = AC_differential_identity(src_data[idx]);
			return;
			break;
		case relu:
			result_data[idx] = AC_differential_relu(src_data[idx]);
			break;
		case softmax:
			break;
		case plane:
			result_data[idx] = AC_differential_plane(src_data[idx]);
			break;
		default:
			break;
		}
	}
	return;
}

void apply_Acvtivate_function(ACTIVATE_FUNCTION func, Sigma_Matrix* src_matrix, Sigma_Matrix* result_matrix){

	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((src_matrix->cols() + block.x - 1) / block.x, (src_matrix->rows() + block.y - 1) / block.y);

	if (func == softmax){
		//cublasを用いる
		const float alpha = 1.0;
		const float beta = 0;
		cublasHandle_t handle;
		BLAS_CH(cublasCreate(&handle));

		float* sum_result;
		int nByte = src_matrix->batch_size() * sizeof(float);
		sum_result = (float*)malloc(nByte);
		apply_functions2D << <grid, block >> >(softmax, src_matrix->data(), src_matrix->rows(), src_matrix->cols(), result_matrix->data());
		//各バッチごとに合計を算出する
		for (int i = 0; i < src_matrix->batch_size(); i++){
			BLAS_CH(cublasSasum(handle,result_matrix->rows(),result_matrix->mini_batch_data(i),1,&sum_result[i]));
		}
		for (int i = 0; i < src_matrix->batch_size(); i++){
			const float alpha2 = 1.0f / sum_result[i];
			BLAS_CH(cublasSscal(handle,result_matrix->rows(), &alpha2, result_matrix->mini_batch_data(i),1));
		}
		BLAS_CH(cublasDestroy(handle));
		CHECK(cudaDeviceSynchronize());
		free(sum_result);

	}
	else{
	apply_functions2D << <grid, block >> >(func, src_matrix->data(), src_matrix->rows(), src_matrix->cols(), result_matrix->data());
	CHECK(cudaDeviceSynchronize());
	}
	return;
}

void apply_d_Acvtivate_function(ACTIVATE_FUNCTION func, Sigma_Matrix* src_matrix, Sigma_Matrix* result_matrix){
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((src_matrix->cols() + block.x - 1) / block.x, (src_matrix->rows() + block.y - 1) / block.y);
	apply_differnetial_functions2D << <grid, block >> >(func, src_matrix->data(), src_matrix->rows(), src_matrix->cols(), result_matrix->data());
	CHECK(cudaDeviceSynchronize());
	return;
}