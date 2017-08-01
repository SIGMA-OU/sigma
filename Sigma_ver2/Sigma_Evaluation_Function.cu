#include"Sigma_Evaluation_Function.cuh"

vector<int> argmax_idx(Sigma_Matrix& matrix){
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	vector<int> idx_array(matrix.cols());
	int idx = 0;
	for (int i = 0; i < matrix.cols(); i++){
		BLAS_CH(cublasIsamax(handle,matrix.rows(),
			matrix.mini_batch_data(i),1,&idx));
		idx_array[i] = idx - 1;
	}
	BLAS_CH(cublasDestroy(handle));
	return idx_array;
}

int equall(vector<int>& label, vector<int>& correct){
	int equall_num = 0;
	for (int i = 0; i < label.size(); i++){
		if (label[i] == correct[i])equall_num++;
	}
	return equall_num;
}

Sigma_Matrix make_onehot_matrix(vector<int>& label_data, unsigned int class_num){
	Sigma_Matrix tmp_matrix(class_num,label_data.size(),ZERO);
	for (int i = 0; i < label_data.size(); i++){
		if (label_data[i] < class_num){
			tmp_matrix.set_value(label_data[i],i,1.0f);
		}
	}
	return tmp_matrix;
}

void next_batch(unsigned int size, unsigned int idx, Sigma_Matrix& src_matrix, Sigma_Matrix& mini_batch_matrix){

	mini_batch_matrix.initialize(src_matrix.height(),src_matrix.width(),src_matrix.channel(),src_matrix.depth(),size,ZERO);

	int idx_num = idx * size;
	size_t nbyte =  src_matrix.rows() * sizeof(float);
	for (int i = 0; i < size; i++){
		int data_num = (idx_num + i) % src_matrix.cols();
		CHECK(cudaMemcpy(mini_batch_matrix.mini_batch_data(i),src_matrix.mini_batch_data(idx_num),nbyte,cudaMemcpyDeviceToDevice));
	}
	return;
}