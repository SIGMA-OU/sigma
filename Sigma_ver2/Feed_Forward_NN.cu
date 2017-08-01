#include"Feed_Forward_NN.cuh"

Feed_Forward_NN::Feed_Forward_NN(){
	training_num_ = 0;
	mode_ = TRAIN;
	loss_ = CROSS_ENTROPY;
	opt_ = SGD;
}

Feed_Forward_NN::Feed_Forward_NN(Input_Layer& first_layer){
	training_num_ = 0;
	mode_ = TRAIN;
	loss_ = CROSS_ENTROPY;
	opt_ = SGD;
	input_ = &first_layer;
}

void Feed_Forward_NN::add_layer(Abstract_Middle_Layer& Layer){
	layers_.push_back(&Layer);
	return;
}

Sigma_Matrix& Feed_Forward_NN::infer(Sigma_Matrix& input_data){
	mode_ = INFER;
	if (input_->output().batch_size() == 0){
		cout << "Start initializing Neural Model" << endl;
		Sigma_Matrix* ptr;
		ptr = &input_data;
		ptr = input_->initialize(ptr);
		for (int i = 0; i < layers_.size(); i++){
			cout << i << ":";
			ptr = layers_[i]->initialize(ptr);
		}
		cout << "Initializing Success!!" << endl;
	}
	else if (input_data.batch_size() != input_->output().batch_size()){
		Sigma_Matrix* ptr;
		ptr = &input_data;
		ptr = input_->change_batchsize(ptr);
		for (int i = 0; i < layers_.size(); i++){
			ptr = layers_[i]->change_batchsize(ptr);
		}
	}

	Sigma_Matrix* ptr;
	ptr = &input_data;
	ptr = input_->forward(ptr, mode_);
	for (int i = 0; i < layers_.size(); i++){
		ptr = layers_[i]->forward(ptr, mode_);
	}
	return *ptr;
}

void Feed_Forward_NN::forward(Sigma_Matrix* input_data){
	mode_ = TRAIN;
	if (input_->output().batch_size() == 0){
		cout << "Start initializing Neural Model" << endl;
		Sigma_Matrix* ptr;
		ptr = input_data;
		ptr = input_->initialize(ptr);
		for (int i = 0; i < layers_.size(); i++){
			cout << i << ":";
			ptr = layers_[i]->initialize(ptr);
		}
		cout << "Initializing Success!!" << endl;
	}
	else if (input_data->batch_size() != input_->output().batch_size()){
		Sigma_Matrix* ptr;
		ptr = input_data;
		ptr = input_->change_batchsize(ptr);
		for (int i = 0; i < layers_.size(); i++){
			ptr = layers_[i]->change_batchsize(ptr);
		}
	}
	Sigma_Matrix* ptr;
	ptr = input_data;
	ptr = input_->forward(ptr, mode_);
	for (int i = 0; i < layers_.size(); i++){
		ptr = layers_[i]->forward(ptr, mode_);
	}
	return;
}

void Feed_Forward_NN::backward(){
	//Optimizer‚ªinitialize‚³‚ê‚Ä‚¢‚é‚©‚ÌŠm”F
	if (layers_[layers_.size() - 1]->get_delta()->batch_size() == 0){
		Sigma_Matrix* ptr;
		ptr = input_->get_output();
		for (int i = 0; i < layers_.size(); i++){
			ptr = layers_[i]->initialize_optimizer(ptr);
		}
	}

	//o—Í‘w‚Å‚Ìƒfƒ‹ƒ^ŒvŽZ
	float alpha = 1.0f;
	float beta = -1.0f;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	BLAS_CH(cublasSgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		layers_[layers_.size() - 1]->get_output()->rows(), layers_[layers_.size() - 1]->get_output()->cols(),
		&alpha,
		layers_[layers_.size() - 1]->get_output()->data(), layers_[layers_.size() - 1]->get_output()->rows(),
		&beta,
		teacher_data_->data(), teacher_data_->rows(),
		layers_[layers_.size() - 1]->get_delta()->data(), layers_[layers_.size() - 1]->get_delta()->rows()));

	//o—Í‘w‚Å‚Ìƒfƒ‹ƒ^ŒvŽZ
	/*int nx = layers_[layers_.size() - 1]->get_output()->cols();
	int ny = layers_[layers_.size() - 1]->get_output()->rows();
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	subtractArray2D << <grid, block >> >(teacher_data_->data(), layers_[layers_.size() - 1]->get_output()->data(), layers_[layers_.size() - 1]->get_delta()->data(), ny, nx);
	CHECK(cudaDeviceSynchronize());*/
	CHECK(cudaDeviceSynchronize());
	BLAS_CH(cublasDestroy(handle));

	//Œë·‚Ì“`”d
	for (int i = layers_.size() - 1; i > 0; i--){
		layers_[i]->back_ward(layers_[i - 1]->get_delta(), layers_[i - 1]->get_d_output());
	}
}

void Feed_Forward_NN::calc_update_param(){
	//ƒpƒ‰ƒ[ƒ^•ÏX—Ê‚ÌŒvŽZ
	Sigma_Matrix* ptr;
	ptr = input_->get_output();
	for (int i = 0; i < layers_.size(); i++){
		ptr = layers_[i]->calc_update_param(ptr);
	}
}

void Feed_Forward_NN::update(){
	for (int i = 0; i < layers_.size(); i++){
		layers_[i]->update_param();
	}
}

void Feed_Forward_NN::set_optimizer(Optimizer& opt){
	for(int i = 1; i < layers_.size(); i++){
		layers_[i]->set_optimizer(opt);
	}
}

float Feed_Forward_NN::learn(Sigma_Matrix& input_data,Sigma_Matrix& teacher_data){
	teacher_data_ = &teacher_data;
	forward(&input_data);
	backward();
	calc_update_param();
	update();

	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));
	unsigned int data_num = layers_[layers_.size() - 1]->get_delta()->rows() * layers_[layers_.size() - 1]->get_delta()->cols();
	float result = 0;
	BLAS_CH(cublasSasum(handle,data_num,layers_[layers_.size() - 1]->get_delta()->data(),1,&result));
	BLAS_CH(cublasDestroy(handle));
	return result / (float)layers_[layers_.size() - 1]->get_delta()->batch_size();
}