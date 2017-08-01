#include"Fully_Connect_Layer.cuh"

Fully_Connect_Layer::Fully_Connect_Layer(){
	param_.func = sig;
	kind_ = FULLY_CONNECT_LAYER;
	param_.nodes = 0;
	bias_.initialize(0, 0,ZERO);
	weight_.initialize(0, 0 ,ZERO);
	net_.initialize(0, 0 , ZERO);
	delta_.initialize(0, 0, ZERO);
	d_output_.initialize(0, 0, ZERO);
	update_weight_.initialize(0, 0 , ZERO);
	update_bias_.initialize(0, 0, ZERO);
	CONSTANT_1_.initialize(0, 0, ZERO);
}

Fully_Connect_Layer::Fully_Connect_Layer(Fully_Connect_Layer::Param &param){
	kind_ = FULLY_CONNECT_LAYER;
	param_.func = param.func;
	param_.nodes = param.nodes;
	param_.mu_ = param.mu_;
	param_.stddev_ = param.stddev_;
	param_.init_bias_ = param.init_bias_;
	bias_.initialize(0, 0,ZERO);
	weight_.initialize(0, 0,ZERO);
	net_.initialize(0, 0, ZERO);
	delta_.initialize(0, 0, ZERO);
	d_output_.initialize(0, 0, ZERO);
	update_weight_.initialize(0, 0, ZERO);
	update_bias_.initialize(0, 0, ZERO);
	CONSTANT_1_.initialize(0, 0, ZERO);
}

Fully_Connect_Layer::Fully_Connect_Layer(int node_num, ACTIVATE_FUNCTION func, float mu, float stddev,float init_bias){
	kind_ = FULLY_CONNECT_LAYER;
	param_.func = func;
	param_.nodes = node_num;
	param_.mu_ = mu;
	param_.stddev_ = stddev;
	param_.init_bias_ = init_bias;
	bias_.initialize(0, 0, ZERO);
	weight_.initialize(0, 0, ZERO);
	net_.initialize(0, 0, ZERO);
	delta_.initialize(0, 0, ZERO);
	d_output_.initialize(0, 0, ZERO);
	update_weight_.initialize(0, 0, ZERO);
	update_bias_.initialize(0, 0, ZERO);
	CONSTANT_1_.initialize(0, 0, ZERO);
}

Sigma_Matrix* Fully_Connect_Layer::forward(Sigma_Matrix* input, NN_MODE mode){

	//cublasを用いて内積計算
	const float alpha = 1.0;
	const float beta = 0;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	//ネット値の計算をする(net = transpose(W) * input)
	BLAS_CH(cublasSgemm(handle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		weight_.cols(), input->cols(), weight_.rows(),
		&alpha,
		weight_.data(), weight_.rows(),
		input->data(), input->rows(),
		&beta,
		net_.data(), net_.rows()));

	//net += bias
	dim3 block(BLOCK_X,BLOCK_Y);
	dim3 grid((net_.cols() + block.x - 1) / block.x, (net_.rows() + block.y - 1) / block.y);

	plus_each_col2D << <grid, block >> >(net_.data(),net_.rows(),net_.cols(),bias_.data());
	CHECK(cudaDeviceSynchronize());

	//output = Active_func(net_);
	apply_Acvtivate_function(param_.func,&net_,&output_);

	//Additional process
	for (int i = 0; i < Process.size(); i++){
		Process[i]->forward(&output_, mode);
	}
	BLAS_CH(cublasDestroy(handle));
	return &output_;

}

Sigma_Matrix* Fully_Connect_Layer::initialize(Sigma_Matrix* prev_output){
	cout << "Fully_Connect initialize... || nodes :" << param_.nodes << ", batch_size : " << prev_output->cols() << endl;
	weight_.initialize(prev_output->rows(),param_.nodes,GAUSSIAN,param_.mu_,param_.stddev_);
	bias_.initialize(param_.nodes,1,CONSTANT,param_.init_bias_);
	net_.initialize(param_.nodes,prev_output->cols(),ZERO);
	output_.initialize(param_.nodes,prev_output->cols(),ZERO);
	return &output_;
}

Sigma_Matrix* Fully_Connect_Layer::change_batchsize(Sigma_Matrix* prev_output){
	net_.initialize(param_.nodes, prev_output->cols(), ZERO);
	output_.initialize(param_.nodes, prev_output->cols(), ZERO);
	delta_.initialize(param_.nodes, prev_output->cols(), ZERO);
	d_output_.initialize(param_.nodes, prev_output->cols(), ZERO);
	return &output_;
}

Sigma_Matrix* Fully_Connect_Layer::get_d_output(){
	//output　＝ Active_func(net_);
	apply_d_Acvtivate_function(param_.func, &net_, &d_output_);
	return &d_output_;
}

Sigma_Matrix* Fully_Connect_Layer::initialize_optimizer(Sigma_Matrix* prev_output){
	delta_.initialize(output_.rows(),output_.cols(),ZERO);
	update_weight_.initialize(weight_.rows(),weight_.cols(),ZERO);
	update_bias_.initialize(bias_.rows(),1,ZERO);
	CONSTANT_1_.initialize(bias_.rows(),1,CONSTANT,1.0f);
	d_output_.initialize(output_.rows(),output_.cols(),ZERO);
	return &output_;
}

void Fully_Connect_Layer::back_ward(Sigma_Matrix* prev_delta,Sigma_Matrix* prev_d_output){
	//cublasを用いて内積計算(prev_delta = delta * traspose(W))
	const float alpha = 1.0;
	const float beta = 0;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	//prev_delta = weight * delta
	BLAS_CH(cublasSgemm(handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						weight_.rows(), delta_.cols(), weight_.cols(),
						&alpha,
						weight_.data(), weight_.rows(),
						delta_.data(), delta_.rows(),
						&beta,
						prev_delta->data(), prev_delta->rows()));

	//要素ごとにprev_d_outputを掛け合わせる
	dim3 block(BLOCK_X,BLOCK_Y);
	dim3 grid((prev_delta->cols() + block.x - 1) / block.x, (prev_delta->rows() + block.y - 1) / block.y);
	multiply2D << <grid, block >> >(prev_delta->data(),prev_delta->rows(),prev_delta->cols(),prev_d_output->data());
	CHECK(cudaDeviceSynchronize());
	BLAS_CH(cublasDestroy(handle));
	return;
}

Sigma_Matrix* Fully_Connect_Layer::calc_update_param(Sigma_Matrix* prev_output){
	switch (optimizer_.opt_)
	{
	case SGD:
	{
		float alpha = optimizer_.learning_rate() * (1.0f / (float)output_.batch_size());
		float beta = 0;
		cublasHandle_t handle;
		BLAS_CH(cublasCreate(&handle));

		//Weightの更新量(W_updata = prev_output * transpose(delta))
		BLAS_CH(cublasSgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_T,
			prev_output->rows(), delta_.rows(), prev_output->cols(),
			&alpha,
			prev_output->data(), prev_output->rows(),
			delta_.data(), delta_.rows(),
			&beta,
			update_weight_.data(), update_weight_.rows()));
		//Biasの更新量(bias_update = transpose(delta) * Constatant_1)
		BLAS_CH(cublasSgemv(handle, CUBLAS_OP_T,
			delta_.rows(), delta_.cols(),
			&alpha,
			delta_.data(), delta_.rows(),
			CONSTANT_1_.data(), 1,
			&beta,
			update_bias_.data(), 1));

		CHECK(cudaDeviceSynchronize());
		BLAS_CH(cublasDestroy(handle));
		break;
	}
	case SGD_MOMENTUM:
	{
		float alpha = optimizer_.learning_rate() * (1.0f / (float)output_.batch_size());
		float beta = optimizer_.param_.momentum_sgd_.momentum;
		cublasHandle_t handle;
		BLAS_CH(cublasCreate(&handle));

		//Weightの更新量(W_updata = prev_output * transpose(delta))
		BLAS_CH(cublasSgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_T,
			prev_output->rows(), delta_.rows(), prev_output->cols(),
			&alpha,
			prev_output->data(), prev_output->rows(),
			delta_.data(), delta_.rows(),
			&beta,
			update_weight_.data(), update_weight_.rows()));
		//Biasの更新量(bias_update = transpose(delta) * Constatant_1)
		BLAS_CH(cublasSgemv(handle, CUBLAS_OP_T,
			delta_.rows(), delta_.cols(),
			&alpha,
			delta_.data(), delta_.rows(),
			CONSTANT_1_.data(), CONSTANT_1_.rows(),
			&beta,
			update_bias_.data(), update_bias_.rows()));
		BLAS_CH(cublasDestroy(handle));
		CHECK(cudaDeviceSynchronize());
	}
	case ADAM:
		break;
	default:
		break;
	}
	return &output_;
}

void Fully_Connect_Layer::update_param(){
	//cublasを用いて各要素を引く(重みの更新)
	float alpha = 1.0f;
	float beta = -1.0f;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));
	
	//重みの更新
	BLAS_CH(cublasSgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		weight_.rows(),weight_.cols(),
		&alpha,
		weight_.data(),weight_.rows(),
		&beta,
		update_weight_.data(),update_weight_.rows(),
		weight_.data(),weight_.rows()));

	//バイアスの更新
	BLAS_CH(cublasSgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		bias_.rows(), bias_.cols(),
		&alpha,
		bias_.data(), bias_.rows(),
		&beta,
		update_bias_.data(), update_bias_.rows(),
		bias_.data(), bias_.rows()));
	BLAS_CH(cublasDestroy(handle));
	CHECK(cudaDeviceSynchronize());
	return;
}

void Fully_Connect_Layer::add_process(Additional_Process& ap){
	Process.push_back(&ap);
	return;
}