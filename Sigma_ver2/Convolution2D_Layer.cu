#include"Convolution2D_Layer.cuh"

Convolution2D_Layer::Convolution2D_Layer(){
	kind_ = CONVOLUTION_LAYER;
	param_.width = 0;
	param_.height = 0;
	param_.filter_num = 0;
	param_.stride = 0;
	param_.func = relu;
	param_.padding = true;
	bias_.initialize(0, 0, ZERO);
	filter_.initialize(0, 0, ZERO);
	net_.initialize(0, 0, ZERO);
	transe_input_.initialize(0, 0, ZERO);
}

Convolution2D_Layer::Convolution2D_Layer(Convolution2D_Layer::Param& param){
	kind_ = CONVOLUTION_LAYER;
	param_.width = param.width;
	param_.height = param.height;
	param_.filter_num = param.filter_num;
	param_.stride = param.stride;
	param_.func = param.func;
	param_.padding = param.padding;
	param_.mu_ = param.mu_;
	param_.stddev_ = param.stddev_;
	param_.init_bias_ = param.init_bias_;
	bias_.initialize(0,0,ZERO);
	filter_.initialize(0, 0, ZERO);
	net_.initialize(0, 0, ZERO);
	transe_input_.initialize(0, 0, ZERO);
}

Convolution2D_Layer::Convolution2D_Layer(unsigned int width, unsigned int height, unsigned int filter_num, unsigned int stride, bool padding, ACTIVATE_FUNCTION func, float mu, float stddev,float init_bias){
	kind_ = CONVOLUTION_LAYER;
	param_.width = width;
	param_.height = height;
	param_.filter_num = filter_num;
	param_.stride = stride;
	param_.func = func;
	param_.padding = padding;
	param_.mu_ = mu;
	param_.stddev_ = stddev;
	param_.init_bias_ = init_bias;
	bias_.initialize(0, 0, ZERO);
	filter_.initialize(0, 0, ZERO);
	net_.initialize(0, 0, ZERO);
	transe_input_.initialize(0, 0, ZERO);
}

Sigma_Matrix* Convolution2D_Layer::forward(Sigma_Matrix* input,NN_MODE mode){


	//入力画像を畳み込みできる形へ変換する。(フィルタとの内積計算を行う。) 
	//Device＿Matrixに格納されている画像の列数を計算する
	unsigned int image_cols = transe_input_.depth();
	unsigned int image_rows = transe_input_.height() * transe_input_.width() *transe_input_.channel();
	dim3 block(BLOCK_X,BLOCK_Y);
	dim3 grid((image_cols + block.x - 1) / block.x, (image_rows + block.y - 1) / block.y);

	//各画像ごとに配置を換える
	for (int i = 0; i < transe_input_.batch_size(); i++){
		replace_array_for_CONV2D << <grid, block >> >(transe_input_.mini_batch_data(i), input->mini_batch_data(i), input->width(), input->height(), input->channel(),
			param_.width, param_.height, param_.stride, param_.padding, image_rows, image_cols);
	}
	//cublasを用いて内積計算(各データに対して内積処理を行う)
	const float alpha = 1.0;
	const float beta = 0;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	//net値の計算を行う(内積処理→biasの計算)
	dim3 block_net(BLOCK_X,BLOCK_Y);
	dim3 grid_net(1, (net_.channel() + block.y - 1) / block.y);
	unsigned int net_row = net_.width()*net_.height();
	for (int i = 0; i < transe_input_.batch_size(); i++){
		
		BLAS_CH(cublasSgemm(handle,
			CUBLAS_OP_T,CUBLAS_OP_N,
			net_row, filter_.cols(), filter_.rows(),
			&alpha,
			transe_input_.mini_batch_data(i),image_rows,
			filter_.data(), filter_.rows(),
			&beta,
			net_.mini_batch_data(i), net_row));
		plus_each_row2D<<<grid_net,block_net>>>(net_.mini_batch_data(i),net_.height() * net_.width(),param_.filter_num,bias_.data());

	}
	CHECK(cudaDeviceSynchronize());
	BLAS_CH(cublasDestroy(handle));

	//output = Active_func(net_);
	apply_Acvtivate_function(param_.func, &net_, &output_);

	//Additional process
	for (int i = 0; i < Process.size(); i++){
		Process[i]->forward(&output_, mode);
	}

	return &output_;
}

Sigma_Matrix* Convolution2D_Layer::initialize(Sigma_Matrix* prev_output){
	cout << "Convolution2D initialize... || filter_num :" << param_.filter_num << ", batch_size : " << prev_output->batch_size() << endl;
	//バイアスの初期化(行数：フィルター数　列数：１)
	bias_.initialize(param_.filter_num,1,CONSTANT,param_.init_bias_);
	
	//フィルターの初期化(行数:fil_width * fil_height * prev_channel, 列数:filter_num)
	filter_.initialize(param_.height,param_.width,prev_output->channel(),1,param_.filter_num,GAUSSIAN,param_.mu_,param_.stddev_);
	//paddingをするかしないかによって、サイズが変化する。
	unsigned int output_width;
	unsigned int output_height;
	if (param_.padding){
		output_width = (prev_output->width() - 1) / param_.stride + 1;
		output_height = (prev_output->height() - 1) / param_.stride + 1;
	}
	else{
		output_width = (prev_output->width() - param_.width) / param_.stride + 1;
		output_height = (prev_output->height() - param_.height) / param_.stride + 1;
	}
	//入力を変換するための行列(各Device_Matrix(行:filter_width * filter_height * prev_output_channel,列：output_height * output_width))
	//depthに各deviceマトリックスの列数を入れている。
	transe_input_.initialize(param_.height, param_.width, prev_output->channel(), output_width * output_height, prev_output->batch_size(), ZERO);

	//ネット値の行列
	net_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);

	//出力の初期化(各列にひとつの画像データが格納されている(各データ(行：output_height 列：output_width * output_channel(filter_num) * depth(1) )))
	output_.initialize(output_height,output_width,param_.filter_num,1,prev_output->batch_size(),ZERO);

	CHECK(cudaDeviceSynchronize());

	return &output_;
}

Sigma_Matrix* Convolution2D_Layer::change_batchsize(Sigma_Matrix* prev_output){
	//paddingをするかしないかによって、サイズが変化する。
	unsigned int output_width;
	unsigned int output_height;
	if (param_.padding){
		output_width = (prev_output->width() - 1) / param_.stride + 1;
		output_height = (prev_output->height() - 1) / param_.stride + 1;
	}
	else{
		output_width = (prev_output->width() - param_.width) / param_.stride + 1;
		output_height = (prev_output->height() - param_.height) / param_.stride + 1;
	}

	//ネット値の行列
	net_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);
	//出力の初期化(各列にひとつの画像データが格納されている
	output_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);
	//transe_inputを初期化する
	transe_input_.initialize(param_.height, param_.width, prev_output->channel(), output_width * output_height, prev_output->batch_size(), ZERO);
	//デルタ行列のサイズ変更
	delta_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);
	//微分出力のサイズ変更
	d_output_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);

	return &output_;
}

Sigma_Matrix* Convolution2D_Layer::get_d_output(){
	//output　＝ Active_d_func(net_);
	apply_d_Acvtivate_function(param_.func,&net_,&d_output_);
	return &d_output_;
}

Sigma_Matrix* Convolution2D_Layer::initialize_optimizer(Sigma_Matrix* prev_output){
	delta_.initialize(output_.height(),output_.width(),output_.channel(),output_.depth(),output_.batch_size(),ZERO);
	update_filter_.initialize(filter_.rows(),filter_.cols(),ZERO);
	transe_delta_.initialize(param_.height, param_.width, prev_output->channel(),transe_input_.depth(), prev_output->batch_size(), ZERO);
	update_bias_.initialize(bias_.rows(),1,ZERO);
	CONSTANT_1_.initialize(bias_.rows(),1,CONSTANT,1.0f);
	d_output_.initialize(output_.height(), output_.width(), output_.channel(), output_.depth(), output_.batch_size(), ZERO);
	return &output_;
}

void Convolution2D_Layer::back_ward(Sigma_Matrix* prev_delta,Sigma_Matrix* prev_d_output){
	//cublasを用いて内積計算(各データに対して内積処理を行うため)
	const float alpha = 1.0;
	const float beta = 0;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	dim3 block(BLOCK_X,BLOCK_Y);
	dim3 grid((prev_delta->width()*prev_delta->channel() + block.x - 1) / block.x, (prev_delta->height()  + block.y - 1) / block.y);

	unsigned int mb_prev_delta_cols = delta_.width()* delta_.height();
	//cout << "OPA.row : " << filter_.rows() << " OPB.col : " << mb_prev_delta_cols << " OPA.col : " << filter_.cols() << endl;
	//cout << "filter_rows() : " << filter_.rows() << " delta_.height() : " << delta_.height() << " transe_delta_.height()" << transe_delta_.height() << endl;
	//各データごとにdeltaを逆伝播する(transe_delta_ = filter * transepose(delta))
	for (int i = 0; i < prev_delta->batch_size(); i++){
		BLAS_CH(cublasSgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_T,
			filter_.rows(),mb_prev_delta_cols, filter_.cols(),
			&alpha,
			filter_.data(), filter_.rows(),
			delta_.mini_batch_data(i), mb_prev_delta_cols,
			&beta,
			transe_delta_.mini_batch_data(i), transe_delta_.height()*transe_delta_.width()*transe_delta_.channel()));

		//prev_deltaにデータを転送する
		replace_array_for_backward_CONV2D << <grid, block >> >(prev_delta->mini_batch_data(i),transe_delta_.mini_batch_data(i) ,
			prev_delta->width(), prev_delta->height(), prev_delta->channel(),
			param_.width, param_.height, param_.stride, param_.padding,
			prev_delta->height(), prev_delta->width()*prev_delta->channel(),
			transe_delta_.height(), transe_delta_.width()*transe_delta_.channel());
	}

	//微分した値を各要素に掛け合わせる
	grid.x = (prev_d_output->cols() + block.x - 1) / block.x;
	grid.y = (prev_d_output->rows() + block.y - 1) / block.y;

	multiply2D << <grid, block >> >(prev_delta->data(), prev_delta->rows(), prev_delta->cols(), prev_d_output->data());

	CHECK(cudaDeviceSynchronize());
	BLAS_CH(cublasDestroy(handle));

	return;
}

Sigma_Matrix* Convolution2D_Layer::calc_update_param(Sigma_Matrix* prev_output){
	//フィルタの更新料を計算する
	float alpha = optimizer_.learning_rate() * (1.0f / (float)delta_.batch_size());
	//最初はデルタを0にする
	const float beta = 0;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));
	//バッチサイズ回だけ計算を行う(最初はbeta0 その後はbeta = 1)(update_filter)
	BLAS_CH(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
		update_filter_.rows(), update_filter_.cols(), delta_.width()*delta_.height(),
		&alpha,
		transe_input_.mini_batch_data(0), update_filter_.rows(),
		delta_.mini_batch_data(0), delta_.height() * delta_.width(),
		&beta,
		update_filter_.data(), update_filter_.rows()));
	const float beta2 = 1;
	
	for (int i = 1; i < delta_.batch_size(); i++){
		BLAS_CH(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			update_filter_.rows(), update_filter_.cols(), delta_.width()*delta_.height(),
			&alpha,
			transe_input_.mini_batch_data(i), update_filter_.rows(),
			delta_.mini_batch_data(i), delta_.height() * delta_.width(),
			&beta2,
			update_filter_.data(), update_filter_.rows()));
	}
	//バイアスの計算
	//バッチサイズ回だけ計算を行う(最初はbeta0 その後はbeta = 1)(update_filter)
	BLAS_CH(cublasSgemv(handle, CUBLAS_OP_T,
		delta_.width()*delta_.height(), delta_.channel(),
		&alpha,
		delta_.mini_batch_data(0), delta_.height() * delta_.width(),
		CONSTANT_1_.data(), 1,
		&beta,
		update_bias_.data(), 1));

	for (int i = 1; i < delta_.batch_size();i++){
		BLAS_CH(cublasSgemv(handle, CUBLAS_OP_T,
			delta_.width()*delta_.height(), delta_.channel(),
			&alpha,
			delta_.mini_batch_data(i), delta_.height() * delta_.width(),
			CONSTANT_1_.data(), 1,
			&beta,
			update_bias_.data(), 1));
	}

	CHECK(cudaDeviceSynchronize());
	BLAS_CH(cublasDestroy(handle));
	return &output_;
}

void Convolution2D_Layer::update_param(){
	//cublasを用いて各要素を引く(重みの更新)
	float alpha = 1.0f;
	float beta = -1.0f;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	//重みの更新
	BLAS_CH(cublasSgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		filter_.rows(), filter_.cols(),
		&alpha,
		filter_.data(), filter_.rows(),
		&beta,
		update_filter_.data(), update_filter_.rows(),
		filter_.data(), filter_.rows()));

	//バイアスの更新
	BLAS_CH(cublasSgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		bias_.rows(), bias_.cols(),
		&alpha,
		bias_.data(), bias_.rows(),
		&beta,
		update_bias_.data(), update_bias_.rows(),
		bias_.data(), bias_.rows()));

	CHECK(cudaDeviceSynchronize());
	BLAS_CH(cublasDestroy(handle));
	return;
}
