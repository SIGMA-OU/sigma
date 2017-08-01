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


	//���͉摜����ݍ��݂ł���`�֕ϊ�����B(�t�B���^�Ƃ̓��όv�Z���s���B) 
	//Device�QMatrix�Ɋi�[����Ă���摜�̗񐔂��v�Z����
	unsigned int image_cols = transe_input_.depth();
	unsigned int image_rows = transe_input_.height() * transe_input_.width() *transe_input_.channel();
	dim3 block(BLOCK_X,BLOCK_Y);
	dim3 grid((image_cols + block.x - 1) / block.x, (image_rows + block.y - 1) / block.y);

	//�e�摜���Ƃɔz�u��������
	for (int i = 0; i < transe_input_.batch_size(); i++){
		replace_array_for_CONV2D << <grid, block >> >(transe_input_.mini_batch_data(i), input->mini_batch_data(i), input->width(), input->height(), input->channel(),
			param_.width, param_.height, param_.stride, param_.padding, image_rows, image_cols);
	}
	//cublas��p���ē��όv�Z(�e�f�[�^�ɑ΂��ē��Ϗ������s��)
	const float alpha = 1.0;
	const float beta = 0;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	//net�l�̌v�Z���s��(���Ϗ�����bias�̌v�Z)
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
	//�o�C�A�X�̏�����(�s���F�t�B���^�[���@�񐔁F�P)
	bias_.initialize(param_.filter_num,1,CONSTANT,param_.init_bias_);
	
	//�t�B���^�[�̏�����(�s��:fil_width * fil_height * prev_channel, ��:filter_num)
	filter_.initialize(param_.height,param_.width,prev_output->channel(),1,param_.filter_num,GAUSSIAN,param_.mu_,param_.stddev_);
	//padding�����邩���Ȃ����ɂ���āA�T�C�Y���ω�����B
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
	//���͂�ϊ����邽�߂̍s��(�eDevice_Matrix(�s:filter_width * filter_height * prev_output_channel,��Foutput_height * output_width))
	//depth�Ɋedevice�}�g���b�N�X�̗񐔂����Ă���B
	transe_input_.initialize(param_.height, param_.width, prev_output->channel(), output_width * output_height, prev_output->batch_size(), ZERO);

	//�l�b�g�l�̍s��
	net_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);

	//�o�͂̏�����(�e��ɂЂƂ̉摜�f�[�^���i�[����Ă���(�e�f�[�^(�s�Foutput_height ��Foutput_width * output_channel(filter_num) * depth(1) )))
	output_.initialize(output_height,output_width,param_.filter_num,1,prev_output->batch_size(),ZERO);

	CHECK(cudaDeviceSynchronize());

	return &output_;
}

Sigma_Matrix* Convolution2D_Layer::change_batchsize(Sigma_Matrix* prev_output){
	//padding�����邩���Ȃ����ɂ���āA�T�C�Y���ω�����B
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

	//�l�b�g�l�̍s��
	net_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);
	//�o�͂̏�����(�e��ɂЂƂ̉摜�f�[�^���i�[����Ă���
	output_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);
	//transe_input������������
	transe_input_.initialize(param_.height, param_.width, prev_output->channel(), output_width * output_height, prev_output->batch_size(), ZERO);
	//�f���^�s��̃T�C�Y�ύX
	delta_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);
	//�����o�͂̃T�C�Y�ύX
	d_output_.initialize(output_height, output_width, param_.filter_num, 1, prev_output->batch_size(), ZERO);

	return &output_;
}

Sigma_Matrix* Convolution2D_Layer::get_d_output(){
	//output�@�� Active_d_func(net_);
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
	//cublas��p���ē��όv�Z(�e�f�[�^�ɑ΂��ē��Ϗ������s������)
	const float alpha = 1.0;
	const float beta = 0;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	dim3 block(BLOCK_X,BLOCK_Y);
	dim3 grid((prev_delta->width()*prev_delta->channel() + block.x - 1) / block.x, (prev_delta->height()  + block.y - 1) / block.y);

	unsigned int mb_prev_delta_cols = delta_.width()* delta_.height();
	//cout << "OPA.row : " << filter_.rows() << " OPB.col : " << mb_prev_delta_cols << " OPA.col : " << filter_.cols() << endl;
	//cout << "filter_rows() : " << filter_.rows() << " delta_.height() : " << delta_.height() << " transe_delta_.height()" << transe_delta_.height() << endl;
	//�e�f�[�^���Ƃ�delta���t�`�d����(transe_delta_ = filter * transepose(delta))
	for (int i = 0; i < prev_delta->batch_size(); i++){
		BLAS_CH(cublasSgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_T,
			filter_.rows(),mb_prev_delta_cols, filter_.cols(),
			&alpha,
			filter_.data(), filter_.rows(),
			delta_.mini_batch_data(i), mb_prev_delta_cols,
			&beta,
			transe_delta_.mini_batch_data(i), transe_delta_.height()*transe_delta_.width()*transe_delta_.channel()));

		//prev_delta�Ƀf�[�^��]������
		replace_array_for_backward_CONV2D << <grid, block >> >(prev_delta->mini_batch_data(i),transe_delta_.mini_batch_data(i) ,
			prev_delta->width(), prev_delta->height(), prev_delta->channel(),
			param_.width, param_.height, param_.stride, param_.padding,
			prev_delta->height(), prev_delta->width()*prev_delta->channel(),
			transe_delta_.height(), transe_delta_.width()*transe_delta_.channel());
	}

	//���������l���e�v�f�Ɋ|�����킹��
	grid.x = (prev_d_output->cols() + block.x - 1) / block.x;
	grid.y = (prev_d_output->rows() + block.y - 1) / block.y;

	multiply2D << <grid, block >> >(prev_delta->data(), prev_delta->rows(), prev_delta->cols(), prev_d_output->data());

	CHECK(cudaDeviceSynchronize());
	BLAS_CH(cublasDestroy(handle));

	return;
}

Sigma_Matrix* Convolution2D_Layer::calc_update_param(Sigma_Matrix* prev_output){
	//�t�B���^�̍X�V�����v�Z����
	float alpha = optimizer_.learning_rate() * (1.0f / (float)delta_.batch_size());
	//�ŏ��̓f���^��0�ɂ���
	const float beta = 0;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));
	//�o�b�`�T�C�Y�񂾂��v�Z���s��(�ŏ���beta0 ���̌��beta = 1)(update_filter)
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
	//�o�C�A�X�̌v�Z
	//�o�b�`�T�C�Y�񂾂��v�Z���s��(�ŏ���beta0 ���̌��beta = 1)(update_filter)
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
	//cublas��p���Ċe�v�f������(�d�݂̍X�V)
	float alpha = 1.0f;
	float beta = -1.0f;
	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	//�d�݂̍X�V
	BLAS_CH(cublasSgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		filter_.rows(), filter_.cols(),
		&alpha,
		filter_.data(), filter_.rows(),
		&beta,
		update_filter_.data(), update_filter_.rows(),
		filter_.data(), filter_.rows()));

	//�o�C�A�X�̍X�V
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
