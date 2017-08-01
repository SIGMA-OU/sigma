#include"Pooling2D_Layer.cuh"

Pooling2D_Layer::Pooling2D_Layer(){
	kind_ = POOLING_LAYER;
	param_.height = param_.width = param_.stride = 1;
	param_.padding = false;
	param_.method = MAX_POOLING;
	delta_.initialize(0,0,ZERO);
	output_.initialize(0, 0, ZERO);
	d_output_.initialize(0, 0, ZERO);
	d_idx_matrix = NULL;
}

Pooling2D_Layer::Pooling2D_Layer(Pooling2D_Layer::Param& param){
	kind_ = POOLING_LAYER;
	param_.height = param.height;
	param_.width = param.width;
	param_.stride = param.stride;
	param_.method = param.method;
	param_.padding = param.padding;
	delta_.initialize(0, 0, ZERO);
	output_.initialize(0, 0, ZERO);
	d_output_.initialize(0, 0, ZERO);
	d_idx_matrix = NULL;
}

Pooling2D_Layer::Pooling2D_Layer(unsigned int width, unsigned int height, unsigned int stride, bool padding, POOLING_METHOD method){
	kind_ = POOLING_LAYER;
	param_.height = height;
	param_.width = width;
	param_.stride = stride;
	param_.method = method;
	param_.padding = padding;
	delta_.initialize(0, 0, ZERO);
	output_.initialize(0, 0, ZERO);
	d_output_.initialize(0, 0, ZERO);
	d_idx_matrix = NULL;
}

Sigma_Matrix* Pooling2D_Layer::forward(Sigma_Matrix* input, NN_MODE mode){

	dim3 block(BLOCK_X,BLOCK_Y);
	dim3 grid((output_.width() * output_.channel() + block.x - 1) / block.x,(output_.height() + block.y - 1) / block.y);
	if (param_.method == MAX_POOLING){
		int data_num = output_.channel() * output_.width() * output_.height() * output_.depth();
		for (int i = 0; i < input->batch_size(); i++){
			int* ptr_idx = d_idx_matrix + (i * data_num);
			max_Pooling2D << <grid, block >> >(output_.mini_batch_data(i), input->mini_batch_data(i), ptr_idx,
												input->width(),input->height(),input->channel(),
												output_.width(),output_.height(),
												param_.width,param_.height,param_.stride,
												param_.padding,output_.height(),output_.width()*output_.channel());
		}
	}
	else{
		for (int i = 0; i < input->batch_size(); i++){
			average_Pooling2D << <grid, block >> >(output_.mini_batch_data(i), input->mini_batch_data(i),
				input->width(), input->height(), input->channel(),
				output_.width(), output_.height(),
				param_.width, param_.height, param_.stride,
				param_.padding, output_.height(), output_.width()*output_.channel());
		}
	}
	CHECK(cudaDeviceSynchronize());

	//Additional process
	for (int i = 0; i < Process.size(); i++){
		Process[i]->forward(&output_, mode);
	}

	return &output_;
}

Sigma_Matrix* Pooling2D_Layer::initialize(Sigma_Matrix* prev_output){
	cout << "Pooling2D initialize... || batch_size : " << prev_output->batch_size() << endl;
	
	//paddingの有無によって、outputのサイズが変化する。
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
	//出力のチャネル数は、変化しない
	output_.initialize(output_height, output_width, prev_output->channel(), 1, prev_output->batch_size(), ZERO);

	//output_の要素数だけd_idx_matrixの要素を用意する
	if (param_.method == MAX_POOLING){
		int nBytes = output_.width() * output_.height() * output_.channel() * output_.depth() * output_.batch_size() * sizeof(int);
		CHECK(cudaMalloc((void**)&d_idx_matrix, nBytes));
	}
	return &output_;
}

Sigma_Matrix* Pooling2D_Layer::change_batchsize(Sigma_Matrix* prev_output){
	//paddingの有無によって、outputのサイズが変化する。
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
	//出力のチャネル数は、変化しない
	output_.initialize(output_height, output_width, prev_output->channel(), 1, prev_output->batch_size(), ZERO);

	//デルタも変化する
	delta_.initialize(output_height, output_width, prev_output->channel(), 1, prev_output->batch_size(), ZERO);

	//微分出力も変化する
	d_output_.initialize(output_height, output_width, prev_output->channel(), 1, prev_output->batch_size(), CONSTANT,1);

	//idx_matrixのサイズを変更する
	if (param_.method == MAX_POOLING){
		CHECK(cudaFree(d_idx_matrix));
		int nBytes = output_.width() * output_.height() * output_.channel() * output_.depth() * output_.batch_size() * sizeof(int);
		CHECK(cudaMalloc((void**)&d_idx_matrix, nBytes));
	}

	return &output_;
}

Sigma_Matrix* Pooling2D_Layer::get_d_output(){
	return &d_output_;
}

Sigma_Matrix* Pooling2D_Layer::initialize_optimizer(Sigma_Matrix* prev_output){
	delta_.initialize(output_.height(),output_.width(),output_.channel(),output_.depth(),output_.batch_size(),ZERO);
	d_output_.initialize(output_.height(), output_.width(), output_.channel(), output_.depth(), output_.batch_size(), CONSTANT,1);
	return &output_;
}

void Pooling2D_Layer::back_ward(Sigma_Matrix* prev_delta, Sigma_Matrix* prev_d_output){

	dim3 block(BLOCK_X,BLOCK_Y);
	dim3 grid((prev_delta->width()*prev_delta->channel() + block.x - 1) / block.x, (prev_delta->height() + block.y - 1) / block.y);

	unsigned int mb_prev_delta_cols = delta_.width()* delta_.height();

	if (param_.method == MAX_POOLING){
		int data_num = delta_.width() * delta_.height() * delta_.channel();
		int *ptr;
		for (int i = 0; i < prev_delta->batch_size(); i++){
			ptr = d_idx_matrix + data_num * i;
			backward_for_max_Pooling2D << <grid, block >> >(prev_delta->mini_batch_data(i),delta_.mini_batch_data(i),ptr,
				prev_delta->width(), prev_delta->height(), prev_delta->channel(),
				delta_.width(), delta_.height(),
				param_.width, param_.height, param_.stride,
				param_.padding, prev_delta->height(), mb_prev_delta_cols);
		}
	}
	else{
		for (int i = 0; i < prev_delta->batch_size(); i++){
			backward_for_average_Pooling2D << <grid, block >> >(prev_delta->mini_batch_data(i), delta_.mini_batch_data(i),
				prev_delta->width(),prev_delta->height(),prev_delta->channel(),
				delta_.width(),delta_.height(),
				param_.width,param_.height,param_.stride,
				param_.padding,prev_delta->height(),mb_prev_delta_cols);
		}
	}
	CHECK(cudaDeviceSynchronize());
	return;
}

void Pooling2D_Layer::add_process(Additional_Process& ap){
	Process.push_back(&ap);
	return;
}

void Pooling2D_Layer::set_param(Pooling2D_Layer::Param& param){
	param_.height = param.height;
	param_.width = param.width;
	param_.stride = param.stride;
	param_.method = param.method;
	param_.padding = param.padding;
	return;
}