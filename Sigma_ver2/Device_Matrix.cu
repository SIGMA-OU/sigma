#include"Device_Matrix.cuh"

Device_Matrix::Device_Matrix(){
	width_ = 1;
	height_ = 1;
	channel_ = 1;
	depth_ = 1;
	rows_ = 1;
	cols_ = 1;
	size_ = sizeof(float);
	CHECK(cudaMalloc((void**)&data_,size_));
}

Device_Matrix::Device_Matrix(const Device_Matrix& obj){
	rows_ = obj.rows_;
	cols_ = obj.cols_;
	size_ = obj.size_;
	CHECK(cudaMalloc((void**)&data_, size_));
	CHECK(cudaMemcpy(data_, obj.data_, size_, cudaMemcpyDeviceToDevice));	
	width_ = obj.width_;
	height_ = obj.cols_;
	channel_ = obj.channel_;
	depth_ = obj.channel_;
}

Device_Matrix::Device_Matrix(unsigned int rows, unsigned int cols, INITIALIZER init, float a, float b){
	rows_ = rows;
	cols_ = cols;
	size_ = rows * cols * sizeof(float);
	CHECK(cudaMalloc((void**)&data_, size_));

	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((cols_ + block.x - 1) / block.x, (rows_ + block.y - 1) / block.y);

	switch (init){
	case ZERO:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	case CONSTANT:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, a);
		break;
	case IDENTITY:
		set_array_identity_2D << <grid, block >> >(data_, rows_, cols_);
		break;
	case RANDOM:{
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_uniform2D << <grid, block >> >(data_, rows_, cols_, 0, 1.0f, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case GAUSSIAN:{
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_normal2D << <grid, block >> >(data_, rows_, cols_, a, b, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case UNIF_DISTRIB:
	{
		if (a >= b){
			cout << "DEVICE_MATRIX ERROR : UNIF_DISTRIB: b should be larger than a" << endl;
		}
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_uniform2D << <grid, block >> >(data_, rows_, cols_, a, b, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case STEP:
		set_array_step_2D << < grid, block >> > (data_, rows_, cols_);
		break;
	case PLANE:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	default:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 11);
		break;
	}
	CHECK(cudaDeviceSynchronize());

	width_ = cols;
	height_ = rows;
	channel_ = 1;
	depth_ = 1;
}

Device_Matrix::Device_Matrix(unsigned int height, unsigned int width, unsigned int channel, unsigned int depth, INITIALIZER init, float a, float b){
	width_ = width;
	height_ = height;
	channel_ = channel;
	depth_ = depth;
	rows_ = height;
	cols_ = width * channel * depth;
	size_ = rows_ * cols_ * sizeof(float);
	CHECK(cudaMalloc((void**)&data_, size_));
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((cols_ + block.x - 1) / block.x, (rows_ + block.y - 1) / block.y);

	switch (init){
	case ZERO:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	case CONSTANT:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, a);
		break;
	case IDENTITY:
		set_array_identity_2D << <grid, block >> >(data_, rows_, cols_);
		break;
	case RANDOM:{
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_uniform2D << <grid, block >> >(data_, rows_, cols_, 0, 1.0f, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case GAUSSIAN:{
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_normal2D << <grid, block >> >(data_, rows_, cols_, a, b, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case UNIF_DISTRIB:
	{
		if (a >= b){
			cout << "DEVICE_MATRIX ERROR : UNIF_DISTRIB: b should be larger than a" << endl;
		}
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_uniform2D << <grid, block >> >(data_, rows_, cols_, a, b, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case STEP:
		set_array_step_2D << < grid, block >> > (data_, rows_, cols_);
		break;
	case PLANE:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	default:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	}
	CHECK(cudaDeviceSynchronize());
}

Device_Matrix::Device_Matrix(unsigned int rows, unsigned int cols, float* host_data){
	rows_ = rows;
	cols_ = cols;
	size_ = cols * rows * sizeof(float);
	CHECK(cudaMalloc((void**)&data_, size_));
	CHECK(cudaMemcpy(data_, host_data, size_, cudaMemcpyHostToDevice));
	width_ = cols;
	height_ = rows;
	channel_ = 1;
	depth_ = 1;
}

void Device_Matrix::initialize(Device_Matrix& obj){
	if (data_ != NULL){
		CHECK(cudaFree(data_));
	}
	rows_ = obj.rows_;
	cols_ = obj.cols_;
	size_ = obj.size_;
	CHECK(cudaMalloc((void**)&data_, size_));
	CHECK(cudaMemcpy(data_, obj.data_, size_, cudaMemcpyDeviceToDevice));
	width_ = obj.width_;
	height_ = obj.cols_;
	channel_ = obj.channel_;
	depth_ = obj.channel_;
}

void Device_Matrix::initialize(unsigned int rows, unsigned int cols, INITIALIZER init, float a, float b){
	if (data_ != NULL){
		CHECK(cudaFree(data_));
	}
	rows_ = rows;
	cols_ = cols;
	size_ = rows * cols * sizeof(float);
	CHECK(cudaMalloc((void**)&data_, size_));

	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((cols_ + block.x - 1) / block.x, (rows_ + block.y - 1) / block.y);
	
	switch (init){
	case ZERO:
		set_array_2D <<<grid,block >>>(data_,rows_,cols_,0);
		break;
	case CONSTANT:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, a);
		break;
	case IDENTITY:
		set_array_identity_2D << <grid, block >> >(data_, rows_, cols_);
		break;
	case RANDOM:{
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state,sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(),rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_uniform2D << <grid, block >> >(data_, rows_, cols_, 0, 1.0f, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case GAUSSIAN:{
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_normal2D << <grid, block >> >(data_, rows_, cols_, a, b, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case UNIF_DISTRIB:
	{
		if (a >= b){
			cout << "DEVICE_MATRIX ERROR : UNIF_DISTRIB: b should be larger than a" << endl;
		}
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_uniform2D << <grid, block >> >(data_, rows_, cols_, a, b, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case STEP:
		set_array_step_2D << < grid, block >> > (data_, rows_, cols_);
		break;
	case PLANE:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	default:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	}
	CHECK(cudaDeviceSynchronize());

	width_ = cols;
	height_ = rows;
	channel_ = 1;
	depth_ = 1;
}

void Device_Matrix::initialize(unsigned int height, unsigned int width, unsigned int channel, unsigned int depth, INITIALIZER init, float a, float b){
	if (data_ != NULL){
		CHECK(cudaFree(data_));
	}
	width_ = width;
	height_ = height;
	channel_ = channel;
	depth_ = depth;
	rows_ = height;
	cols_ = width * channel * depth;
	size_ = rows_ * cols_ * sizeof(float);

	CHECK(cudaMalloc((void**)&data_, size_));

	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((cols_ + block.x - 1) / block.x, (rows_ + block.y - 1) / block.y);

	switch (init){
	case ZERO:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	case CONSTANT:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, a);
		break;
	case IDENTITY:
		set_array_identity_2D << <grid, block >> >(data_, rows_, cols_);
		break;
	case RANDOM:{
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_uniform2D << <grid, block >> >(data_, rows_, cols_, 0, 1.0f, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case GAUSSIAN:{
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_normal2D << <grid, block >> >(data_, rows_, cols_, a, b, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case UNIF_DISTRIB:
	{
		if (a >= b){
			cout << "DEVICE_MATRIX ERROR : UNIF_DISTRIB: b should be larger than a" << endl;
		}
		std::random_device rd;
		curandState_t *state;
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ * cols_));
		rand_init2D << <grid, block >> > (rd(), rows_, cols_, state);
		CHECK(cudaDeviceSynchronize());
		initialize_uniform2D << <grid, block >> >(data_, rows_, cols_, a, b, state);
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case STEP:
		set_array_step_2D << < grid, block >> > (data_, rows_, cols_);
		break;
	case PLANE:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	default:
		set_array_2D << <grid, block >> >(data_, rows_, cols_, 0);
		break;
	}
	CHECK(cudaDeviceSynchronize());
}

void Device_Matrix::read_data(string file_name){

}

void Device_Matrix::write_data(string file_name){

}

////////////////////////////	operator	///////////////////

Device_Matrix& Device_Matrix::operator = (const Device_Matrix& obj){
	if (&obj == this) return *this;
	if (data_ != NULL){
		CHECK(cudaFree(data_));
	}
	rows_ = obj.rows_;
	cols_ = obj.cols_;
	size_ = obj.size_;
	CHECK(cudaMalloc((void**)&data_, size_));
	CHECK(cudaMemcpy(data_, obj.data_, size_, cudaMemcpyDeviceToDevice));
	width_ = obj.width_;
	height_ = obj.cols_;
	channel_ = obj.channel_;
	depth_ = obj.channel_;
	return *this;
}

Device_Matrix& Device_Matrix::operator * (const Device_Matrix& obj){
	return *this;
}

Device_Matrix& Device_Matrix::operator + (const Device_Matrix& obj){
	return *this;
}

Device_Matrix& Device_Matrix::operator - (const Device_Matrix& obj){
	return *this;
}

void Device_Matrix::print(bool All_status){
	if (data_ == NULL){
		cout << "This Matrix doesn't have data" << endl;
	}
	else{
		cout << "-------------   Device_Matrix   -------------" << endl;
		cout << "rows : " << rows_ << ", cols : " << cols_ << endl;
		cout << "width : " << width_ << ", height : " << height_ << ",channel : " << channel_ << ", depth : " << depth_ << endl;
		cout << "data_adress : " << data_ << endl;
		if (All_status){
			float* tmp_data;
			tmp_data = (float*)malloc(size_);
			cudaMemcpy(tmp_data,data_,size_,cudaMemcpyDeviceToHost);
			//列優先のデータの表示
			for (int i = 0; i < rows_; i++){
				for (int j = 0; j < cols_; j++){
					unsigned int idx = j * rows_ + i;
					cout <<setw(5)<< tmp_data[idx] << ",";
				}
				cout << endl;
			}
			free(tmp_data);
		}
		cout << "--------------------------------------------" << endl;
	}
}