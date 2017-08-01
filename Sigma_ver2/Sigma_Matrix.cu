#include"Sigma_Matrix.cuh"

Sigma_Matrix::Sigma_Matrix(){
	width_ = height_ = channel_ = depth_ = batch_size_ = rows_ = cols_ = 0;
	data_ = NULL;
	size_ = 0;
	d_mx.initialize(1,1,ZERO);
}

Sigma_Matrix::Sigma_Matrix(const Sigma_Matrix &obj){
	width_ = obj.width_;
	height_ = obj.height_;
	channel_ = obj.channel_;
	depth_ = obj.depth_;

	batch_size_ = obj.batch_size_;
	rows_ = obj.rows_;
	cols_ = obj.cols_;
	size_ = obj.size_;
	
	CHECK(cudaMalloc((void**)&data_,size_));
	CHECK(cudaMemcpy(data_,obj.data_,size_,cudaMemcpyDeviceToDevice));
}

Sigma_Matrix::Sigma_Matrix(unsigned int rows, unsigned int cols, INITIALIZER init, float a, float b){
	height_ = rows;
	width_ = 1;
	channel_ = 1;
	depth_ = 1;
	batch_size_ = cols;
	rows_ = rows;
	cols_ = cols;
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

		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_uniform2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, 0, 1.0f, state);

		}
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case GAUSSIAN:{
		std::random_device rd;
		curandState_t *state;

		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_normal2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, a, b , state);
		}
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
		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_uniform2D << <grid, block >> >(&data_[rows_ * i], 1, rows_, a, b, state);
		}
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

Sigma_Matrix::Sigma_Matrix(unsigned height, unsigned int width, unsigned int channel, unsigned int depth, unsigned int batch_size , INITIALIZER init, float a, float b){
	height_ = height;
	width_ = width;
	channel_ = channel;
	depth_ = depth;
	batch_size_ = batch_size;
	rows_ = height * width * channel * depth;
	cols_ = batch_size;
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

		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_uniform2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, 0, 1.0f, state);

		}
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case GAUSSIAN:{
		std::random_device rd;
		curandState_t *state;

		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_normal2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, a, b, state);
		}
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
		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_uniform2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, a, b, state);
		}
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

void Sigma_Matrix::initialize(unsigned height, unsigned int width, unsigned int channel, unsigned int depth, unsigned int batch_size, INITIALIZER init, float a, float b){
	if (data_ != NULL){
		CHECK(cudaFree(data_));
	}
	height_ = height;
	width_ = width;
	channel_ = channel;
	depth_ = depth;
	batch_size_ = batch_size;
	rows_ = height * width * channel * depth;
	cols_ = batch_size;
	size_ = rows_ * cols_ * sizeof(float);

	CHECK(cudaMalloc((void**)&data_,size_));

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

		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_uniform2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, 0, 1.0f, state);
		}
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case GAUSSIAN:{
		std::random_device rd;
		curandState_t *state;

		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_ ));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_normal2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, a, b, state);
		}
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
		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_uniform2D << <grid, block >> >(&data_[rows_ * i],1, rows_, a, b, state);
		}
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

void Sigma_Matrix::initialize(unsigned int rows, unsigned int cols, INITIALIZER init, float a, float b){

	if (data_ != NULL){
		CHECK(cudaFree(data_));
	}
	height_ = rows;
	width_ = 1;
	channel_ = 1;
	depth_ = 1;
	batch_size_ = cols;
	rows_ = rows;
	cols_ = cols;
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

		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_uniform2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, 0, 1.0f, state);

		}
		CHECK(cudaDeviceSynchronize());
		cudaFree(state);
		break;
	}
	case GAUSSIAN:{
		std::random_device rd;
		curandState_t *state;

		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_normal2D << <grid, block >> >(&data_[rows_ * i], rows_, 1, a, b, state);
		}
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
		/*ミニバッチごとに初期化を行う(メモリ節約のため)*/
		CHECK(cudaMalloc((curandState_t **)&state, sizeof(curandState_t) * rows_));
		for (int i = 0; i < batch_size_; i++){
			rand_init2D << <grid, block >> > (rd(), rows_, 1, state);
			initialize_uniform2D << <grid, block >> >(&data_[rows_ * i], 1, rows_, a, b, state);
		}
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

void Sigma_Matrix::add_Matrix(Device_Matrix &Matrix){
	//最初のデータ基準に初期化する
	if (batch_size_ == 0){
		width_ = Matrix.width();
		height_ = Matrix.height();
		channel_ = Matrix.channel();
		depth_ = Matrix.depth();
		rows_ = width_ * height_ * channel_ * depth_;
	}
	else{
		if (width_ != Matrix.width() || height_ != Matrix.height() || channel_ != Matrix.channel() || depth_ != Matrix.depth()){
			cout << "Error : Sigma_Matrix add_Matrix : Size is wrong" << endl;
			return;
		}
	}
	//データを全てつなげる処理を行う
	float* tmp;
	CHECK(cudaMalloc((void**)&tmp, size_ + Matrix.size()));
	
	//すでに持っているデータを転送する
	if (size_ > 0){
		CHECK(cudaMemcpy(tmp,data_,size_,cudaMemcpyDeviceToDevice));
		/*dim3 block(1024);
		dim3 grid((size_ / sizeof(float) + block.x - 1) / block.x);
		block1D << <grid, block >> >(tmp, 0, data_, size_ / sizeof(float));*/
	}
	CHECK(cudaDeviceSynchronize());
	dim3 block(1024);
	dim3 grid((Matrix.rows() * Matrix.cols() + block.x - 1) / block.x);
	block1D << <grid, block >> >(tmp, size_ / sizeof(float), Matrix.data(), Matrix.size() / sizeof(float));
	CHECK(cudaDeviceSynchronize());

	//後処理する
	CHECK(cudaFree(data_));
	data_ = tmp;
	size_ += Matrix.size();
	batch_size_++;
	cols_++;
}

Device_Matrix& Sigma_Matrix::get_Matrix(unsigned int idx){
	if (idx >= batch_size_){
		cout << "Error : Sigma_Matrix :get_Matrix : idx is over batch size" << endl;
	}
	else{
		d_mx.initialize(height_,width_,channel_,depth_,ZERO);

		CHECK(cudaMemcpy(d_mx.data(), &data_[idx * d_mx.rows() * d_mx.cols()], d_mx.size(), cudaMemcpyDeviceToDevice));
		return d_mx;
	}
}

void Sigma_Matrix::print(bool All_status){
	if (data_ == NULL){
		cout << "This Matrix doesn't have data" << endl;
	}
	else{
		cout << "-------------   Sigma_Matrix   -------------" << endl;
		cout << "rows : " << rows_ << ", cols : " << cols_ << endl;
		cout << "width : " << width_ << ", height : " << height_ << ",channel : " << channel_ << ", depth : " << depth_ << ", batch_size" << batch_size_ << endl;
		cout << "data_adress : " << data_ << endl;
		cout << "data size" << size_ << endl;
		if (All_status){
			float* tmp_data;
			tmp_data = (float*)malloc(size_);
			CHECK(cudaMemcpy(tmp_data, data_, size_, cudaMemcpyDeviceToHost));
			//列優先のデータの表示
			for (int i = 0; i < rows_; i++){
				for (int j = 0; j < cols_; j++){
					unsigned int idx = j * rows_ + i;
					cout <<setw(10)<< tmp_data[idx] << ",";
				}
				cout << endl;
			}
			free(tmp_data);
		}
		cout << "--------------------------------------------" << endl;
	}
}

void Sigma_Matrix::set_value(unsigned int row, unsigned int col, float value){
	if (row >= rows_ || col >= cols_){
		cout << "error : Sigma_Matrix : set valu : index is over " << endl;
		return;
	}
	CHECK(cudaMemcpy(&data_[col * rows_ + row],&value,sizeof(float),cudaMemcpyHostToDevice));
	return;
}

void Sigma_Matrix::set_matrix(float* data){
	CHECK(cudaMemcpy(data_,data,size_,cudaMemcpyHostToDevice));
}

void Sigma_Matrix::set_matrix(float* data, unsigned int start_idx, unsigned int data_num){
	CHECK(cudaMemcpy(&data_[start_idx * rows_], data, rows_ * data_num * sizeof(float), cudaMemcpyHostToDevice));
}

Sigma_Matrix& Sigma_Matrix::operator = (const Sigma_Matrix& obj){
	if (&obj == this) return *this;
	if (data_ != NULL){
		CHECK(cudaFree(data_));
	}
	width_ = obj.width_;
	height_ = obj.height_;
	channel_ = obj.channel_;
	depth_ = obj.channel_;
	batch_size_ = obj.batch_size_;
	rows_ = obj.rows_;
	cols_ = obj.cols_;
	size_ = obj.size_;

	CHECK(cudaMalloc((void**)&data_, size_));
	CHECK(cudaMemcpy(data_, obj.data_, size_, cudaMemcpyDeviceToDevice));

	return *this;
}

Sigma_Matrix& Sigma_Matrix::operator /= (float x){
	if (x == 0){
		cout << "error : 0 cannot use " << endl;
		return *this;
	}
	const float alpha = 1.0 / x;
	const float beta = 0;

	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	BLAS_CH(cublasSgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		rows_,cols_,
		&alpha,
		data_,rows_,
		&beta,
		data_,rows_,
		data_,rows_));

	return *this;
}

Sigma_Matrix& Sigma_Matrix::operator *= (float x){
	if (x == 0){
		cout << "error : 0 cannot use " << endl;
		return *this;
	}
	const float alpha = x;
	const float beta = 0;

	cublasHandle_t handle;
	BLAS_CH(cublasCreate(&handle));

	BLAS_CH(cublasSgeam(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		rows_, cols_,
		&alpha,
		data_, rows_,
		&beta,
		data_, rows_,
		data_, rows_));

	return *this;
}

Device_Matrix& Sigma_Matrix::operator [](unsigned int idx){
	return get_Matrix(idx);
}

void Sigma_Matrix::save(string file_path ,bool binary_output){
	
	if (file_path.size() == 0){
		file_path = "temp";
	}

	if (binary_output){
		string file_name = file_path + ".dat";
		ofstream ofs(file_name, ios::out|ios::binary|ios::app);

		float* tmp_data;
		tmp_data = (float*)malloc(size_);
		CHECK(cudaMemcpy(tmp_data, data_, size_, cudaMemcpyDeviceToHost));
		//先頭の7 * 4byteは、int型で、rows,cols,height,widht,channel,depth,batch_sizeの順に入れられる。
		ofs.write((char*)(&rows_),sizeof(unsigned int));
		ofs.write((char*)(&cols_), sizeof(unsigned int));
		ofs.write((char*)(&height_), sizeof(unsigned int));
		ofs.write((char*)(&width_), sizeof(unsigned int));
		ofs.write((char*)(&channel_), sizeof(unsigned int));
		ofs.write((char*)(&depth_), sizeof(unsigned int));
		ofs.write((char*)(&batch_size_), sizeof(unsigned int));
		int num = rows_ * cols_;
		ofs.write((char*)tmp_data, sizeof(float) * num);
		free(tmp_data);
		ofs.close();
	}
	else{
		string file_name = file_path + ".csv";
		ofstream ofs(file_name,ios::app|ios::out);
		ofs << "rows_," << rows_ << endl;
		ofs << "cols_," << cols_ << endl;
		ofs << "height_," << height_ << endl;
		ofs << "width_," << width_ << endl;
		ofs << "channel_," << channel_ << endl;
		ofs << "depth_," << depth_ << endl;
		ofs << "batch_size_," << batch_size_ << endl;

		float* tmp_data;
		tmp_data = (float*)malloc(size_);
		CHECK(cudaMemcpy(tmp_data, data_, size_, cudaMemcpyDeviceToHost));
		for (int i = 0; i < cols_; i++){
			for (int j = 0; j < rows_; j++){
				unsigned int idx = i * rows_ + j;
				ofs << tmp_data[idx] << ",";
			}
			ofs << endl;
		}
		free(tmp_data);
		ofs.close();
	}
}

void Sigma_Matrix::load(string file_path){

	file_path += ".dat";
	ifstream ifs(file_path,ios::binary);
	if (ifs.fail()){
		cout << "Sigma Matrix : file error :" << file_path << " :It doesnt exist" << endl;
		return;
	}
	unsigned int status[7];
	ifs.read((char*)status, sizeof(unsigned int)*7);
	rows_ = status[0];
	cols_ = status[1];
	height_ = status[2];
	width_ = status[3];
	channel_ = status[4];
	depth_ = status[5];
	batch_size_ = status[6];
	size_ = rows_ * cols_ * sizeof(float);
	float* tmp_data = (float*)malloc(size_);
	ifs.read((char*)tmp_data,size_);
	if (data_ != NULL){
		CHECK(cudaFree(data_));
	}
	CHECK(cudaMalloc((void**)&data_, size_));
	CHECK(cudaMemcpy(data_,tmp_data,size_,cudaMemcpyHostToDevice));
	free(tmp_data);
	ifs.close();
}