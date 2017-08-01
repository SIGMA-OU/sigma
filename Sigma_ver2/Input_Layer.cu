#include"Input_Layer.cuh"

Input_Layer::Input_Layer(){
	kind_ = INPUT_LAYER;
}

void Input_Layer::save(std::string file_path){
	return;
}

void Input_Layer::load(std::string file_path){
	return;
}

Sigma_Matrix* Input_Layer::initialize(Sigma_Matrix* input){
	cout << "Input_Layer initilize... || width :" << input->width() << ", height : " << input->height() << ", channel : " << input->channel() << ", batch_size : " << input->batch_size() << " ||" << endl;
	output_.initialize(input->width(), input->height(), input->channel(),input->depth(), input->batch_size(),ZERO);
	return &output_;
}

Sigma_Matrix* Input_Layer::change_batchsize(Sigma_Matrix* input){
	output_.initialize(input->width(), input->height(), input->channel(),input->depth() ,input->batch_size());
	return &output_;
}

Sigma_Matrix* Input_Layer::forward(Sigma_Matrix* input, NN_MODE mode){

	cudaMemcpy(output_.data(), input->data(),output_.size(), cudaMemcpyDeviceToDevice);
	for (int i = 0; i < Process.size(); i++){
		Process[i]->forward(&output_, mode);
	}
	return &output_;
}

void Input_Layer::add_process(Additional_Process& ap){
	Process.push_back(&ap);
	return;
}