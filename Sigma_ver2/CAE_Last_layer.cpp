/*
channel–ˆ‚ÉAutoencoder‚ð‚©‚¯‚½ŽÀŒ±‚Ì‘æ‚P‘w‚Å‚·
*/

#include"for_debug.cuh"
#include "Imaging.cuh"

#define TRAINING_NUM 5000

int main5(){

	string load_path = "E:\\Dataset\\CAE_RESULT\\CAE_third_feature\\middle_output\\";
	vector<Sigma_Matrix> training_set;
	for (int i = 0; i < 600; i++){
		string file_name = load_path + to_string(i);
		Sigma_Matrix temp;
		temp.load(file_name);
		//temp /= 10.0f;
		training_set.push_back(temp);
	}
	Input_Layer input;
	Fully_Connect_Layer middle(2, iden, 0, 0.01f), output(100, iden, 0, 0.01f);

	Feed_Forward_NN model(input);
	model.add_layer(middle);
	model.add_layer(output);

	Optimizer opt;
	opt.learning_rate_ = 0.1f;
	model.set_optimizer(opt);

	for (int j = 0; j < TRAINING_NUM; j++){
		int idx = rand() % 600;
		cout << model.learn(training_set[idx], training_set[idx]) << endl;
	}
	cout << "change the learnin lrate" << endl;
	opt.learning_rate_ = 0.01f;
	model.set_optimizer(opt);
	for (int j = 0; j < TRAINING_NUM; j++){
		int idx = rand() % 600;
		cout << model.learn(training_set[idx], training_set[idx]) << endl;
	}

	string output_dir_path = "E:\\Dataset\\CAE_RESULT\\CAE_Last_feature\\";
	Sigma_Matrix result(2,1,1,1,10000,ZERO);
	size_t size = 2 * 100 * sizeof(float);
	for (int j = 0; j < 600; j++){
		string output_file_path = output_dir_path + to_string(j);
		model.infer(training_set[j]);
		CHECK(cudaMemcpy(result.mini_batch_data(j * 100),middle.output().data(),size,cudaMemcpyDeviceToDevice));
		//middle.output().save(output_file_path);
		//middle.output().save(output_file_path,false);
	}
	result.save("E:\\Dataset\\CAE_RESULT\\CAE_Last_feature\\all_data",false);

	return 0;
}