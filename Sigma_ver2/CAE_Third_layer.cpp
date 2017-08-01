/*
channel–ˆ‚ÉAutoencoder‚ð‚©‚¯‚½ŽÀŒ±‚Ì‘æ‚P‘w‚Å‚·
*/

#include"for_debug.cuh"
#include "Imaging.cuh"

#define TRAINING_NUM 5000

int mainThird(){
	/*string load_path = "E:\\Dataset\\CAE_RESULT\\CAE_second_feature\\channel";
	vector<vector<Sigma_Matrix>> training_merge_set;

	for (int i = 0; i < 32; i++){
		string load_dir_path = load_path + to_string(i) + "\\";
		vector<Sigma_Matrix> training_set;
		for (int j = 0; j < 600; j++){
			string file_path = load_dir_path + to_string(j);
			Sigma_Matrix test;
			test.load(file_path);
			training_set.push_back(test);
		}
		training_merge_set.push_back(training_set);
	}

	vector<Sigma_Matrix> training_dataset;
	size_t size = 20 * sizeof(float);
	for (int i = 0; i < 600; i++){
		Sigma_Matrix temp(20,32,1,1,100,ZERO);
		for (int j = 0; j < 100; j++){
			for (int k = 0; k < 32; k++){
					CHECK(cudaMemcpy(temp.data()+(k*20 + j*640),training_merge_set[k][i].data() + 20*j,size,cudaMemcpyDeviceToDevice));
			}
		}
		training_dataset.push_back(temp);
	}

	for (int i = 0; i < training_dataset.size(); i++){
		training_dataset[i].print();
		string name = "E:\\Dataset\\CAE_RESULT\\CAE_third_feature\\merge_data\\" + to_string(i);
		training_dataset[i].save(name);
	}*/

	
	string load_path = "E:\\Dataset\\CAE_RESULT\\CAE_third_feature\\merge_data\\";
	vector<Sigma_Matrix> training_set;
	for (int i = 0; i < 600; i++){
		string file_name = load_path + to_string(i);
		Sigma_Matrix temp;
		temp.load(file_name);
		temp /= 10.0f;
		training_set.push_back(temp);
	}
	Input_Layer input;
	Fully_Connect_Layer middle(100, iden, 0, 0.01f), output(640, iden, 0, 0.01f);

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
	
	string output_dir_path = "E:\\Dataset\\CAE_RESULT\\CAE_third_feature\\middle_output\\";
	for (int j = 0; j < 600; j++){
		string output_file_path = output_dir_path + to_string(j);
		model.infer(training_set[j]);
		middle.output().save(output_file_path);
	}
	return 0;
}