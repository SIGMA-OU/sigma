/*
channel–ˆ‚ÉAutoencoder‚ð‚©‚¯‚½ŽÀŒ±‚Ì‘æ‚P‘w‚Å‚·
*/

#include"for_debug.cuh"
#include "Imaging.cuh"

#define TRAINING_NUM 5000

int mainSecond(){
	string load_path = "E:\\Dataset\\CAE_RESULT\\CAE_first_feature\\channel";
	for (int i = 0; i < 32; i++){
		string load_dir_path = load_path + to_string(i) + "\\";
		vector<Sigma_Matrix> training_set;
		for (int j = 0; j < 600; j++){
			string file_path = load_dir_path + to_string(j);
			Sigma_Matrix test;
			test.load(file_path);
			training_set.push_back(test);
		}

		Input_Layer input;
		Fully_Connect_Layer middle(20, iden), output(100, iden);

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

		string output_dir_path = "E:\\Dataset\\CAE_RESULT\\CAE_second_feature\\channel" + to_string(i) + "\\";
		for (int j = 0; j < 600; j++){
			string output_file_path = output_dir_path + to_string(j);
			model.infer(training_set[j]);
			middle.output().save(output_file_path);
		}
	}
	return 0;
}