/*
CAEのデモです。
*/

#include"for_debug.cuh"
#include "Imaging.cuh"

int maincae(){
	//データの読み込み
	cout << "Load Data set" << endl;
	vector<Sigma_Matrix> training_data;
	vector<Sigma_Matrix> training_label_data;
	Sigma_Matrix test_data;
	Sigma_Matrix test_label_data;
	int training_minibatch_num = 600;
	for (int i = 0; i < 600; i++){
		Sigma_Matrix training_temp, label_temp;
		string input_data_path = "E:\\Dataset\\CAE_RESULT\\CAE\\" + to_string(i);
		training_temp.load(input_data_path);
		training_data.push_back(training_temp);
	}
	cout << "load_data end" << endl;

	string path = "E:\\Dataset\\CAE_RESULT\\CAE_divide_channel\\channel";
	for (int b = 0; b < 600; b++){
		for (int i = 0; i < 32; i++){
			string dir_path = path + to_string(i) + "\\"+ to_string(b);
			Sigma_Matrix test(28, 28, 1, 1, 100, ZERO);
			for (int j = 0; j < 100; j++){
				CHECK(cudaMemcpy(test.mini_batch_data(j),training_data[b].mini_batch_data(j)+(784 * i),784*sizeof(float),cudaMemcpyDeviceToDevice));
			}
			test.save(dir_path);
		}
		cout << "batch_num" << b << endl;
	}


	return 0;
}