#include"for_debug.cuh"
#include"Sigma_Matrix.cuh"
#include"Input_Layer.cuh"
#include"Activate_Function.cuh"
#include"imaging.cuh"
#include"Dropout_Process.cuh"
#include"Noising_Process.cuh"
#include"Fully_Connect_Layer.cuh"
#include"Convolution2D_Layer.cuh"
#include"Pooling2D_Layer.cuh"
#include"Feed_Forward_NN.cuh"
#include<vector>
#include<sstream>

int main_pra(){

	//教師データを用意する
	float teacher_data[50] = { 0 };
	for (int i = 25; i < 50; i++){
		teacher_data[i] = 1.0;
	}
	Sigma_Matrix teacher(1,1,1,1,50,ZERO);
	cudaMemcpy(teacher.data(),teacher_data,50 *sizeof(float),cudaMemcpyHostToDevice);

	string control_file_path = "F:\\Tanabe\\DAT_DATA_Blank\\control_1";
	Sigma_Matrix control;
	control.load(control_file_path);
	control /= 128.0;

	ofstream ofs_all_delta("CNN_1800/result1/all_delta.csv", ios::app |ios::out);
	ofstream ofs_data_num("CNN_1800/result1/data_num.csv", ios::app | ios::out);

	for (int i = 0; i < 1800; i++){
		if (i == 1192){
			continue;
		}
		else if (i == 1215){
			continue;
		}
		string output_file = "F:\\Tanabe\\DAT_DATA\\" + to_string(i);
		string ofs_name = "CNN_1800/result1/delta_" + to_string(i) + ".csv";
		ofstream ofs_delta(ofs_name);
		Sigma_Matrix cmpd;
		cmpd.load(output_file);
		ofs_data_num <<i<<","<< cmpd.batch_size() << endl;
		cmpd /= 128.0;

		//method 1
		/*
		const int minibatch_size = 50;
		Sigma_Matrix mini_batch(150,150,4,1,minibatch_size,ZERO);

		for (int j = 0; j < minibatch_size / 2; j++){
			int idx = rand() % control.batch_size();
			size_t size = control.size() / control.batch_size();
			cudaMemcpy(mini_batch.mini_batch_data(j), control.mini_batch_data(idx),size,cudaMemcpyDeviceToDevice);
		}
		for (int j = minibatch_size / 2; j < minibatch_size; j++){
			int idx = rand() % cmpd.batch_size();
			size_t size = cmpd.size() / cmpd.batch_size();
			cudaMemcpy(mini_batch.mini_batch_data(j), cmpd.mini_batch_data(idx), size, cudaMemcpyDeviceToDevice);
		}*/

		//method 2
		vector<Sigma_Matrix> input_data;
		const int minibatch_num = 100;
		const int minibatch_size = 50;

		for (int j = 0; j < minibatch_num; j++){
			Sigma_Matrix temp_mini_batch(150, 150, 4, 1, minibatch_size, ZERO);
			for (int k = 0; k < minibatch_size / 2; k++){
				int idx = rand() % control.batch_size();
				size_t size = control.size() / control.batch_size();
				cudaMemcpy(temp_mini_batch.mini_batch_data(k), control.mini_batch_data(idx), size, cudaMemcpyDeviceToDevice);
			}
			for (int k = minibatch_size / 2; k < minibatch_size; k++){
				int idx = rand() % cmpd.batch_size();
				size_t size = cmpd.size() / cmpd.batch_size();
				cudaMemcpy(temp_mini_batch.mini_batch_data(k), cmpd.mini_batch_data(idx), size, cudaMemcpyDeviceToDevice);
			}
			input_data.push_back(temp_mini_batch);
		}

		//構造を作る
		Input_Layer input;
		Convolution2D_Layer conv1(5, 5, 32, 1, true, relu, 0, 0.1), conv2(5, 5, 64, 1, true, relu, 0, 0.1);
		Pooling2D_Layer pool1(2, 2, 2, true, MAX_POOLING), pool2(2, 2, 2, true, MAX_POOLING);
		Fully_Connect_Layer fc1(1024, relu,0,0.1),output(1, sig,0,0.1);

		Feed_Forward_NN model(input);
		model.add_layer(conv1);
		model.add_layer(pool1);
		model.add_layer(conv2);
		model.add_layer(pool2);
		model.add_layer(fc1);
		model.add_layer(output);

		Optimizer opt;
		opt.learning_rate_ = 0.1f;
		model.set_optimizer(opt);
		
		ofs_all_delta << i;
		for (int j = 0; j < 1000; j++){
			int idx = rand() % input_data.size();
			float delta = model.learn(input_data[idx],teacher);
			cout << i << " ," << j <<":"<<delta << endl;
			ofs_all_delta << delta << ",";
			ofs_delta << delta << ",";
		}
		ofs_all_delta << endl;
		ofs_delta.close();
		/*Matrixを画像化する
		vector<Mat> images = Matrix2Image(test[0]);
		vector<vector<Mat>> images2 = Matrix2Image(test);

		for (int j = 0; j < images.size(); j++){
			imshow("test",images[j]);
			waitKey();
		}*/

	}
	
	ofs_all_delta.close();
	ofs_data_num.close();

	return 0;
}