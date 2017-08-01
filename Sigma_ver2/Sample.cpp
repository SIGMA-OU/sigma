/*
MNISTを畳み込みニューラルネットワークを用いて学習するデモです。
*/

#include"for_debug.cuh"
#include "Imaging.cuh"

int mainMNIST(){

	//データの読み込み
	cout << "Load Data set" << endl;
	vector<Sigma_Matrix> training_data;
	vector<Sigma_Matrix> training_label_data;
	Sigma_Matrix test_data;
	Sigma_Matrix test_label_data;
	int training_minibatch_num = 600;
	for (int i = 0; i < 600; i++){
		Sigma_Matrix training_temp,label_temp;
		string input_data_path = "DATA_SET\\MNIST\\TRAINING_MINIBATCH_DATA\\MINIBATCH" + to_string(i);
		string label_data_path = "DATA_SET\\MNIST\\TRAINING_MINIBATCH_DATA\\LABEL_MINIBATCH" + to_string(i);
		training_temp.load(input_data_path);
		label_temp.load(label_data_path);
		training_data.push_back(training_temp);
		training_label_data.push_back(label_temp);
	}
	test_data.load("DATA_SET\\MNIST\\TEST_BATCH_DATA\\MINIBATCH");
	test_label_data.load("DATA_SET\\MNIST\\TEST_BATCH_DATA\\LABEL_MINIBATCH");
	vector<int> test_label = argmax_idx(test_label_data);
	test_data /= 255.0;
	cout << "Finish Loading" << endl;

	Input_Layer input;
	Convolution2D_Layer conv1(5, 5, 32, 1, true, relu, 0, 0.05f), conv2(7, 7, 64, 1, true, relu, 0, 0.05f);
	Pooling2D_Layer pool1(2, 2, 2, false, MAX_POOLING), pool2(2, 2, 2, false, MAX_POOLING);
	Fully_Connect_Layer full1(300, relu), full2(10, softmax);

	Feed_Forward_NN model(input);
	model.add_layer(conv1);
	model.add_layer(pool1);
	model.add_layer(conv2);
	model.add_layer(pool2);
	model.add_layer(full1);
	model.add_layer(full2);

	Optimizer opt;
	opt.learning_rate_ = 0.1;

	model.set_optimizer(opt);

	for (int i = 0; i < 100; i++){
		for (int j = 0; j < 600; j++){
			float val = model.learn(training_data[j], training_label_data[j]);
			cout << val << endl;
		}

		int acc = 0;

		for (int j = 0; j < 100; j++){
			Sigma_Matrix temp;
			next_batch(100,j,test_data,temp);
			vector<int> infer_label = argmax_idx(model.infer(temp));
			next_batch(100,j,test_label_data,temp);
			vector<int> temp_label = argmax_idx(temp);
			acc += equall(infer_label,temp_label);
		}
		cout << "acc = "<<(float)((float)acc / (float)10000.0f)<< endl;
	}

	return 0;
}