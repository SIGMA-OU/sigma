/*
channel–ˆ‚ÉAutoencoder‚ð‚©‚¯‚½ŽÀŒ±‚Ì‘æ‚P‘w‚Å‚·
*/

#include"for_debug.cuh"
#include"MNIST.hpp"
#include "Imaging.cuh"

#define TRAINING_NUM 20000

int main_test(){

	std::cout << "read_files" << endl;
	vector<vector<float>> training_dataset;
	vector<float> training_label_dataset;

	const int batchsize = 100;
	vector<Sigma_Matrix> training_data;
	vector<Sigma_Matrix> label_data;

	cout << "read_MNIST_data" << endl;
	Mnist mnist;
	training_dataset = mnist.readTrainingFile("c:\\Data_Set\\MNIST\\train-images.idx3-ubyte");
	training_label_dataset = mnist.readLabelFile("c:\\Data_Set\\MNIST\\train-labels.idx1-ubyte");
	cout << "read_label_data" << endl;

	Sigma_Matrix trainig(28, 28, 1, 1, 60000, ZERO);
	cout << "input_data" << endl;
	int idx = 0;
	for (int i = 0; i < 60000 / batchsize; i++){
		Sigma_Matrix temp(28, 28, 1, 1, batchsize, ZERO);
		Sigma_Matrix temp_label;

		for (int j = 0; j < batchsize; j++){
			Device_Matrix label(10, 1, ZERO);
			temp.set_matrix(training_dataset[idx].data(), j);
			label.set(1, (int)training_label_dataset[idx]);
			temp_label.add_Matrix(label);
			idx++;

		}

		temp /= 255.0;

		training_data.push_back(temp);
		label_data.push_back(temp_label);
		cout << i << endl;
	}
	cout << "data_set_end" << endl;


	for (int i = 0; i < training_data.size(); i++){
		string traing_dir = "c:\\Data_Set\\MNIST\\binary\\training_" + to_string(i);
		string label_dir = "c:\\Data_Set\\MNIST\\binary\\label_" + to_string(i);

		//Binary‚Å•Û‘¶
		training_data[i].save(traing_dir);
		label_data[i].save(label_dir);

		//csv‚Å•Û‘¶
		training_data[i].save(traing_dir,false);
		label_data[i].save(label_dir,false);

	}

	getchar();

	Input_Layer input;
	Convolution2D_Layer conv1(5, 5, 32, 1, true, relu), conv2(7, 7, 64, 1, true, relu, 0, 0.05f);
	Pooling2D_Layer pool1(2, 2, 2, false, MAX_POOLING), pool2(2, 2, 2, false, MAX_POOLING);
	Fully_Connect_Layer full1(100, relu), full2(10, softmax);

	Optimizer opt;
	opt.learning_rate_ = 0.1;

	Feed_Forward_NN model(input);
	model.add_layer(conv1);
	model.add_layer(pool1);
	model.add_layer(conv2);
	model.add_layer(pool2);
	model.add_layer(full1);
	model.add_layer(full2);

	model.set_optimizer(opt);

	full1.weight().print();

	for (int i = 0; i < 10000; i++){
		int idx = rand() / 600;
		cout << model.learn(training_data[idx], label_data[idx]) << endl;
	}

	for (int i = 0; i < 100; i++){
		for (int j = 0; j < batchsize; j++){
			cout << "infer------------------" << endl;
			model.infer(training_data[i]).get_Matrix(j).print(true);
			cout << "label------------------" << endl;
			label_data[i].get_Matrix(j).print(true);
			cout << "full---------------" << endl;
			//full.weight().print(true);
			getchar();
		}

	}
	return 0;
}