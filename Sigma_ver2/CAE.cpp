/*
CAEのデモです。
*/

#include"for_debug.cuh"
#include "Imaging.cuh"

int main3(){
	//データの読み込み
	cout << "Load Data set" << endl;
	vector<Sigma_Matrix> training_data;
	vector<Sigma_Matrix> training_label_data;
	Sigma_Matrix test_data;
	Sigma_Matrix test_label_data;
	int training_minibatch_num = 600;
	for (int i = 0; i < 600; i++){
		Sigma_Matrix training_temp, label_temp;
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
	Convolution2D_Layer conv1(3, 3, 96, 1, true, relu, 0, 0.01f), conv2(3, 3, 1, 1, true, relu,0, 0.01f);

	Feed_Forward_NN model(input);
	model.add_layer(conv1);
	model.add_layer(conv2);

	model.infer(training_data[0]);

	Optimizer opt;
	opt.learning_rate_ = 0.001f;
	model.set_optimizer(opt);

	for (int i = 0; i < 10000; i++){
		int idx = rand() % training_data.size();
		cout << model.learn(training_data[idx], training_data[idx]) << endl;
		if (i % 500 == 499){
			vector<vector<Mat>> images = Matrix2Image(conv2.output());
			for (int j = 0; j < images.size(); j++){
				imshow("test",images[j][0]);
				waitKey();
			}
			vector<vector<Mat>> filter = Matrix2Image(conv1.filter());
			for (int j = 0; j < filter.size();j++){
				Mat dst_image;
				resize(filter[j][0],dst_image,cv::Size(),20,20);

				imshow("filter", dst_image);
				waitKey();
			}
			vector<vector<Mat>> middle_output = Matrix2Image(conv1.output());
			for (int j = 0; j < middle_output.size(); j++){
				for (int k = 0; k < 96; k++){
				imshow("feature_maps", middle_output[j][k]);
				waitKey();
				}
			}
		}
	}
	
}