#include"for_debug.cuh"
#include"MNIST.hpp"
#include "Imaging.cuh"

using namespace std;

int main(){

	vector<Sigma_Matrix> training_data;
	vector<Sigma_Matrix> label_data;

	cout << "load start" << endl;
	//全データの読み込み
	for (int i = 0; i < 600; i++){
		string traing_dir = "c:\\Data_Set\\MNIST\\binary\\training_" + to_string(i);
		string label_dir = "c:\\Data_Set\\MNIST\\binary\\label_" + to_string(i);

		Sigma_Matrix training_temp;
		Sigma_Matrix label_temp;

		//Binaryでload
		training_temp.load(traing_dir);
		label_temp.load(label_dir);

		//push back
		training_data.push_back(training_temp);
		label_data.push_back(label_temp);
	}
	cout << "load end" << endl;

	Input_Layer input;
	Convolution2D_Layer conv1(5, 5, 32, 1, true, relu), conv2(7, 7, 64, 1, true, relu, 0, 0.05f);


	Pooling2D_Layer pool1(2, 2, 2, false, MAX_POOLING), pool2(2, 2, 2, false, MAX_POOLING);
	Fully_Connect_Layer full1(100, relu), full2(10, softmax);

	Optimizer opt;
	opt.learning_rate_ = 0.1;

	//モデルの作成
	Feed_Forward_NN model(input);
	model.add_layer(conv1);
	model.add_layer(pool1);
	model.add_layer(conv2);
	model.add_layer(pool2);
	model.add_layer(full1);
	model.add_layer(full2);

	//学習係数の設定
	model.set_optimizer(opt);

	for (int n = 0; n < 10; n++){
		//学習(100回)
		for (int i = 0; i < 100; i++){
			int idx = rand() / 600;
			cout << model.learn(training_data[idx], label_data[idx]) << endl;
		}

		/*conv1.filter().save("");

		vector<Mat> image =  Matrix2Image(conv1.output()[0]);

		//画像の出力
		for (int i = 0; i < image.size(); i++){
		imshow("test",image[i]);
		waitKey(30);
		}*/

		model.infer(training_data[50]).print(true);

		vector<int> infer_idx;
		vector<int> label_idx;

		infer_idx = argmax_idx(full2.output());
		label_idx = argmax_idx(label_data[50]);

		int acc = equall(infer_idx, label_idx);

		cout << "accuracy = " << acc / 100.0 << endl;
	}
	getchar();

	return 0;
}