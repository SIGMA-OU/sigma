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

int maindat(){
	int image_data_num = 0;
	float* image_data;
	image_data = new float[100 * 100 * 3 * 3500];

	string output_file = "F:\\kahun\\sugi_2_DAT5";
	string dir_path = "F:\\kahun\\input_data\\cut_sugi\\";
	//Œ³‰æ‘œƒ‹[ƒv
	for (int j = 251; j <= 302; j++){
		string file_path = dir_path;

		
		
		string chan0_name = file_path + "0\\image (" + to_string(j) + ").png";
		Mat channel0 = imread(chan0_name, 0);
		if (channel0.empty())break;
		string chan1_name = file_path + "1\\image (" + to_string(j) + ").png";
		string chan2_name = file_path + "2\\image (" + to_string(j) + ").png";
		Mat channel1 = imread(chan1_name, 0);
		Mat channel2 = imread(chan2_name, 0);

		for (int n = 0; n < 10000; n++){
			image_data[image_data_num * 30000 + n] = (float)channel0.data[n];
		}
		for (int n = 0; n < 10000; n++){
			image_data[image_data_num * 30000 + 10000 + n] = (float)channel1.data[n];
		}
		for (int n = 0; n < 10000; n++){
			image_data[image_data_num * 30000 + 10000 * 2 + n] = (float)channel2.data[n];
		}
		image_data_num++;
		cout << " image_num:" << image_data_num << endl;
		
	}
	for (int j = 1; j <= 250; j++){
		string file_path = dir_path;

		int image_num = 0;


		string chan0_name = file_path + "0\\image (" + to_string(j) + ").png";
		Mat channel0 = imread(chan0_name, 0);
		if (channel0.empty())break;
		string chan1_name = file_path + "1\\image (" + to_string(j) + ").png";
		string chan2_name = file_path + "2\\image (" + to_string(j) + ").png";
		Mat channel1 = imread(chan1_name, 0);
		Mat channel2 = imread(chan2_name, 0);
		image_num++;

		for (int n = 0; n < 10000; n++){
			image_data[image_data_num * 30000 + n] = (float)channel0.data[n];
		}
		for (int n = 0; n < 10000; n++){
			image_data[image_data_num * 30000 + 10000 + n] = (float)channel1.data[n];
		}
		for (int n = 0; n < 10000; n++){
			image_data[image_data_num * 30000 + 10000 * 2 + n] = (float)channel2.data[n];
		}
		image_data_num++;
		cout << " image_num:" << image_data_num << endl;

	}

	Sigma_Matrix temp(100, 100, 3, 1, image_data_num, ZERO);
	cudaMemcpy(temp.data(), image_data, temp.size(), cudaMemcpyHostToDevice);
	temp.save(output_file);
	cout << "push any key" << endl;
	getchar();
	delete image_data;

	return 0;
}