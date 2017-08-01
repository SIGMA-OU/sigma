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

#define DISH 4

int main__(){
	
	int image_data_num = 0;
	float* image_data;
	image_data = new float[150 * 150 * 4 * 3500];

	string output_file = "F:\\Tanabe\\DAT_DATA_Blank\\control_1";

	string dir_path = "F:\\Tanabe\\new_Blank\\";

	//for (int i = 1; i <= 2318;){
		string file_path = dir_path;

		int image_num = 1;

		while (true){
			string chan0_name = file_path + to_string(0) + "\\image (" + to_string(image_num) + ").png";
			Mat channel0 = imread(chan0_name, 0);
			if (channel0.empty())break;
			string chan1_name = file_path + to_string(1) + "\\image (" + to_string(image_num) + ").png";
			string chan2_name = file_path + to_string(2) + "\\image (" + to_string(image_num) + ").png";
			string chan3_name = file_path + to_string(3) + "\\image (" + to_string(image_num) + ").png";
			Mat channel1 = imread(chan1_name, 0);
			Mat channel2 = imread(chan2_name, 0);
			Mat channel3 = imread(chan3_name, 0);
			image_num++;

			for (int n = 0; n < 22500; n++){
				image_data[image_data_num * 90000 + n] = (float)channel0.data[n];
			}
			for (int n = 0; n < 22500; n++){
				image_data[image_data_num * 90000 + 22500 + n] = (float)channel1.data[n];
			}
			for (int n = 0; n < 22500; n++){
				image_data[image_data_num * 90000 + 22500 * 2 + n] = (float)channel2.data[n];
			}
			for (int n = 0; n < 22500; n++){
				image_data[image_data_num * 90000 + 22500 * 3 + n] = (float)channel3.data[n];
			}
			image_data_num++;
			std::cout << " image_num:" << image_data_num << endl;
		}
	//}
		Sigma_Matrix temp(150, 150, 4, 1, image_data_num, ZERO);
		cudaMemcpy(temp.data(), image_data, temp.size(), cudaMemcpyHostToDevice);
		temp.save(output_file);
		std::cout << "save_file" << endl;
		delete image_data;
	
	return 0;
}