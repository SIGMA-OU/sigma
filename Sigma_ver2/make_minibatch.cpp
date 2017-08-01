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

int main_make_minibatch(){

	ifstream ifs("C:\\Data_Set\\Tanabe\\cell_name.csv");
	for (int i = 0; i < 1854; i++){
		int image_data_num = 0;
		float* image_data;
		image_data = new float[150 * 150 * 4 * 3500];

		string output_file = "F:\\Tanabe\\DAT_DATA\\" + to_string(i);
		string dir_num = to_string(i);

		string dir_path = "C:\\Data_Set\\Tanabe\\cut_data\\" + dir_num + "\\";

		string plate_num[4];
		for (int j = 0; j < 4; j++){
			string tmp;
			getline(ifs, tmp);
			string token;
			istringstream stream(tmp);
			for (int k = 0; k < 2; k++){
				getline(stream, token, ',');
				plate_num[j] += token;
				if (k == 0)plate_num[j] += '_';
			}
		}

		for (int j = 0; j < 4; j++){
			
			for (int k = 0; k < 4; k++){
				string file_path = dir_path + plate_num[j] +"f0" + to_string(k);

				int image_num = 0;
				
				while (true){
					string chan0_name = file_path +"d0_" + to_string(image_num) + ".png";
					Mat channel0 = imread(chan0_name,0);
					if (channel0.empty())break;
					string chan1_name = file_path + "d1_" + to_string(image_num) + ".png";
					string chan2_name = file_path + "d2_" + to_string(image_num) + ".png";
					string chan3_name = file_path + "d3_" + to_string(image_num) + ".png";
					Mat channel1 = imread(chan1_name,0);
					Mat channel2 = imread(chan2_name,0);
					Mat channel3 = imread(chan3_name,0);
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
					cout << "dir: "<<i<<" image_num:" << image_data_num << endl;
				}			
			}
		}
		Sigma_Matrix temp(150,150,4,1,image_data_num,ZERO);
		cudaMemcpy(temp.data(),image_data,temp.size(),cudaMemcpyHostToDevice);
		temp.save(output_file);
		cout << "save_file" << i << endl;
		delete image_data;
	}

	return 0;
}