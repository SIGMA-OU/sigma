#include"for_debug.cuh"
#include"Sigma_Matrix.cuh"
#include"Input_Layer.cuh"
#include"Activate_Function.cuh"

#include"Dropout_Process.cuh"
#include"Noising_Process.cuh"
#include"Fully_Connect_Layer.cuh"
#include"Convolution2D_Layer.cuh"
#include"Pooling2D_Layer.cuh"
#include"Feed_Forward_NN.cuh"
#include<vector>

#define	CONTROL_DATA 9000
#define NO_CONTROL_DATA 9000
#define TEST_DATA	200
#define IMAGE_WIDTH	150
#define IMAGE_HEIGHT 150
#define IMAGE_CHANNEL 4
#define MINIBATCH_SIZE 50
#define TRAINING_MINIBATCH_NUM 200
#define TEST_MINIBATCH_NUM 8
#define by_valu 100.0f
#define TRAINING_NUM	100000

int main_AE(){
	/*
	//データの読み込み
	cout << "read files" << endl;
	string Control_file = "E:\\Kanako_data\\CELL\\new_Blank3.txt";
	string NOT_Control_file = "E:\\Kanako_data\\CELL\\new_Cmpd3.txt";

	ifstream Control(Control_file);
	ifstream NOT_Control(NOT_Control_file);

	if (Control.fail() || NOT_Control.fail()){
	cout << "file error" << endl; return -1;
	}

	vector<vector<float>> ROW_input_data_set;

	for (int i = 0; i < CONTROL_DATA; i++){
	vector<float> tmp_data;
	string tmp;
	for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL; j++){
	getline(Control,tmp);
	tmp_data.push_back(stof(tmp));
	}
	ROW_input_data_set.push_back(tmp_data);
	}
	for (int i = 0; i < NO_CONTROL_DATA; i++){
	vector<float> tmp_data;
	string tmp;
	for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL; j++){
	getline(NOT_Control, tmp);
	tmp_data.push_back(stof(tmp));
	}
	ROW_input_data_set.push_back(tmp_data);
	}
	cout << "finish reading files" << endl;

	vector<Sigma_Matrix> training_set;
	vector<Sigma_Matrix> training_label_set;
	vector<Sigma_Matrix> test_set;
	vector<Sigma_Matrix> test_label_set;

	cout << "make training data" << endl;
	for (int i = 0; i < TRAINING_MINIBATCH_NUM; i++){
	Sigma_Matrix temp_matrix(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, 1, MINIBATCH_SIZE, ZERO);
	Sigma_Matrix tmp_label_matrix(1, 1, 1, 1, MINIBATCH_SIZE, ZERO);

	for (int j = 0; j < MINIBATCH_SIZE; j++){
	//CONTROL or NOT_CONTROL
	int a = rand() % 2;
	//Select datanum
	int b = rand() % (CONTROL_DATA - TEST_DATA);

	if (a == 0){
	temp_matrix.set_matrix(ROW_input_data_set[b].data(), j);
	}
	else{
	temp_matrix.set_matrix(ROW_input_data_set[CONTROL_DATA + b].data(), j);
	float x = 1;
	tmp_label_matrix.set_matrix(&x, j, 1);
	}
	}
	temp_matrix /= by_valu;
	training_set.push_back(temp_matrix);
	training_label_set.push_back(tmp_label_matrix);
	}
	cout << "finish making training data" << endl;

	cout << "output_training_data" << endl;

	for (int i = 0; i < training_set.size(); i++){
	string file_name = "E:\\Kanako_data\\CELL\\TRAINING_DATA\\TRAINING_MINIBATCH_";
	file_name += to_string(i);
	training_set[i].save(file_name);
	}
	for (int i = 0; i < training_label_set.size(); i++){
	string file_name = "E:\\Kanako_data\\CELL\\TRAINING_DATA\\TRAINING_LABEL_MINIBATCH_";
	file_name += to_string(i);
	training_label_set[i].save(file_name);
	}

	cout << "finish outputing" << endl;

	cout << "make test data" << endl;

	for (int i = 0; i < TEST_MINIBATCH_NUM / 2; i++){
	Sigma_Matrix temp_matrix(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, 1, MINIBATCH_SIZE, ZERO);
	Sigma_Matrix tmp_label_matrix(1, 1, 1, 1, MINIBATCH_SIZE, ZERO);
	for (int j = 0; j < MINIBATCH_SIZE; j++){
	temp_matrix.set_matrix(ROW_input_data_set[CONTROL_DATA - TEST_DATA + MINIBATCH_SIZE *i + j].data(), j);
	}
	test_set.push_back(temp_matrix);
	test_label_set.push_back(tmp_label_matrix);
	}
	for (int i = 0; i < TEST_MINIBATCH_NUM / 2; i++){
	Sigma_Matrix temp_matrix(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, 1, MINIBATCH_SIZE, ZERO);
	Sigma_Matrix tmp_label_matrix(1, 1, 1, 1, MINIBATCH_SIZE, ZERO);
	for (int j = 0; j < MINIBATCH_SIZE; j++){
	temp_matrix.set_matrix(ROW_input_data_set[CONTROL_DATA + NO_CONTROL_DATA - TEST_DATA + MINIBATCH_SIZE *i + j].data(), j);
	float x = 1;
	tmp_label_matrix.set_matrix(&x, j, 1);
	}
	test_set.push_back(temp_matrix);
	test_label_set.push_back(tmp_label_matrix);
	}
	cout << "finish making test data" << endl;

	cout << "output test data" << endl;
	for (int i = 0; i < test_set.size(); i++){
	string file_name = "E:\\Kanako_data\\CELL\\TEST_DATA\\TEST_MINIBATCH_";
	file_name += to_string(i);
	test_set[i].save(file_name);
	}
	for (int i = 0; i < test_label_set.size(); i++){
	string file_name = "E:\\Kanako_data\\CELL\\TEST_DATA\\TEST_LABEL_MINIBATCH_";
	file_name += to_string(i);
	test_label_set[i].save(file_name);
	}
	cout << "finish outputing data" << endl;
	getchar();*/
	/*cout << "read files" << endl;
	string Control_file = "E:\\Kanako_data\\CELL\\Blank_marge.txt";
	string NOT_Control_file = "E:\\Kanako_data\\CELL\\Cmpd_marge.txt";

	ifstream Control(Control_file);
	ifstream NOT_Control(NOT_Control_file);

	if (Control.fail() || NOT_Control.fail()){
	cout << "file error" << endl; return -1;
	}

	vector<vector<float>> ROW_input_data_set;

	for (int i = 0; i < CONTROL_DATA; i++){
	vector<float> tmp_data;
	string tmp;
	for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL; j++){
	getline(Control, tmp);
	tmp_data.push_back(stof(tmp));
	}
	ROW_input_data_set.push_back(tmp_data);
	}
	for (int i = 0; i < NO_CONTROL_DATA; i++){
	vector<float> tmp_data;
	string tmp;
	for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL; j++){
	getline(NOT_Control, tmp);
	tmp_data.push_back(stof(tmp));
	}
	ROW_input_data_set.push_back(tmp_data);
	}
	cout << "finish reading files" << endl;

	vector<Sigma_Matrix> training_set;
	vector<Sigma_Matrix> traiing_label_set;
	vector<Sigma_Matrix> test_set;
	vector<Sigma_Matrix> test_label_set;

	cout << "make training data" << endl;
	for (int i = 0; i < TRAINING_MINIBATCH_NUM; i++){
	Sigma_Matrix temp_matrix(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, 1, MINIBATCH_SIZE, ZERO);
	Sigma_Matrix tmp_label_matrix(1, 1, 1, 1, MINIBATCH_SIZE, ZERO);

	for (int j = 0; j < MINIBATCH_SIZE; j++){
	//CONTROL or NOT_CONTROL
	int a = rand() % 2;
	//Select datanum
	int b = rand() % (CONTROL_DATA - TEST_DATA);

	if (a == 0){
	temp_matrix.set_matrix(ROW_input_data_set[b].data(), j);
	}
	else{
	temp_matrix.set_matrix(ROW_input_data_set[CONTROL_DATA + b].data(), j);
	float x = 1;
	tmp_label_matrix.set_matrix(&x, j, 1);
	}
	}
	temp_matrix /= by_valu;
	training_set.push_back(temp_matrix);
	traiing_label_set.push_back(tmp_label_matrix);
	}
	cout << "finish making training data" << endl;

	cout << "make test data" << endl;

	for (int i = 0; i < TEST_MINIBATCH_NUM / 2; i++){
	Sigma_Matrix temp_matrix(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, 1, MINIBATCH_SIZE, ZERO);
	Sigma_Matrix tmp_label_matrix(1, 1, 1, 1, MINIBATCH_SIZE, ZERO);
	for (int j = 0; j < MINIBATCH_SIZE; j++){
	temp_matrix.set_matrix(ROW_input_data_set[CONTROL_DATA - TEST_DATA + MINIBATCH_SIZE *i + j].data(), j);
	}
	test_set.push_back(temp_matrix);
	test_label_set.push_back(tmp_label_matrix);
	}
	for (int i = 0; i < TEST_MINIBATCH_NUM / 2; i++){
	Sigma_Matrix temp_matrix(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, 1, MINIBATCH_SIZE, ZERO);
	Sigma_Matrix tmp_label_matrix(1, 1, 1, 1, MINIBATCH_SIZE, ZERO);
	for (int j = 0; j < MINIBATCH_SIZE; j++){
	temp_matrix.set_matrix(ROW_input_data_set[CONTROL_DATA + NO_CONTROL_DATA - TEST_DATA + MINIBATCH_SIZE *i + j].data(), j);
	float x = 1;
	tmp_label_matrix.set_matrix(&x, j, 1);
	}
	test_set.push_back(temp_matrix);
	test_label_set.push_back(tmp_label_matrix);
	}
	cout << "finish making test data" << endl;*/

	std::cout << "make minibatch" << endl;
	string Training_data_dir = "C:\\Data_Set\\Tanabe\\TRAINING_DATA\\";
	string Test_data_dir = "C:\\Data_Set\\Tanabe\\TRAINING_DATA\\";

	vector<Sigma_Matrix> training_set;
	//vector<Sigma_Matrix> training_label_set;
	//vector<Sigma_Matrix> test_set;
	//vector<Sigma_Matrix> test_label_set;

	for (int i = 0; i < TRAINING_MINIBATCH_NUM; i++){
		string training_file_path = Training_data_dir + "TRAINING_MINIBATCH_" + to_string(i);
		string label_file_path = Training_data_dir + "TRAINING_LABEL_MINIBATCH_" + to_string(i);
		Sigma_Matrix training_temp;
		Sigma_Matrix label_temp;
		training_temp.load(training_file_path);
		label_temp.load(label_file_path);
		training_set.push_back(training_temp);
		//training_label_set.push_back(training_temp);
	}
	/*for (int i = 0; i < TEST_MINIBATCH_NUM; i++){
		string test_file_path = Test_data_dir + "TRAINING_MINIBATCH_" + to_string(i);
		string label_file_path = Test_data_dir + "TRAINING_LABEL_MINIBATCH_" + to_string(i);
		Sigma_Matrix test_temp;
		Sigma_Matrix label_temp;
		test_temp.load(test_file_path);
		label_temp.load(label_file_path);
		test_set.push_back(test_temp);
		//test_label_set.push_back(test_temp);
	}
	*/
	std::cout << "finish making dataset" << endl;

	//モデルの作成
	Input_Layer input;
	Fully_Connect_Layer full1(2000, relu, 1/90000), full2(90000, relu, 1/1000);

	Optimizer opt;
	opt.learning_rate_ = 0.00000001;

	Feed_Forward_NN model(input);
	model.add_layer(full1);
	model.add_layer(full2);
	model.set_optimizer(opt);

	ofstream delta("delta.csv", ios::app | ios::out);
	ofstream result("result.csv", ios::app | ios::out);
	//ofstream accuracy("accuracy.csv", ios::app | ios::out);

	//学習
	for (int i = 0; i <= TRAINING_NUM; i++){
		int idx = rand() % TRAINING_MINIBATCH_NUM;
		float delta_val = model.learn(training_set[idx], training_set[idx]);
		if (i % 1 == 0){
			std::cout << i << " : " << delta_val << endl;
			delta << i << "," << delta_val << endl;
		}
		if (i % TRAINING_NUM == 99999){
			int CONTROL_acc = 0;
			int NOT_CONTROL_acc = 0;
		//	for (int j = 7; j < TEST_MINIBATCH_NUM; j++){
				Sigma_Matrix result_matrix = model.infer(training_set[7]);
					for (int l = 0; l < 90000; l++){
						float val = result_matrix.get_Matrix(0).at(l);
						result << val << endl;
					}
		//	}
			//accuracy << i << "," << (float)CONTROL_acc / 200.0f << "," << (float)NOT_CONTROL_acc / 200.0f << "," << (float)(CONTROL_acc + NOT_CONTROL_acc) / 400.0f << endl;
			//std::cout << "accuracy : " << (float)(CONTROL_acc + NOT_CONTROL_acc) / 400.0f << endl;
		}
	}
	/*
	opt.learning_rate_ = 0.01f;
	model.set_optimizer(opt);

	for (int i = TRAINING_NUM + 1; i <= TRAINING_NUM * 2; i++){
		int idx = rand() % TRAINING_MINIBATCH_NUM;
		float delta_val = model.learn(training_set[idx], training_label_set[idx]);
		std::cout << i << " : " << delta_val << endl;
		delta << i << "," << delta_val << endl;

		if (i % 500 == 0){
			int CONTROL_acc = 0;
			int NOT_CONTROL_acc = 0;
			for (int j = 7; j < TEST_MINIBATCH_NUM; j++){
				Sigma_Matrix result_matrix = model.infer(test_set[j]);
				//result << i << ",";
				for (int k = 0; k < 50; k++){
					float val = result_matrix.get_Matrix(k).at(0);
					//result << val << ",";
					if (j < TEST_MINIBATCH_NUM / 2){
						if (val < 0.5)CONTROL_acc++;
					}
					else{
						if (val > 0.5)NOT_CONTROL_acc++;
					}
				}
			}
		//	result << endl;
			accuracy << i << "," << (float)CONTROL_acc / 200.0f << "," << (float)NOT_CONTROL_acc / 200.0f << "," << (float)(CONTROL_acc + NOT_CONTROL_acc) / 400.0f << endl;
			std::cout << "accuracy : " << (float)(CONTROL_acc + NOT_CONTROL_acc) / 400.0f << endl;
		}
	}
	*/
	delta.close();
	result.close();
	//accuracy.close();

	return 0;
}