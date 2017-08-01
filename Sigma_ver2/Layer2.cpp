#include"for_debug.cuh"

#define TRAIN_NUM 10000
#define DATA_NUM 13617
#define DATA_DIMENSION 512
#define MINI_BATCH_SIZE 100

int main2(){

	string file_path = "influenza_HA\\save_data\\1024-512-1024\\";

	Sigma_Matrix training_set(DATA_DIMENSION, DATA_NUM, ZERO, 0, 0);

	//データの読み込み
	cout << "loading data..." << endl;

	for (int i = 0; i < DATA_NUM; i++)
	{
		string file_name = file_path + "output" + to_string(i); //読み込むファイルの指定

		Sigma_Matrix tmp;
		tmp.load(file_name);

		training_set.set_matrix(tmp.data(), i); //入力データに値を設定
	}
	cout << "finish loading" << endl;

	Input_Layer input;
	Fully_Connect_Layer middle(100, relu, 0, 0.01f), output(512, relu, 0, 0.01f);

	Feed_Forward_NN model(input);
	model.add_layer(middle);
	model.add_layer(output);

	Optimizer opt;
	opt.learning_rate_ = 0.00001;

	model.set_optimizer(opt);

	ofstream delta("influenza_HA//save_data\\512-100-512\\layer2_delta.csv", ios::app | ios::out);

	cout << "learn start" << endl;
	for (int i = 0; i < TRAIN_NUM; i++){

		Sigma_Matrix sgm; //空のミニバッチを作成

		//ミニバッチにデータをひとつずつ追加
		for (int j = 0; j < MINI_BATCH_SIZE; j++)
		{
			int idx = rand() % (DATA_NUM-1);
			sgm.add_Matrix(training_set.get_Matrix(idx));
		}

		float valu = model.learn(sgm, sgm); //入力と出力との誤差(1データ分)を計算
		delta << i << "," << valu << endl; //学習の回数と誤差を出力
		cout << valu << endl;
	}

	cout << "learn end" << endl;
	delta.close();


	string dir_path = "influenza_HA\\save_data\\512-100-512\\"; //結果を出力するファイルの場所を指定

	//最後の学習で得た中間層の値を保存
	for (int i = 0; i < DATA_NUM; i++){
		Sigma_Matrix sgm;
		sgm.add_Matrix(training_set.get_Matrix(i));
		model.infer(sgm);
		string output_file = dir_path + "output" + to_string(i);
		middle.output().save(output_file);
		middle.output().save(output_file, false);
	}

	//最後の学習で得た重みとバイアスを保存
	string middle_weight_file = dir_path + "middle_weight";
	middle.weight().save(middle_weight_file);
	string middle_bias_file = dir_path + "middle_bias";
	middle.bias().save(middle_bias_file);


	string output_weight_file = dir_path + "output_weight";
	output.weight().save(output_weight_file);
	string output_bias_file = dir_path + "output_bias";
	output.bias().save(output_bias_file);

	cout << "press Enter key" << endl;
	getchar();

	return 0;

}