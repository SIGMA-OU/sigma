#include"for_debug.cuh"

#define DATA_END 13617 //データの終了番号(データ数)
#define DATA_START 1 //データの開始番号
#define DATA_DIMENSION 1024 //入力データの次元数
#define MINI_BATCH_SIZE 100 //ミニバッチのサイズ
#define TRAIN_NUM 10000 //学習回数

int main(){

	string file_path = "inputh"; //読み込むファイルの場所を指定

	//入力データの定義
	Sigma_Matrix training_set(1024, DATA_END, ZERO, 0, 0);

	//データの読み込み
	cout << "load data" << endl;
	
	for (int i = 0; i < DATA_END; i++)
	{
		float data[DATA_DIMENSION] = { 0 }; //配列の初期化

		string file_name = file_path + to_string(i+1) + ".txt"; //読み込むテキストファイルの指定
		ifstream ifs(file_name);

		for (int k = 0; k < DATA_DIMENSION; k++){
			string tmp_str;
			getline(ifs, tmp_str);
			data[k] = stof(tmp_str); //テキストファイル内の値を配列に格納する
		}

		training_set.set_matrix(data,i); //入力データに値を設定
	}
	cout << "finish loading" << endl;

	//入力層の指定
	Input_Layer input;

	//中間，出力層の(層数，活性化関数，平均，分散)を指定
	Fully_Connect_Layer middle(512, relu, 0, 0.01f), output(DATA_DIMENSION, relu, 0, 0.01f);
	Feed_Forward_NN model(input);
	model.add_layer(middle);
	model.add_layer(output);

	Optimizer opt;
	opt.learning_rate_ = 0.00001; //学習係数の指定

	model.set_optimizer(opt);

	ofstream delta("influenza_HA\\save_data\\1024-512-1024\\layer1_delta.csv", ios::app | ios::out); //誤差を出力するファイルを指定

	//学習開始
	cout << "learn start" << endl;
	for (int i = 0; i < TRAIN_NUM; i++){

		Sigma_Matrix sgm; //空のミニバッチを作成

		//ミニバッチにデータをひとつずつ追加
		for (int j = 0; j < MINI_BATCH_SIZE; j++)
		{
			int idx = rand() % (DATA_END-1);
			sgm.add_Matrix(training_set.get_Matrix(idx));
		}

		float valu = model.learn(sgm, sgm); //入力と出力との誤差(1データごと)を計算
		delta << i << "," << valu << endl; //学習の回数と誤差を出力
		cout << valu << endl;
	}

	cout << "learn end" << endl;
	delta.close();
	//学習終了

	string dir_path = "influenza_HA\\save_data\\1024-512-1024\\"; //結果を出力するファイルの場所を指定

	//最後の学習で得た中間層の値を保存
	for (int i = 0; i < DATA_END; i++){
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