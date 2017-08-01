#include"for_debug.cuh"

#define TRAIN_NUM 10000
#define DATA_NUM 13617
#define DATA_DIMENSION 512
#define MINI_BATCH_SIZE 100

int main2(){

	string file_path = "influenza_HA\\save_data\\1024-512-1024\\";

	Sigma_Matrix training_set(DATA_DIMENSION, DATA_NUM, ZERO, 0, 0);

	//�f�[�^�̓ǂݍ���
	cout << "loading data..." << endl;

	for (int i = 0; i < DATA_NUM; i++)
	{
		string file_name = file_path + "output" + to_string(i); //�ǂݍ��ރt�@�C���̎w��

		Sigma_Matrix tmp;
		tmp.load(file_name);

		training_set.set_matrix(tmp.data(), i); //���̓f�[�^�ɒl��ݒ�
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

		Sigma_Matrix sgm; //��̃~�j�o�b�`���쐬

		//�~�j�o�b�`�Ƀf�[�^���ЂƂ��ǉ�
		for (int j = 0; j < MINI_BATCH_SIZE; j++)
		{
			int idx = rand() % (DATA_NUM-1);
			sgm.add_Matrix(training_set.get_Matrix(idx));
		}

		float valu = model.learn(sgm, sgm); //���͂Əo�͂Ƃ̌덷(1�f�[�^��)���v�Z
		delta << i << "," << valu << endl; //�w�K�̉񐔂ƌ덷���o��
		cout << valu << endl;
	}

	cout << "learn end" << endl;
	delta.close();


	string dir_path = "influenza_HA\\save_data\\512-100-512\\"; //���ʂ��o�͂���t�@�C���̏ꏊ���w��

	//�Ō�̊w�K�œ������ԑw�̒l��ۑ�
	for (int i = 0; i < DATA_NUM; i++){
		Sigma_Matrix sgm;
		sgm.add_Matrix(training_set.get_Matrix(i));
		model.infer(sgm);
		string output_file = dir_path + "output" + to_string(i);
		middle.output().save(output_file);
		middle.output().save(output_file, false);
	}

	//�Ō�̊w�K�œ����d�݂ƃo�C�A�X��ۑ�
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