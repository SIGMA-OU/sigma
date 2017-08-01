#include"for_debug.cuh"

#define DATA_END 13617 //�f�[�^�̏I���ԍ�(�f�[�^��)
#define DATA_START 1 //�f�[�^�̊J�n�ԍ�
#define DATA_DIMENSION 1024 //���̓f�[�^�̎�����
#define MINI_BATCH_SIZE 100 //�~�j�o�b�`�̃T�C�Y
#define TRAIN_NUM 10000 //�w�K��

int main(){

	string file_path = "inputh"; //�ǂݍ��ރt�@�C���̏ꏊ���w��

	//���̓f�[�^�̒�`
	Sigma_Matrix training_set(1024, DATA_END, ZERO, 0, 0);

	//�f�[�^�̓ǂݍ���
	cout << "load data" << endl;
	
	for (int i = 0; i < DATA_END; i++)
	{
		float data[DATA_DIMENSION] = { 0 }; //�z��̏�����

		string file_name = file_path + to_string(i+1) + ".txt"; //�ǂݍ��ރe�L�X�g�t�@�C���̎w��
		ifstream ifs(file_name);

		for (int k = 0; k < DATA_DIMENSION; k++){
			string tmp_str;
			getline(ifs, tmp_str);
			data[k] = stof(tmp_str); //�e�L�X�g�t�@�C�����̒l��z��Ɋi�[����
		}

		training_set.set_matrix(data,i); //���̓f�[�^�ɒl��ݒ�
	}
	cout << "finish loading" << endl;

	//���͑w�̎w��
	Input_Layer input;

	//���ԁC�o�͑w��(�w���C�������֐��C���ρC���U)���w��
	Fully_Connect_Layer middle(512, relu, 0, 0.01f), output(DATA_DIMENSION, relu, 0, 0.01f);
	Feed_Forward_NN model(input);
	model.add_layer(middle);
	model.add_layer(output);

	Optimizer opt;
	opt.learning_rate_ = 0.00001; //�w�K�W���̎w��

	model.set_optimizer(opt);

	ofstream delta("influenza_HA\\save_data\\1024-512-1024\\layer1_delta.csv", ios::app | ios::out); //�덷���o�͂���t�@�C�����w��

	//�w�K�J�n
	cout << "learn start" << endl;
	for (int i = 0; i < TRAIN_NUM; i++){

		Sigma_Matrix sgm; //��̃~�j�o�b�`���쐬

		//�~�j�o�b�`�Ƀf�[�^���ЂƂ��ǉ�
		for (int j = 0; j < MINI_BATCH_SIZE; j++)
		{
			int idx = rand() % (DATA_END-1);
			sgm.add_Matrix(training_set.get_Matrix(idx));
		}

		float valu = model.learn(sgm, sgm); //���͂Əo�͂Ƃ̌덷(1�f�[�^����)���v�Z
		delta << i << "," << valu << endl; //�w�K�̉񐔂ƌ덷���o��
		cout << valu << endl;
	}

	cout << "learn end" << endl;
	delta.close();
	//�w�K�I��

	string dir_path = "influenza_HA\\save_data\\1024-512-1024\\"; //���ʂ��o�͂���t�@�C���̏ꏊ���w��

	//�Ō�̊w�K�œ������ԑw�̒l��ۑ�
	for (int i = 0; i < DATA_END; i++){
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