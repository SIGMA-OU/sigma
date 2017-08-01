#include"Optimizer.cuh"

Optimizer::Optimizer(){
	Adam();
	Momentum_SGD();
	opt_ = SGD;
	learning_rate_ = 0.01f;
}

Optimizer::Optimizer(OPTIMIZER opt, Optimizer::Param &param){
	opt_ = opt;
	param_ = param;
}

Optimizer Optimizer::Momentum_SGD(float momentum){
	opt_ = SGD_MOMENTUM;
	param_.momentum_sgd_.momentum = momentum;
	return *this;
}

Optimizer Optimizer::Adam(float alpha, float beta_1, float beta_2, float epsilon){
	opt_ = ADAM;
	param_.adam_.alfa = alpha;
	param_.adam_.beta1 = beta_1;
	param_.adam_.beta2 = beta_2;
	param_.adam_.epsiron = epsilon;
	param_.adam_.time = 0;
	return *this;
}