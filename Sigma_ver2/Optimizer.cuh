/*!
* \file    Optimizer.cuh
* \brief	Optimiezer‚ð’è‹`‚µ‚Ä‚¢‚Ü‚·
* \date    2016/06/04
* \author  Asatani Satoshi(s.asatani@bpe.es.osaka-u.ac)
*/

#ifndef OPTIMIZER_CUH_
#define OPTIMIZER_CUH_
#undef NDEBUG

#include"DEFINES.cuh"
#include"Sigma_Matrix.cuh"

class Optimizer{
public:
	struct Param{

		struct Momentum_SGD{
			float momentum;
		}momentum_sgd_;

		struct Adam{
			float alfa;
			float beta1;
			float beta2;
			float epsiron;
			unsigned int time;
			Sigma_Matrix Fist_Moment;
			Sigma_Matrix First_Moment_Hut;
			Sigma_Matrix Second_Moment;
			Sigma_Matrix Second_Moment_Hut;
		}adam_;
	};

	Optimizer();

	Optimizer(OPTIMIZER opt, Optimizer::Param &param);

	Optimizer::Param::Momentum_SGD& Momentum_SGD_param(){ return param_.momentum_sgd_; };
	Optimizer::Param::Adam& Adam_param(){ return param_.adam_; };

	OPTIMIZER opt(){ return opt_; };

	float learning_rate(){ return learning_rate_; };

	Optimizer Momentum_SGD(float momentum = 0.5);
	Optimizer Adam(float alpha = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float epsilon = 1.0E-8);

public:
	OPTIMIZER opt_;
	Optimizer::Param param_;
	float learning_rate_;
};

#endif