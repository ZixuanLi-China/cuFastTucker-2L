#include <stdio.h>
#include "tools.h"
#include "kernel.h"

using namespace std;

int iter_number;

type_of_data learn_alpha_a;
type_of_data learn_beta_a;
type_of_data lambda_a;
type_of_data learn_rate_a;

type_of_data train_rmse;
type_of_data test_rmse;

type_of_data best_train_rmse;
type_of_data best_test_rmse;

char *InputPath_train_index;
char *InputPath_train_value;

char *InputPath_test_index;
char *InputPath_test_value;

int order;
int *dimen;

int rank_R;
int rank_J;

type_of_data data_norm;

int nnz_train;
int *index_train_host;
int *index_train_device;
type_of_data *value_train_host;
type_of_data *value_train_device;

int nnz_test;
int *index_test_host;
int *index_test_device;
type_of_data *value_test_host;
type_of_data *value_test_device;

type_of_data **parameter_cp_device;
type_of_data **parameter_cp_host_to_device;

type_of_data **parameter_a_device;
type_of_data **parameter_a_host_to_device;

type_of_data **parameter_b_device;
type_of_data **parameter_b_host_to_device;

double time_spend = 0.0;
double start_time;
double stop_time;

int main(int argc, char *argv[]) {

	if (argc == atoi(argv[5]) + 15) {

		InputPath_train_index = argv[1];
		InputPath_train_value = argv[2];

		InputPath_test_index = argv[3];
		InputPath_test_value = argv[4];

		order = atoi(argv[5]);
		dimen = (int*) malloc(sizeof(int) * order);

		for (int i = 0; i < order; i++) {
			dimen[i] = atoi(argv[6 + i]);
		}

		nnz_train = atoi(argv[6 + order]);
		nnz_test = atoi(argv[7 + order]);
		data_norm = atof(argv[8 + order]);

		rank_R = atoi(argv[9 + order]);
		rank_J = atoi(argv[10 + order]);

		iter_number = atoi(argv[11 + order]);

		learn_alpha_a = atof(argv[12 + order]);
		learn_beta_a = atof(argv[13 + order]);
		lambda_a = atof(argv[14 + order]);

	}

	index_train_host = (int*) malloc(sizeof(int) * nnz_train * order);
	value_train_host = (type_of_data*) malloc(sizeof(type_of_data) * nnz_train);

	index_test_host = (int*) malloc(sizeof(int) * nnz_test * order);
	value_test_host = (type_of_data*) malloc(sizeof(type_of_data) * nnz_test);

	start_time = Seconds();
	Getting_Input_Bin(InputPath_train_index, InputPath_train_value,
			InputPath_test_index, InputPath_test_value, order, nnz_train,
			nnz_test, index_train_host, value_train_host, index_test_host,
			value_test_host);
	stop_time = Seconds();
	printf("Tensor Loading:\t%f seconds\n", stop_time - start_time);

	start_time = Seconds();
	Cuda_Parameter_Initialization(order, rank_R, rank_J, dimen, nnz_train,
			value_train_host, &value_train_device, index_train_host,
			&index_train_device, nnz_test, value_test_host, &value_test_device,
			index_test_host, &index_test_device, &parameter_cp_device,
			&parameter_cp_host_to_device);
	stop_time = Seconds();
	printf("H2D Transfer:\t%f seconds\n", stop_time - start_time);

	start_time = Seconds();
	Parameter_Initialization(parameter_cp_host_to_device, data_norm, order,
			dimen, rank_R, 0.5, 1.5);
	stop_time = Seconds();
	printf("Initialization:\t%f seconds\n", stop_time - start_time);

	GET_RMSE_AND_MAE_CP(order, rank_R, parameter_cp_device, nnz_train,
			value_train_device, index_train_device, &best_train_rmse);
	GET_RMSE_AND_MAE_CP(order, rank_R, parameter_cp_device, nnz_test,
			value_test_device, index_test_device, &best_test_rmse);

	printf("Initial:\tTrain RMSE:%f\tTest RMSE:%f\t\n", best_train_rmse,
			best_test_rmse);

	printf("Iter\tTrain RMSE\tTest RMSE\tFactor Time\tTotal Time\n");

	for (int i = 0; i < iter_number; i++) {

		learn_rate_a = learn_alpha_a / (1 + learn_beta_a * pow(i, 1.5));

		start_time = Seconds();

		Update_Parameter_CP(order, rank_R, parameter_cp_device, nnz_train,
				value_train_device, index_train_device, learn_rate_a, lambda_a);

		stop_time = Seconds();

		time_spend += stop_time - start_time;

		GET_RMSE_AND_MAE_CP(order, rank_R, parameter_cp_device, nnz_train,
				value_train_device, index_train_device, &train_rmse);
		GET_RMSE_AND_MAE_CP(order, rank_R, parameter_cp_device, nnz_test,
				value_test_device, index_test_device, &test_rmse);

		Select_Best_Result(&train_rmse, &test_rmse, &best_train_rmse,
				&best_test_rmse);

		printf("%d\t%f\t%f\t%f\t%f\n", i, train_rmse, test_rmse,
				stop_time - start_time, time_spend);
	}

	printf("Best:\tTrain RMSE:%f\tTest RMSE:%f\t\n", best_train_rmse,
			best_test_rmse);

	start_time = Seconds();
	Update_Parameter_SVD(order, dimen, rank_R, rank_J,
			parameter_cp_host_to_device, &parameter_a_device,
			&parameter_a_host_to_device, &parameter_b_device,
			&parameter_b_host_to_device);
	stop_time = Seconds();
	printf("SVD Time:\t%f seconds\n", stop_time - start_time);

	Trans(order, dimen,  rank_R,  rank_J,
			parameter_a_host_to_device,
			parameter_b_host_to_device);

	start_time = Seconds();
	GET_RMSE_AND_MAE_Tucker(order, rank_R, rank_J, parameter_a_device,
			parameter_b_device, nnz_train, value_train_device,
			index_train_device, &best_train_rmse);
	GET_RMSE_AND_MAE_Tucker(order, rank_R, rank_J, parameter_a_device,
			parameter_b_device, nnz_test, value_test_device, index_test_device,
			&best_test_rmse);
	stop_time = Seconds();
	printf("Tucker RMSE:\t%f seconds\n", stop_time - start_time);

	printf("Best_tucker:\tTrain RMSE:%f\tTest RMSE:%f\t\n", best_train_rmse,
			best_test_rmse);

	return 0;
}
