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
type_of_data train_mae;

type_of_data test_rmse;
type_of_data test_mae;

type_of_data best_train_rmse;
type_of_data best_train_mae;

type_of_data best_test_rmse;
type_of_data best_test_mae;

char *InputPath_train;
char *InputPath_test;

int order;
int core_kernel;
int core_length;

int *dimen;
int core_dimen;

double data_norm;

int nnz_train;
type_of_data **value_train_host;
type_of_data **value_train_device;
type_of_data **value_train_host_to_device;
int **index_train_host;
int **index_train_device;
int **index_train_host_to_device;

int nnz_test;
type_of_data *value_test_host;
type_of_data *value_test_device;
int *index_test_host;
int *index_test_device;

type_of_data **parameter_cp_host;
type_of_data **parameter_cp_host_to_device;
type_of_data **parameter_cp_device;

type_of_data **parameter_a_device;
type_of_data **parameter_a_host_to_device;

type_of_data **parameter_b_device;
type_of_data **parameter_b_host_to_device;

double time_spend = 0.0;
double start_time;
double stop_time;

int main(int argc, char *argv[]) {

	if (argc == 10) {

		InputPath_train = argv[1];
		InputPath_test = argv[2];
		core_kernel = atoi(argv[3]);
		order = atoi(argv[4]);
		core_dimen = atoi(argv[5]);

		iter_number = atoi(argv[6]);

		learn_alpha_a = atof(argv[7]);
		learn_beta_a = atof(argv[8]);
		lambda_a = atof(argv[9]);

		core_length = 1;
		for (int i = 0; i < order; i++) {
			core_length *= core_kernel;
		}

	}

	Getting_Input(InputPath_train, InputPath_test, order, &dimen, &nnz_train,
			&nnz_test, &index_train_host, &value_train_host, &index_test_host,
			&value_test_host, &data_norm);

	printf("nnz_train:\t%d\n", nnz_train);
	printf("nnz_test:\t%d\n", nnz_test);

	for (int i = 0; i < order; i++) {
		printf("order %d:\t%d\n", i + 1, dimen[i]);
	}
	printf("data_norm:\t%f\n", data_norm);

	Parameter_Initialization(order, core_kernel, core_length, dimen, data_norm,
			&parameter_cp_host);

	Cuda_Parameter_Initialization(order, core_kernel, core_dimen, dimen,
			nnz_train, value_train_host, &value_train_device,
			&value_train_host_to_device, index_train_host, &index_train_device,
			&index_train_host_to_device, nnz_test, value_test_host,
			&value_test_device, index_test_host, &index_test_device,
			parameter_cp_host, &parameter_cp_device,
			&parameter_cp_host_to_device, &parameter_a_device,
			&parameter_a_host_to_device, &parameter_b_device,
			&parameter_b_host_to_device);

	GET_RMSE_AND_MAE_CPD(order, core_kernel, parameter_cp_device, nnz_train,
			value_train_host_to_device, index_train_host_to_device,
			&best_train_rmse, &best_train_mae);

	GET_RMSE_AND_MAE_CPD(order, core_kernel, parameter_cp_device, nnz_test,
			value_test_device, index_test_device, &best_test_rmse,
			&best_test_mae);

	printf(
			"initial_cp:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			best_train_rmse, best_test_rmse, best_train_mae, best_test_mae);

	Update_Parameter_SVD(order, dimen, core_kernel, core_dimen,
			parameter_a_host_to_device, parameter_b_host_to_device,
			parameter_cp_host_to_device);

	GET_RMSE_AND_MAE_Tucker(order, core_kernel, core_dimen, parameter_a_device,
			parameter_b_device, nnz_train, value_train_host_to_device,
			index_train_host_to_device, &best_train_rmse, &best_train_mae);

	GET_RMSE_AND_MAE_Tucker(order, core_kernel, core_dimen, parameter_a_device,
			parameter_b_device, nnz_test, value_test_device, index_test_device,
			&best_test_rmse, &best_test_mae);

	printf(
			"initial_tucker:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			best_train_rmse, best_test_rmse, best_train_mae, best_test_mae);

	printf(
			"iter\ttrain rmse\ttest rmse\ttrain mae\ttest mae\tfactor time\ttotal time\tcumulative time\n");

	for (int i = 0; i < iter_number; i++) {

		learn_rate_a = learn_alpha_a / (1 + learn_beta_a * pow(i, 1.5));

		start_time = Seconds();

		Update_Parameter_CP(order, core_kernel, parameter_cp_device, nnz_train,
				value_train_host_to_device, index_train_host_to_device,
				learn_rate_a, lambda_a);

		stop_time = Seconds();

		time_spend += stop_time - start_time;

		GET_RMSE_AND_MAE_CPD(order, core_kernel, parameter_cp_device, nnz_train,
				value_train_host_to_device, index_train_host_to_device,
				&train_rmse, &train_mae);
		GET_RMSE_AND_MAE_CPD(order, core_kernel, parameter_cp_device, nnz_test,
				value_test_device, index_test_device, &test_rmse, &test_mae);

		Select_Best_Result(&train_rmse, &train_mae, &test_rmse, &test_mae,
				&best_train_rmse, &best_train_mae, &best_test_rmse,
				&best_test_mae);

		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", i, train_rmse, test_rmse,
				train_mae, test_mae, stop_time - start_time,
				stop_time - start_time, time_spend);
	}

	printf("best:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			best_train_rmse, best_test_rmse, best_train_mae, best_test_mae);

	start_time = Seconds();

	Update_Parameter_SVD(order, dimen, core_kernel, core_dimen,
			parameter_a_host_to_device, parameter_b_host_to_device,
			parameter_cp_host_to_device);

	stop_time = Seconds();

	printf("svd time:\t%f\n", stop_time - start_time);

	GET_RMSE_AND_MAE_Tucker(order, core_kernel, core_dimen, parameter_a_device,
			parameter_b_device, nnz_train, value_train_host_to_device,
			index_train_host_to_device, &best_train_rmse, &best_train_mae);

	GET_RMSE_AND_MAE_Tucker(order, core_kernel, core_dimen, parameter_a_device,
			parameter_b_device, nnz_test, value_test_device, index_test_device,
			&best_test_rmse, &best_test_mae);

	printf(
			"best_tucker:\ttrain rmse:%f\ttest rmse:%f\ttrain mae:%f\ttest mae:%f\t\n",
			best_train_rmse, best_test_rmse, best_train_mae, best_test_mae);

	return 0;
}
