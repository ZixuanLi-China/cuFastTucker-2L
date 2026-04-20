#include <sys/time.h>
#include "parameter.h"

inline double Seconds() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void Getting_Input_Bin(char *InputPath_train_index, char *InputPath_train_value,
		char *InputPath_test_index, char *InputPath_test_value, int order,
		int nnz_train, int nnz_test, int *index_train,
		type_of_data *value_train, int *index_test,
		type_of_data *value_test);

void Parameter_Initialization(int order, int core_kernel, int *dimen,
		type_of_data data_norm, type_of_data ***parameter_cp);

void Cuda_Parameter_Initialization(int order, int core_kernel, int core_dimen,
		int *dimen, int nnz_train,
		type_of_data *value_train_host,
		type_of_data **value_train_device, int *index_train_host,
		int **index_train_device, int nnz_test,
		type_of_data *value_test_host, type_of_data **value_test_device,
		int *index_test_host, int **index_test_device,
		type_of_data ***parameter_cp_device,
		type_of_data ***parameter_cp_host_to_device);

void Select_Best_Result(type_of_data *train_rmse, type_of_data *test_rmse,
		type_of_data *best_train_rmse, type_of_data *best_test_rmse);
