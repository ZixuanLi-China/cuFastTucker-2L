#include <numeric>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>
#include "parameter.h"

void loadBin(const char *indexFile, const char *valueFile, int *index,
type_of_data *value, const int order, const int nnz) {

	FILE *index_File = fopen(indexFile, "rb");
	FILE *value_File = fopen(valueFile, "rb");

	if (!index_File || !value_File) {
		printf("Unable to open file!");
		return;
	}

	fread(&index[0], sizeof(int) * order * nnz, 1, index_File);
	fread(&value[0], sizeof(type_of_data) * nnz, 1, value_File);

	fclose(index_File);
	fclose(value_File);

}

void Getting_Input_Bin(char *InputPath_train_index, char *InputPath_train_value,
		char *InputPath_test_index, char *InputPath_test_value, int order,
		int nnz_train, int nnz_test, int *index_train,
		type_of_data *value_train, int *index_test,
		type_of_data *value_test) {

	loadBin(InputPath_train_index, InputPath_train_value, index_train,
			value_train, order, nnz_train);
	loadBin(InputPath_test_index, InputPath_test_value, index_test, value_test,
			order, nnz_test);

}

void Cuda_Parameter_Initialization(int order, int core_kernel, int core_dimen,
		int *dimen, int nnz_train,
		type_of_data *value_train_host,
		type_of_data **value_train_device, int *index_train_host,
		int **index_train_device, int nnz_test,
		type_of_data *value_test_host, type_of_data **value_test_device,
		int *index_test_host, int **index_test_device,
		type_of_data ***parameter_cp_device,
		type_of_data ***parameter_cp_host_to_device) {

	*parameter_cp_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);

	cudaMalloc((void**) &(*index_train_device),
			sizeof(int) * nnz_train * order);
	cudaMalloc((void**) &(*value_train_device),
			sizeof(type_of_data) * nnz_train);

	cudaMemcpy(*index_train_device, index_train_host,
			sizeof(int) * nnz_train * order, cudaMemcpyHostToDevice);
	cudaMemcpy(*value_train_device, value_train_host,
			sizeof(type_of_data) * nnz_train, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(*index_test_device), sizeof(int) * nnz_test * order);
	cudaMalloc((void**) &(*value_test_device), sizeof(type_of_data) * nnz_test);

	cudaMemcpy(*index_test_device, index_test_host,
			sizeof(int) * nnz_test * order, cudaMemcpyHostToDevice);
	cudaMemcpy(*value_test_device, value_test_host,
			sizeof(type_of_data) * nnz_test, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(*parameter_cp_device), sizeof(type_of_data*) * order);


	for (int i = 0; i < order; i++) {

		type_of_data *temp_cp;
		cudaMalloc((void**) &temp_cp,
				sizeof(type_of_data) * dimen[i] * core_kernel);
		(*parameter_cp_host_to_device)[i] = temp_cp;

	}

	cudaMemcpy(*parameter_cp_device, *parameter_cp_host_to_device,
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);

}


void Select_Best_Result(type_of_data *train_rmse, type_of_data *test_rmse,
type_of_data *best_train_rmse, type_of_data *best_test_rmse) {

	if (*train_rmse < *best_train_rmse) {
		*best_train_rmse = *train_rmse;
	}
	if (*test_rmse < *best_test_rmse) {
		*best_test_rmse = *test_rmse;
	}

}
