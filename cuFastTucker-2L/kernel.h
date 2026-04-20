#include "parameter.h"

void Parameter_Initialization(type_of_data **parameter_cp,
		type_of_data data_norm, int order, int *dimen, int core_kernel,
		type_of_data min_val, type_of_data max_val);

void Update_Parameter_CP(const int order, const int core_kernel,
		type_of_data **parameter_a_device, const int nnz_train,
		type_of_data *value_train_device, int *index_train_device,
		type_of_data learn_rate_a, type_of_data lambda_a);

void Update_Parameter_SVD(const int order, int *dimen, const int rank_R,
		const int rank_J,
		type_of_data **parameter_cp_host_to_device,
		type_of_data ***parameter_a_device,
		type_of_data ***parameter_a_host_to_device,
		type_of_data ***parameter_b_device,
		type_of_data ***parameter_b_host_to_device);

void GET_RMSE_AND_MAE_CP(const int order, const int rank_R,
		type_of_data **parameter_a, const int nnz, type_of_data *value,
		int *index, type_of_data *rmse);

void GET_RMSE_AND_MAE_Tucker(const int order, const int rank_R,
		const int rank_J, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, type_of_data *value,
		int *index, type_of_data *rmse);

void Trans(const int order, int *dimen, const int rank_R, const int rank_J,
		type_of_data **parameter_a_host_to_device,
		type_of_data **parameter_b_host_to_device);
