#include "parameter.h"

void Update_Parameter_CP(const int order, const int core_kernel,
		type_of_data **parameter_a_device, const int nnz_train,
		type_of_data **value_train_device, int **index_train_device,
		type_of_data learn_rate_a, type_of_data lambda_a);

void Update_Parameter_SVD(const int order, int *dimen, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a_host_to_device,
		type_of_data **parameter_b_host_to_device,
		type_of_data **parameter_cp_host_to_device);

void GET_RMSE_AND_MAE_CPD(const int order, const int core_kernel,
		type_of_data **parameter_a, const int nnz, type_of_data **value,
		int **index, type_of_data *rmse, type_of_data *mae);

void GET_RMSE_AND_MAE_CPD(const int order, const int core_kernel,
		type_of_data **parameter_a, const int nnz, type_of_data *value,
		int *index, type_of_data *rmse, type_of_data *mae);

void GET_RMSE_AND_MAE_Tucker(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data **value, int **index, type_of_data *rmse,
		type_of_data *mae);

void GET_RMSE_AND_MAE_Tucker(const int order, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, type_of_data *value,
		int *index, type_of_data *rmse, type_of_data *mae);
