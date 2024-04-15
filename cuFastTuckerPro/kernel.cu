#include <vector>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "parameter.h"

__global__ void Update_Parameter_CPD_SGD(const int order, const int core_kernel,
		type_of_data **parameter_cp, const int nnz, const type_of_data *value,
		const int *index, const type_of_data learn_rate_a,
		const type_of_data lambda_a) {

	int core = core_kernel;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	type_of_data gs[order_size];
	type_of_data a[order_size];

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
			gs[order_index] = 1.0;
		}
#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {

			a[order_index] = parameter_cp[order_index][index[nnz_index * order
					+ order_index] * core_kernel + lane_id];

#pragma unroll
			for (int inner_order_index = 0; inner_order_index < order_size;
					inner_order_index++) {
				if (inner_order_index != order_index) {
					gs[inner_order_index] *= a[order_index];
				}
			}
		}

		type_of_data x_r_pre = a[0] * gs[0];

		if (core_kernel == 4) {
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
			x_r_pre = __shfl_sync(mask, x_r_pre, 0, 4);
		} else if (core_kernel == 8) {
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 4);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
			x_r_pre = __shfl_sync(mask, x_r_pre, 0, 8);
		} else if (core_kernel == 16) {
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 8);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 4);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
			x_r_pre = __shfl_sync(mask, x_r_pre, 0, 16);
		} else if (core_kernel == 32) {
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 16);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 8);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 4);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
			x_r_pre = __shfl_sync(mask, x_r_pre, 0);
		}

		x_r_pre -= value[nnz_index];

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
			parameter_cp[order_index][index[nnz_index * order + order_index]
					* core_kernel + lane_id] -= learn_rate_a
					* (x_r_pre * gs[order_index] + lambda_a * a[order_index]);

		}
	}
}

void Update_Parameter_SVD(const int order, int *dimen, const int core_kernel,
		const int core_dimen, type_of_data **parameter_a_host_to_device,
		type_of_data **parameter_b_host_to_device,
		type_of_data **parameter_cp_host_to_device) {

	for (int i = 0; i < order; i++) {

		cusolverDnHandle_t cusolverH = NULL;
		cublasHandle_t cublasH = NULL;

		const int batchSize = 1;
		const int m = dimen[i];
		const int n = core_kernel;
		const int rank = core_dimen;
		const long long int strideA = static_cast<long long int>(m * n);
		const long long int strideS = rank;
		const long long int strideU = static_cast<long long int>(m * rank);
		const long long int strideV = static_cast<long long int>(n * rank);
		const type_of_data alpha = 1.0;
		const type_of_data beta = 0.0;

		type_of_data *d_A = nullptr;
		type_of_data *d_S = nullptr;
		type_of_data *d_U = nullptr;
		type_of_data *d_V = nullptr;

		int *d_info = nullptr;

		int lwork = 0;
		type_of_data *d_work = nullptr;

		std::vector<int> info(batchSize, 0);
		std::vector<double> RnrmF(batchSize, 0);

		const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

		cusolverDnCreate(&cusolverH);
		cublasCreate(&cublasH);

		cudaMalloc(reinterpret_cast<void**>(&d_A),
				sizeof(type_of_data) * strideA);
		cudaMalloc(reinterpret_cast<void**>(&d_S), sizeof(type_of_data) * n);
		cudaMalloc(reinterpret_cast<void**>(&d_U),
				sizeof(type_of_data) * strideU);
		cudaMalloc(reinterpret_cast<void**>(&d_V),
				sizeof(type_of_data) * strideV);
		cudaMalloc(reinterpret_cast<void**>(&d_info),
				sizeof(int) * info.size());

		cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha,
				parameter_cp_host_to_device[i], n, &beta, nullptr, m, d_A, m);

		cusolverDnSgesvdaStridedBatched_bufferSize(cusolverH, jobz, rank, m, n,
				d_A, m, strideA, d_S, strideS, d_U, m, strideU, d_V, n, strideV,
				&lwork, batchSize);

		cudaMalloc(reinterpret_cast<void**>(&d_work),
				sizeof(type_of_data) * lwork);

		cusolverDnSgesvdaStridedBatched(cusolverH, jobz, rank, m, n, d_A, m,
				strideA, d_S, strideS, d_U, m, strideU, d_V, n, strideV, d_work,
				lwork, d_info, RnrmF.data(), batchSize);

		cublasSdgmm(cublasH, CUBLAS_SIDE_RIGHT, n, rank, d_V, n, d_S, 1, d_V,
				n);
		cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, rank, m, &alpha, d_U, m,
				&beta, d_U, rank, parameter_a_host_to_device[i], rank);
		cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, rank, n, &alpha, d_V, n,
				&beta, d_V, rank, parameter_b_host_to_device[i], rank);

		cudaFree(d_A);
		cudaFree(d_S);
		cudaFree(d_U);
		cudaFree(d_V);
		cudaFree(d_info);
		cudaFree(d_work);

		cusolverDnDestroy(cusolverH);
		cublasDestroy(cublasH);

	}

}

void Update_Parameter_CP(const int order, const int core_kernel,
		type_of_data **parameter_a_device, const int nnz_train,
		type_of_data **value_train_device, int **index_train_device,
		type_of_data learn_rate_a, type_of_data lambda_a) {

	int data_per_part = nnz_train / data_part + 1;

	for (int i = 0; i < data_part - 1; i++) {
		Update_Parameter_CPD_SGD <<<grid_size, block_size>>>(order, core_kernel,
				parameter_a_device, data_per_part, value_train_device[i],
				index_train_device[i], learn_rate_a, lambda_a);
		cudaDeviceSynchronize();
	}
	Update_Parameter_CPD_SGD<<<grid_size, block_size>>>(order, core_kernel,
			parameter_a_device, nnz_train - (data_part - 1) * data_per_part,
			value_train_device[data_part - 1],
			index_train_device[data_part - 1], learn_rate_a, lambda_a);
	cudaDeviceSynchronize();

}

__global__ void RMSE_AND_MAE_CPD(const int order, const int core_kernel,
		type_of_data **parameter_cp, const int nnz, const type_of_data *value,
		const int *index, type_of_data *rmse, type_of_data *mae) {

	int core = core_kernel;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data x_r_pre = 1.0;

		for (int order_index = 0; order_index < order; order_index++) {
			x_r_pre *= parameter_cp[order_index][index[nnz_index * order
					+ order_index] * core_kernel + lane_id];
		}

		if (core_kernel == 4) {
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
			x_r_pre = __shfl_sync(mask, x_r_pre, 0, 4);
		} else if (core_kernel == 8) {
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 4);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
			x_r_pre = __shfl_sync(mask, x_r_pre, 0, 8);
		} else if (core_kernel == 16) {
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 8);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 4);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
			x_r_pre = __shfl_sync(mask, x_r_pre, 0, 16);
		} else if (core_kernel == 32) {
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 16);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 8);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 4);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
			x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
			x_r_pre = __shfl_sync(mask, x_r_pre, 0);
		}

		x_r_pre -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], x_r_pre * x_r_pre);
			atomicAdd(&mae[nnz_index % error_size], abs(x_r_pre));
		}
	}

}

__global__ void RMSE_AND_MAE_Tucker(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		const type_of_data *value, const int *index, type_of_data *rmse,
		type_of_data *mae) {

	int core = core_dimen;
	int worker = block_size / core;
	int lane_id = threadIdx.x % core;
	int local_id = threadIdx.x / core;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {
		type_of_data p_a_gs = 0.0;
		type_of_data gs = 0.0;

		for (int core_kernel_index = 0; core_kernel_index < core_kernel;
				core_kernel_index++) {
			type_of_data gs_temp = parameter_b[0][core_kernel_index * core_dimen
					+ lane_id];

			for (int inner_order_index = 0; inner_order_index < order;
					inner_order_index++) {
				if (inner_order_index != 0) {
					type_of_data temp =
							parameter_a[inner_order_index][index[nnz_index
									* order + inner_order_index] * core_dimen
									+ lane_id]
									* parameter_b[inner_order_index][core_kernel_index
											* core_dimen + lane_id];
					if (core_dimen == 4) {
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 4);
					} else if (core_dimen == 8) {
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 8);
					} else if (core_dimen == 16) {
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0, 16);
					} else if (core_dimen == 32) {
						temp += __shfl_down_sync(mask, temp, 16);
						temp += __shfl_down_sync(mask, temp, 8);
						temp += __shfl_down_sync(mask, temp, 4);
						temp += __shfl_down_sync(mask, temp, 2);
						temp += __shfl_down_sync(mask, temp, 1);
						temp = __shfl_sync(mask, temp, 0);
					}

					gs_temp *= temp;

				}
			}
			gs += gs_temp;
		}

		p_a_gs = parameter_a[0][index[nnz_index * order] * core_dimen + lane_id]
				* gs;

		if (core_dimen == 4) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 4);
		} else if (core_dimen == 8) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 8);
		} else if (core_dimen == 16) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0, 16);
		} else if (core_dimen == 32) {
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 16);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 8);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 4);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 2);
			p_a_gs += __shfl_down_sync(mask, p_a_gs, 1);
			p_a_gs = __shfl_sync(mask, p_a_gs, 0);
		}

		p_a_gs -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], p_a_gs * p_a_gs);
			atomicAdd(&mae[nnz_index % error_size], abs(p_a_gs));
		}

	}

}

void GET_RMSE_AND_MAE_Tucker(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data **value, int **index, type_of_data *rmse,
		type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse, error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	int data_per_part = nnz / data_part + 1;
	for (int i = 0; i < data_part - 1; i++) {
		RMSE_AND_MAE_Tucker
				<<<data_per_part / block_size + 1, block_size>>>(order,
				core_kernel, core_dimen, parameter_a, parameter_b,
				data_per_part, value[i], index[i], errors_rmse, errors_mae);
		cudaDeviceSynchronize();
	}
	RMSE_AND_MAE_Tucker<<<data_per_part / block_size + 1, block_size>>>(order,
			core_kernel, core_dimen, parameter_a, parameter_b,
			nnz - (data_part - 1) * data_per_part, value[data_part - 1],
			index[data_part - 1], errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	free(rmse_sum);
	free(mae_sum);

}

void GET_RMSE_AND_MAE_Tucker(const int order, const int core_kernel,
		const int core_dimen,
		type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		type_of_data *value, int *index,
		type_of_data *rmse,
		type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse, error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	RMSE_AND_MAE_Tucker<<<nnz / block_size + 1, block_size,
	(order * block_size + order * core_kernel * core_dimen)
	* sizeof(type_of_data)>>>(order, core_kernel, core_dimen, parameter_a,
			parameter_b, nnz, value, index, errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	free(rmse_sum);
	free(mae_sum);

}

void GET_RMSE_AND_MAE_CPD(const int order, const int core_kernel,
		type_of_data **parameter_a, const int nnz, type_of_data **value,
		int **index, type_of_data *rmse, type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse, error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	int data_per_part = nnz / data_part + 1;
	for (int i = 0; i < data_part - 1; i++) {
		RMSE_AND_MAE_CPD<<<data_per_part / block_size + 1, block_size>>>(order,
				core_kernel, parameter_a, data_per_part, value[i], index[i],
				errors_rmse, errors_mae);
		cudaDeviceSynchronize();
	}
	RMSE_AND_MAE_CPD<<<data_per_part / block_size + 1, block_size>>>(order,
			core_kernel, parameter_a, nnz - (data_part - 1) * data_per_part,
			value[data_part - 1], index[data_part - 1], errors_rmse,
			errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	free(rmse_sum);
	free(mae_sum);

}

void GET_RMSE_AND_MAE_CPD(const int order, const int core_kernel,
		type_of_data **parameter_a, const int nnz, type_of_data *value,
		int *index, type_of_data *rmse, type_of_data *mae) {

	type_of_data *errors_rmse;
	type_of_data *errors_mae;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cublasHandle_t handle_mae;
	cublasCreate(&handle_mae);
	cudaMalloc((void**) &errors_rmse, error_size * sizeof(type_of_data));
	cudaMalloc((void**) &errors_mae, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));
	cudaMemset(errors_mae, 0, error_size * sizeof(type_of_data));

	RMSE_AND_MAE_CPD<<<nnz / block_size + 1, block_size>>>(order, core_kernel,
			parameter_a, nnz, value, index, errors_rmse, errors_mae);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));
	type_of_data *mae_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();
	cublasSasum(handle_mae, error_size, errors_mae, 1, mae_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	*mae = (*mae_sum) / nnz;
	cudaFree(errors_rmse);
	cudaFree(errors_mae);
	cublasDestroy(handle_rmse);
	cublasDestroy(handle_mae);
	free(rmse_sum);
	free(mae_sum);

}
