#include <vector>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <fstream>
#include <time.h>
#include "parameter.h"

__global__ void Random_Init(type_of_data *parameter_cp, int dimen, int rank_R,
type_of_data mean, type_of_data min_val,
type_of_data max_val, unsigned long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state;
	curand_init(seed, idx, 0, &state);
	type_of_data temp = max_val - min_val;
	for (int index = idx; index < dimen * rank_R;
			index += grid_size_init * block_size) {
		parameter_cp[index] = mean * (curand_uniform(&state) * temp + min_val);
	}
}

void Parameter_Initialization(type_of_data **parameter_cp,
type_of_data data_norm, int order, int *dimen, int rank_R,
type_of_data min_val, type_of_data max_val) {

	unsigned long seed = time(0);
	type_of_data mean = pow(data_norm / rank_R, 1.0 / order);
	for (int i = 0; i < order; i++) {
		Random_Init<<<grid_size_init, block_size>>>(parameter_cp[i], dimen[i],
				rank_R, mean, min_val, max_val, seed);
		cudaDeviceSynchronize();
	}
}

__global__ void Update_Parameter_CP_SGD(type_of_data **parameter_cp,
		const int nnz, const type_of_data *value, const int *index,
		const type_of_data learn_rate_a, const type_of_data lambda_a) {

	int worker = block_size / warp_size;
	int lane_id = threadIdx.x % warp_size;
	int local_id = threadIdx.x / warp_size;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	type_of_data f[rank_R_size];
	type_of_data a[order_size][rank_R_size];

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {

#pragma unroll
		for (int rank_R_index = 0; rank_R_index < rank_R_size; rank_R_index++) {
			f[rank_R_index] = 1.0f;
		}

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
			for (int rank_R_index = 0; rank_R_index < rank_R_size;
					rank_R_index++) {
				a[order_index][rank_R_index] =
						parameter_cp[order_index][index[nnz_index * order_size
								+ order_index] * rank_R_size * warp_size
								+ rank_R_index * warp_size + lane_id];
			}

#pragma unroll
			for (int rank_R_index = 0; rank_R_index < rank_R_size;
					rank_R_index++) {
				f[rank_R_index] *= a[order_index][rank_R_index];
			}

		}

		type_of_data x_r_pre = 0.0f;
#pragma unroll
		for (int rank_R_index = 0; rank_R_index < rank_R_size; rank_R_index++) {
			x_r_pre += f[rank_R_index];
		}

		x_r_pre += __shfl_down_sync(mask, x_r_pre, 16);
		x_r_pre += __shfl_down_sync(mask, x_r_pre, 8);
		x_r_pre += __shfl_down_sync(mask, x_r_pre, 4);
		x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
		x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
		x_r_pre = __shfl_sync(mask, x_r_pre, 0);

		x_r_pre -= value[nnz_index];

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
			for (int rank_R_index = 0; rank_R_index < rank_R_size;
					rank_R_index++) {
				parameter_cp[order_index][index[nnz_index * order_size
						+ order_index] * rank_R_size * warp_size
						+ rank_R_index * warp_size + lane_id] -= learn_rate_a
						* (x_r_pre * f[rank_R_index]
								/ a[order_index][rank_R_index]
								+ lambda_a * a[order_index][rank_R_index]);
			}
		}
	}
}

void Update_Parameter_SVD(const int order, int *dimen, const int rank_R,
		const int rank_J,
		type_of_data **parameter_cp_host_to_device,
		type_of_data ***parameter_a_device,
		type_of_data ***parameter_a_host_to_device,
		type_of_data ***parameter_b_device,
		type_of_data ***parameter_b_host_to_device) {

	*parameter_a_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);
	*parameter_b_host_to_device = (type_of_data**) malloc(
			sizeof(type_of_data*) * order);

	cudaMalloc((void**) &(*parameter_a_device), sizeof(type_of_data*) * order);
	cudaMalloc((void**) &(*parameter_b_device), sizeof(type_of_data*) * order);

	for (int i = 0; i < order; i++) {

		cusolverDnHandle_t cusolverH = NULL;
		cublasHandle_t cublasH = NULL;

		const int batchSize = 1;
		const int m = dimen[i];
		const int n = rank_R;
		const int rank = rank_J;
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
		cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha,
				parameter_cp_host_to_device[i], n, &beta, nullptr, m, d_A, m);

		cudaFree(parameter_cp_host_to_device[i]);

		cudaMalloc(reinterpret_cast<void**>(&d_S), sizeof(type_of_data) * n);
		cudaMalloc(reinterpret_cast<void**>(&d_U),
				sizeof(type_of_data) * strideU);
		cudaMalloc(reinterpret_cast<void**>(&d_V),
				sizeof(type_of_data) * strideV);
		cudaMalloc(reinterpret_cast<void**>(&d_info),
				sizeof(int) * info.size());

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

		(*parameter_a_host_to_device)[i] = d_U;
		(*parameter_b_host_to_device)[i] = d_V;

		cudaFree(d_A);
		cudaFree(d_S);
		cudaFree(d_info);
		cudaFree(d_work);

		cusolverDnDestroy(cusolverH);
		cublasDestroy(cublasH);

	}

	cudaMemcpy(*parameter_a_device, *parameter_a_host_to_device,
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);
	cudaMemcpy(*parameter_b_device, *parameter_b_host_to_device,
			sizeof(type_of_data*) * order, cudaMemcpyHostToDevice);

}

void Update_Parameter_CP(const int order, const int core_kernel,
type_of_data **parameter_a_device, const int nnz_train,
type_of_data *value_train_device, int *index_train_device,
type_of_data learn_rate_a, type_of_data lambda_a) {

	Update_Parameter_CP_SGD <<<grid_size, block_size>>>(parameter_a_device,
			nnz_train, value_train_device, index_train_device, learn_rate_a,
			lambda_a);
	cudaDeviceSynchronize();

}

__global__ void RMSE_AND_MAE_CP(type_of_data **parameter_cp, const int nnz,
		const type_of_data *value, const int *index, type_of_data *rmse) {

	int worker = block_size / warp_size;
	int lane_id = threadIdx.x % warp_size;
	int local_id = threadIdx.x / warp_size;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	type_of_data f[rank_R_size];

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {

#pragma unroll
		for (int rank_R_index = 0; rank_R_index < rank_R_size; rank_R_index++) {
			f[rank_R_index] = 1.0f;
		}

#pragma unroll
		for (int order_index = 0; order_index < order_size; order_index++) {
#pragma unroll
			for (int rank_R_index = 0; rank_R_index < rank_R_size;
					rank_R_index++) {
				f[rank_R_index] *= parameter_cp[order_index][index[nnz_index
						* order_size + order_index] * rank_R_size * warp_size
						+ rank_R_index * warp_size + lane_id];
			}

		}

		type_of_data x_r_pre = 0.0;
#pragma unroll
		for (int rank_R_index = 0; rank_R_index < rank_R_size; rank_R_index++) {
			x_r_pre += f[rank_R_index];
		}

		x_r_pre += __shfl_down_sync(mask, x_r_pre, 16);
		x_r_pre += __shfl_down_sync(mask, x_r_pre, 8);
		x_r_pre += __shfl_down_sync(mask, x_r_pre, 4);
		x_r_pre += __shfl_down_sync(mask, x_r_pre, 2);
		x_r_pre += __shfl_down_sync(mask, x_r_pre, 1);
		x_r_pre = __shfl_sync(mask, x_r_pre, 0);

		x_r_pre -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], x_r_pre * x_r_pre);
		}

	}
}

__global__ void RMSE_AND_MAE_Tucker(const int rank_R, const int rank_J,
type_of_data **parameter_a, type_of_data **parameter_b, const int nnz,
		const type_of_data *value, const int *index, type_of_data *rmse) {

	int worker = block_size / warp_size;
	int lane_id = threadIdx.x % warp_size;
	int local_id = threadIdx.x / warp_size;
	int worker_id = worker * blockIdx.x + local_id;
	int workers = worker * gridDim.x;

	for (int nnz_index = worker_id; nnz_index < nnz; nnz_index += workers) {

		type_of_data x_r_pre = 0.0f;
#pragma unroll
		for (int rank_R_index = 0; rank_R_index < rank_R; rank_R_index++) {

			type_of_data x_r = 1.0f;
#pragma unroll
			for (int order_index = 0; order_index < order_size; order_index++) {

				type_of_data temp = 0.0f;
#pragma unroll
				for (int rank_J_index = 0; rank_J_index < rank_J_size;
						rank_J_index++) {
					temp += parameter_a[order_index][index[nnz_index
							* order_size + order_index] * rank_J
							+ rank_J_index * warp_size + lane_id]
							* parameter_b[order_index][rank_R_index * rank_J
									+ rank_J_index * warp_size + lane_id];
				}

				temp += __shfl_down_sync(mask, temp, 16);
				temp += __shfl_down_sync(mask, temp, 8);
				temp += __shfl_down_sync(mask, temp, 4);
				temp += __shfl_down_sync(mask, temp, 2);
				temp += __shfl_down_sync(mask, temp, 1);
				temp = __shfl_sync(mask, temp, 0);

				x_r *= temp;
			}
			x_r_pre += x_r;

		}

		x_r_pre -= value[nnz_index];

		if (lane_id == 0) {
			atomicAdd(&rmse[nnz_index % error_size], x_r_pre * x_r_pre);
		}

	}

}

void GET_RMSE_AND_MAE_CP(const int order, const int rank_R,
type_of_data **parameter_a, const int nnz, type_of_data *value, int *index,
type_of_data *rmse) {

	type_of_data *errors_rmse;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cudaMalloc((void**) &errors_rmse, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));

	RMSE_AND_MAE_CP <<<nnz / block_size + 1, block_size>>>(parameter_a, nnz,
			value, index, errors_rmse);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	cudaFree(errors_rmse);
	cublasDestroy(handle_rmse);
	free(rmse_sum);

}

void GET_RMSE_AND_MAE_Tucker(const int order, const int rank_R,
		const int rank_J, type_of_data **parameter_a,
		type_of_data **parameter_b, const int nnz, type_of_data *value,
		int *index, type_of_data *rmse) {

	type_of_data *errors_rmse;
	cublasHandle_t handle_rmse;
	cublasCreate(&handle_rmse);
	cudaMalloc((void**) &errors_rmse, error_size * sizeof(type_of_data));
	cudaMemset(errors_rmse, 0, error_size * sizeof(type_of_data));

	RMSE_AND_MAE_Tucker <<<nnz / block_size + 1, block_size>>>(
			rank_R, rank_J, parameter_a, parameter_b, nnz, value, index,
			errors_rmse);
	cudaDeviceSynchronize();

	type_of_data *rmse_sum = (type_of_data*) malloc(sizeof(type_of_data));

	cublasSasum(handle_rmse, error_size, errors_rmse, 1, rmse_sum);
	cudaDeviceSynchronize();

	*rmse = sqrt((*rmse_sum) / nnz);
	cudaFree(errors_rmse);
	cublasDestroy(handle_rmse);
	free(rmse_sum);

}

void Trans(const int order, int *dimen, const int rank_R, const int rank_J,
		type_of_data **parameter_a_host_to_device,
		type_of_data **parameter_b_host_to_device) {

	cublasHandle_t cublasH = NULL;
	cublasCreate(&cublasH);
	const type_of_data alpha = 1.0;
	const type_of_data beta = 0.0;

	for (int i = 0; i < order; i++) {
		type_of_data *temp_a;
		cudaMalloc((void**) &temp_a, sizeof(type_of_data) * dimen[i] * rank_J);
		type_of_data *temp_b;
		cudaMalloc((void**) &temp_b, sizeof(type_of_data) * rank_R * rank_J);
		cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, rank_J, dimen[i], &alpha,
				parameter_a_host_to_device[i], dimen[i], &beta,
				nullptr, rank_J, temp_a, rank_J);
		cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, rank_J, rank_R, &alpha,
				parameter_b_host_to_device[i], rank_R, &beta,
				nullptr, rank_J, temp_b, rank_J);
		cudaMemcpy(parameter_a_host_to_device[i], temp_a,
				sizeof(type_of_data) * dimen[i] * rank_J,
				cudaMemcpyDeviceToDevice);
		cudaMemcpy(parameter_b_host_to_device[i], temp_b,
				sizeof(type_of_data) * rank_R * rank_J,
				cudaMemcpyDeviceToDevice);
		cudaFree(temp_a);
		cudaFree(temp_b);
	}
	cublasDestroy(cublasH);
}
