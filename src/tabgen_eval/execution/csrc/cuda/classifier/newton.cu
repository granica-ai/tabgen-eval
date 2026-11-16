/** @file
 * @brief Header-free NVRTC primitives for dense FP32/FP64 logistic Newton.
 * Kernels launch 256 threads, mask tail indices and use column-major designs.
 * Vectors match selected precision; fixed-owner diagnostic reductions use double.
 * Every output is caller-owned and disjoint from read-only inputs, except the
 * documented in-place gradient, ridge-diagonal and damping operations.
 */
#ifdef LTM_NEWTON_FLOAT
using Real = float;
__device__ Real real_exp(Real x) {
  return expf(x);
}
__device__ Real real_log1p(Real x) {
  return log1pf(x);
}
__device__ Real real_expm1(Real x) {
  return expm1f(x);
}
__device__ Real real_sqrt(Real x) {
  return sqrtf(x);
}
#else
using Real = double;
__device__ Real real_exp(Real x) {
  return exp(x);
}
__device__ Real real_log1p(Real x) {
  return log1p(x);
}
__device__ Real real_expm1(Real x) {
  return expm1(x);
}
__device__ Real real_sqrt(Real x) {
  return sqrt(x);
}
#endif

__device__ Real absolute(Real x) {
  return x < 0 ? -x : x;
}
__device__ Real softplus(Real x) {
  return (x > 0 ? x : Real(0)) + real_log1p(real_exp(-absolute(x)));
}
// Every reduction has one deterministic owner; diagnostic accumulation uses
// double without promoting stored FP32 designs, derivatives or linear solves.
__device__ double block_sum(double value) {
  __shared__ double shared[256];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int shift = 128; shift; shift /= 2) {
    if (threadIdx.x < shift)
      shared[threadIdx.x] += shared[threadIdx.x + shift];
    __syncthreads();
  }
  return shared[0];
}
extern "C" __global__ void newton_logistic(const Real* logits, const Real* labels, Real* residual,
                                           Real* roots, double* parts, int n) {
  int row = blockIdx.x * 256 + threadIdx.x;
  Real loss = 0;
  if (row < n) {
    Real z = logits[row], y = labels[row];
    Real e = real_exp(-absolute(z)), denominator = 1 + e;
    Real probability = z >= 0 ? 1 / denominator : e / denominator;
    Real r = y == 0 ? probability : (z >= 0 ? -e / denominator : -1 / denominator);
    residual[row] = r / n;
    roots[row] = real_sqrt(e / (denominator * denominator * n));
    loss = softplus(y == 1 ? -z : z);
  }
  double sum = block_sum(double(loss));
  if (threadIdx.x == 0)
    parts[blockIdx.x] = sum / n;
}
extern "C" __global__ void newton_gradient(Real* gradient, const Real* weights, Real ridge,
                                           double* parts, int d) {
  int f = blockIdx.x * 256 + threadIdx.x;
  Real value = 0;
  if (f < d) {
    value = gradient[f] + (f < d - 1 ? ridge * weights[f] : Real(0));
    gradient[f] = value;
  }
  double sum = block_sum(double(value) * double(value));
  if (threadIdx.x == 0)
    parts[blockIdx.x] = sum;
}
extern "C" __global__ void newton_reduce(const double* parts, double* result, int count) {
  double value = 0;
  for (int i = threadIdx.x; i < count; i += 256)
    value += parts[i];
  double sum = block_sum(value);
  if (threadIdx.x == 0)
    *result = sum;
}
extern "C" __global__ void newton_weight_design(const Real* design, const Real* roots,
                                                Real* weighted, long long count, int n) {
  long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i < count)
    weighted[i] = design[i] * roots[i % n];
}
extern "C" __global__ void newton_ridge_scale(Real* matrix, Real ridge, double* result, int d) {
  __shared__ double shared[256];
  double maximum = 1;
  for (int f = threadIdx.x; f < d; f += 256) {
    Real value = matrix[(long long)f * (d + 1)] + (f < d - 1 ? ridge : Real(0));
    matrix[(long long)f * (d + 1)] = value;
    double magnitude = double(absolute(value));
    if (!isfinite(magnitude))
      maximum = 1.0 / 0.0;
    else if (magnitude > maximum)
      maximum = magnitude;
  }
  shared[threadIdx.x] = maximum;
  __syncthreads();
  for (int shift = 128; shift; shift /= 2) {
    if (threadIdx.x < shift && shared[threadIdx.x + shift] > shared[threadIdx.x])
      shared[threadIdx.x] = shared[threadIdx.x + shift];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    *result = shared[0];
}
extern "C" __global__ void newton_damping(Real* factor, Real damping, int d) {
  int f = blockIdx.x * 256 + threadIdx.x;
  if (f < d)
    factor[(long long)f * (d + 1)] += damping;
}
extern "C" __global__ void newton_dot(const Real* a, const Real* b, double* parts, int d) {
  int f = blockIdx.x * 256 + threadIdx.x;
  double value = f < d ? double(a[f]) * double(b[f]) : 0;
  double sum = block_sum(value);
  if (threadIdx.x == 0)
    parts[blockIdx.x] = sum;
}
extern "C" __global__ void newton_candidate(const Real* weights, const Real* step, Real* candidate,
                                            Real* change, Real rate, double ridge, double* parts,
                                            int d) {
  int f = blockIdx.x * 256 + threadIdx.x;
  double value = 0;
  if (f < d) {
    Real next = weights[f] - rate * step[f];
    Real delta = next - weights[f];
    candidate[f] = next;
    change[f] = delta;
    // Use the rounded change represented by the candidate, including in X*delta.
    if (f < d - 1)
      value = ridge * (double(weights[f]) * double(delta) + 0.5 * double(delta) * double(delta));
  }
  double sum = block_sum(value);
  if (threadIdx.x == 0)
    parts[blockIdx.x] = sum;
}
extern "C" __global__ void newton_loss_difference(const Real* logits, const Real* delta,
                                                  const Real* labels, double* parts, int n) {
  int row = blockIdx.x * 256 + threadIdx.x;
  Real difference = 0;
  if (row < n) {
    Real z = labels[row] == 1 ? -logits[row] : logits[row];
    Real change = labels[row] == 1 ? -delta[row] : delta[row];
    Real e = real_exp(-absolute(z));
    Real probability = z >= 0 ? 1 / (1 + e) : e / (1 + e);
    difference = absolute(change) <= 10 ? real_log1p(probability * real_expm1(change))
                                        : softplus(z + change) - softplus(z);
  }
  double sum = block_sum(double(difference));
  if (threadIdx.x == 0)
    parts[blockIdx.x] = sum / n;
}
