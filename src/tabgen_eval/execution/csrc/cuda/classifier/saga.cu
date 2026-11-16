#ifdef LTM_SAGA_FLOAT
using Real = float;
#else
using Real = double;
#endif

// Full-training preparation uses bounded vectors and one owner per reduction.
extern "C" __global__ void prep_diagonal_numeric(const Real* numbers, Real* partial, int n, int pn,
                                                 int feature_blocks) {
  __shared__ Real sum[256];
  int tile = blockIdx.x / feature_blocks;
  int lane = threadIdx.x % 32, row_lane = threadIdx.x / 32,
      f = (blockIdx.x % feature_blocks) * 32 + lane;
  Real value = 0;
  if (f < pn)
    for (int row = tile * 4096 + row_lane; row < n && row < (tile + 1) * 4096; row += 8) {
      Real x = numbers[(long long)row * pn + f];
      value += x * x;
    }
  sum[threadIdx.x] = value;
  __syncthreads();
  if (row_lane == 0 && f < pn) {
    Real result = 0;
    for (int i = 0; i < 8; ++i)
      result += sum[i * 32 + lane];
    partial[(long long)tile * pn + f] = result;
  }
}
extern "C" __global__ void prep_diagonal_finish(const Real* partial, Real* diagonal, int n, int pn,
                                                int tiles, Real ridge) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f < pn) {
    Real value = 0;
    for (int tile = 0; tile < tiles; ++tile)
      value += partial[(long long)tile * pn + f];
    diagonal[f] = value * (Real(0.25) / n) + ridge;
  }
}
extern "C" __global__ void prep_diagonal_category(const long long* offsets, const Real* centers,
                                                  const Real* scales, Real* diagonal, int k, int pn,
                                                  Real ridge) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f < k) {
    Real count = Real(offsets[f + 1] - offsets[f]);
    // N is supplied in the final diagonal slot before this kernel.
    Real n = diagonal[pn + k];
    Real center = centers[f], scale = scales[f];
    diagonal[pn + f] =
        Real(0.25) * (count / n * (1 - 2 * center) + center * center) / (scale * scale) + ridge;
  }
}
extern "C" __global__ void prep_logits(Real* logits, const long long* identities,
                                       const Real* scaled, const Real* baseline, const Real* v,
                                       int n, int pc, int d) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    Real z = logits[row] - *baseline + v[d - 1];
    for (int c = 0; c < pc; ++c) {
      long long f = identities[(long long)row * pc + c];
      if (f >= 0)
        z += scaled[f];
    }
    logits[row] = z * (Real(0.25) / n);
  }
}
extern "C" __global__ void prep_adjoint(const long long* rows, const long long* offsets,
                                        const Real* values, const Real* total, const Real* centers,
                                        const Real* scales, const Real* v, Real* out, int k, int pn,
                                        Real ridge) {
  int f = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  Real value = 0;
  if (f < k)
    for (long long i = offsets[f] + lane; i < offsets[f + 1]; i += 32)
      value += values[rows[i]];
  for (int s = 16; s; s >>= 1)
    value += __shfl_down_sync(0xffffffffU, value, s);
  if (f < k && lane == 0)
    out[pn + f] = (value - centers[f] * (*total)) / scales[f] + ridge * v[pn + f];
}
extern "C" __global__ void prep_finish_adjoint(Real* out, const Real* v, const Real* total, int pn,
                                               int d, Real ridge) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f < pn)
    out[f] += ridge * v[f];
  if (f == 0)
    out[d - 1] = *total;
}
// Isolated FP64 exact-feature minibatch kernels. NVRTC compiles this file.
// No floating atomics: every output reduction has one deterministic owner.

extern "C" __global__ void gather_numbers(const Real* numbers, const int* rows, Real* batch, int b,
                                          int pn) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < b * pn)
    batch[k] = numbers[(long long)rows[k / pn] * pn + k % pn];
}

extern "C" __global__ void scale_categories(const Real* w, const Real* center, const Real* scale,
                                            Real* scaled, Real* partial, int k, int pn) {
  __shared__ Real sum[256];
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  Real value = f < k ? w[pn + f] / scale[f] : 0;
  if (f < k)
    scaled[f] = value;
  sum[threadIdx.x] = f < k ? center[f] * value : 0;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s)
      sum[threadIdx.x] += sum[threadIdx.x + s];
    __syncthreads();
  }
  if (!threadIdx.x)
    partial[blockIdx.x] = sum[0];
}

extern "C" __global__ void reduce_sum(const Real* values, Real* out, int n) {
  __shared__ Real sum[256];
  Real v = 0;
  for (int i = threadIdx.x; i < n; i += 256)
    v += values[i];
  sum[threadIdx.x] = v;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s)
      sum[threadIdx.x] += sum[threadIdx.x + s];
    __syncthreads();
  }
  if (!threadIdx.x)
    *out = sum[0];
}

extern "C" __global__ void logistic_delta(const Real* numeric_logits, const long long* identities,
                                          const int* rows, const Real* scaled, const Real* baseline,
                                          const Real* w, const Real* labels, Real* table,
                                          Real* delta, Real* partial, int b, int pc, int d) {
  __shared__ Real sum[256];
  int i = blockIdx.x * 256 + threadIdx.x;
  Real change = 0;
  if (i < b) {
    int row = rows[i];
    Real z = numeric_logits[i] - *baseline + w[d - 1];
    for (int c = 0; c < pc; ++c) {
      long long id = identities[(long long)row * pc + c];
      if (id >= 0)
        z += scaled[id];
    }
    Real e = exp(-fabs(z)), p;
    if (labels[row] == 0)
      p = z >= 0 ? 1 / (1 + e) : e / (1 + e);
    else
      p = z >= 0 ? -e / (1 + e) : -1 / (1 + e);
    change = (p - table[row]) / b;
    table[row] = p;
    delta[i] = change;
  }
  sum[threadIdx.x] = change;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s)
      sum[threadIdx.x] += sum[threadIdx.x + s];
    __syncthreads();
  }
  if (!threadIdx.x)
    partial[blockIdx.x] = sum[0];
}

extern "C" __global__ void sort_categories(const long long* identities, const int* rows,
                                           unsigned long long* sorted_keys, int* sorted_values,
                                           int b, int pc, int sort_n) {
  extern __shared__ unsigned char storage[];
  auto* keys = reinterpret_cast<unsigned long long*>(storage);
  auto* values = reinterpret_cast<int*>(keys + sort_n);
  int c = blockIdx.x;
  for (int j = threadIdx.x; j < sort_n; j += blockDim.x) {
    long long id = j < b ? identities[(long long)rows[j] * pc + c] : -1;
    keys[j] = id >= 0
                  ? (static_cast<unsigned long long>(id) << 32) | static_cast<unsigned int>(rows[j])
                  : ~0ULL;
    values[j] = j;
  }
  __syncthreads();
  for (int width = 2; width <= sort_n; width <<= 1) {
    for (int step = width >> 1; step; step >>= 1) {
      for (int j = threadIdx.x; j < sort_n; j += blockDim.x) {
        int other = j ^ step;
        if (other > j) {
          bool ascending = (j & width) == 0;
          if ((keys[j] > keys[other]) == ascending) {
            unsigned long long key = keys[j];
            keys[j] = keys[other];
            keys[other] = key;
            int value = values[j];
            values[j] = values[other];
            values[other] = value;
          }
        }
      }
      __syncthreads();
    }
  }
  for (int j = threadIdx.x; j < sort_n; j += blockDim.x) {
    int loc = c * sort_n + j;
    sorted_keys[loc] = keys[j];
    sorted_values[loc] = values[j];
  }
}

extern "C" __global__ void category_bounds(const unsigned long long* keys, int* begin, int* end,
                                           int total, int sort_n) {
  int loc = blockIdx.x * blockDim.x + threadIdx.x;
  if (loc >= total)
    return;
  unsigned int f = keys[loc] >> 32;
  if (f == 0xffffffffU)
    return;
  bool first = loc % sort_n == 0 || static_cast<unsigned int>(keys[loc - 1] >> 32) != f;
  bool last = (loc + 1) % sort_n == 0 || static_cast<unsigned int>(keys[loc + 1] >> 32) != f;
  if (first)
    begin[f] = loc;
  if (last)
    end[f] = loc + 1;
}

extern "C" __global__ void category_adjoint(const int* values, const int* begin, const int* end,
                                            const Real* delta, const Real* total,
                                            const Real* center, const Real* scale, Real* correction,
                                            int k, int pn) {
  int f = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  Real v = 0;
  if (f < k) {
    int lo = begin[f], hi = end[f];
    if (lo >= 0)
      for (int j = lo + lane; j < hi; j += 32)
        v += delta[values[j]];
  }
  for (int step = 16; step; step >>= 1)
    v += __shfl_down_sync(0xffffffffU, v, step);
  if (f < k && lane == 0)
    correction[pn + f] = (v - center[f] * (*total)) / scale[f];
}

extern "C" __global__ void gradient_average(const Real* correction, const Real* total,
                                            Real* average, const Real* w, Real* gradient, int d,
                                            Real ridge, Real fraction) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= d)
    return;
  Real c = f == d - 1 ? *total : correction[f];
  gradient[f] = average[f] + c + (f == d - 1 ? 0 : ridge * w[f]);
  average[f] += fraction * c;
}

extern "C" __global__ void scale_vector(const Real* x, const Real* factor, Real* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = x[i] * factor[i];
}

extern "C" __global__ void update_weights(Real* w, const Real* step, const Real* inverse_root,
                                          Real* averaged_sum, int d, Real rate, int averaged,
                                          int preconditioned) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= d)
    return;
  if (averaged)
    averaged_sum[f] += w[f];
  w[f] -= rate * step[f] * (preconditioned ? inverse_root[f] : 1);
}

// A single first-update reduction captures the online unpreconditioned g0.
extern "C" __global__ void gradient_initial(const Real* gradient, double* out, int d) {
  __shared__ double values[256];
  double maximum = 0;
  for (int i = threadIdx.x; i < d; i += 256) {
    double value = double(gradient[i]);
    maximum =
        isfinite(value) ? fmax(maximum, fabs(value)) : __longlong_as_double(0x7ff0000000000000LL);
  }
  values[threadIdx.x] = maximum;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s)
      values[threadIdx.x] = fmax(values[threadIdx.x], values[threadIdx.x + s]);
    __syncthreads();
  }
  if (!threadIdx.x)
    *out = values[0];
}

extern "C" __global__ void epoch_change(const Real* w, Real* previous, const Real* gradient,
                                        double* partial, int d) {
  __shared__ double maximum[256], change[256], finite[256], grad[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double a = 0, c = 0, ok = 1, g = 0;
  if (i < d) {
    ok = isfinite(w[i]) && isfinite(previous[i]) && isfinite(gradient[i]);
    g = isfinite(gradient[i]) ? fabs(double(gradient[i]))
                              : __longlong_as_double(0x7ff0000000000000LL);
    if (i < d - 1) {
      a = fabs(w[i]);
      c = fabs(w[i] - previous[i]);
      ok = ok && isfinite(c);
    } else {
      partial[gridDim.x * 4] = fabs(double(w[i]) - double(previous[i]));
      ok = ok && isfinite(partial[gridDim.x * 4]);
    }
    previous[i] = w[i];
  }
  maximum[threadIdx.x] = a;
  change[threadIdx.x] = c;
  finite[threadIdx.x] = ok;
  grad[threadIdx.x] = g;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s) {
      maximum[threadIdx.x] = fmax(maximum[threadIdx.x], maximum[threadIdx.x + s]);
      change[threadIdx.x] = fmax(change[threadIdx.x], change[threadIdx.x + s]);
      finite[threadIdx.x] = fmin(finite[threadIdx.x], finite[threadIdx.x + s]);
      grad[threadIdx.x] = fmax(grad[threadIdx.x], grad[threadIdx.x + s]);
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    partial[blockIdx.x * 4] = maximum[0];
    partial[blockIdx.x * 4 + 1] = change[0];
    partial[blockIdx.x * 4 + 2] = finite[0];
    partial[blockIdx.x * 4 + 3] = grad[0];
  }
}

extern "C" __global__ void epoch_reduce(const double* partial, const double* initial, double* out,
                                        int count) {
  __shared__ double maximum[256], change[256], finite[256], grad[256];
  double a = 0, c = 0, ok = 1, g = 0;
  for (int i = threadIdx.x; i < count; i += 256) {
    a = fmax(a, partial[i * 4]);
    c = fmax(c, partial[i * 4 + 1]);
    ok = fmin(ok, partial[i * 4 + 2]);
    g = fmax(g, partial[i * 4 + 3]);
  }
  maximum[threadIdx.x] = a;
  change[threadIdx.x] = c;
  finite[threadIdx.x] = ok;
  grad[threadIdx.x] = g;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (threadIdx.x < s) {
      maximum[threadIdx.x] = fmax(maximum[threadIdx.x], maximum[threadIdx.x + s]);
      change[threadIdx.x] = fmax(change[threadIdx.x], change[threadIdx.x + s]);
      finite[threadIdx.x] = fmin(finite[threadIdx.x], finite[threadIdx.x + s]);
      grad[threadIdx.x] = fmax(grad[threadIdx.x], grad[threadIdx.x + s]);
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    out[0] = maximum[0];
    out[1] = change[0];
    out[2] = finite[0];
    out[3] = partial[count * 4];
    out[4] = grad[0];
    out[5] = *initial;
  }
}
