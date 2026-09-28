// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_NO_MULTITHREADING 1
#endif

#ifndef POCKETFFT_CACHE_SIZE
#define POCKETFFT_CACHE_SIZE 64
#endif

#if !defined(_WIN32) && !defined(POCKETFFT_USE_POSIX_MEMALIGN)
#define POCKETFFT_USE_POSIX_MEMALIGN 1
#endif

#include "pocketfft_hdronly.h"

#include <complex>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(_WIN32)
#define POCKETFFT_EXPORT extern "C" __declspec(dllexport)
#else
#define POCKETFFT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

typedef struct {
  double r;
  double i;
} kiss_fft_cpx;

struct kiss_fft_state {
  int nfft;
  int inverse;
};
typedef struct kiss_fft_state* kiss_fft_cfg;

struct kiss_fftr_state {
  int nfft;
  int inverse;
};
typedef struct kiss_fftr_state* kiss_fftr_cfg;

struct kiss_fftnd_state {
  int inverse;
  int ndims;
  int dims[1];
};
typedef struct kiss_fftnd_state* kiss_fftnd_cfg;

namespace {

inline const pocketfft::detail::pocketfft_c<double>& get_c_plan(size_t n) {
  thread_local size_t last_n = 0;
  thread_local std::shared_ptr<pocketfft::detail::pocketfft_c<double>>
      last_plan;
  if (last_n != n || !last_plan) {
    last_plan =
        pocketfft::detail::get_plan<pocketfft::detail::pocketfft_c<double>>(n);
    last_n = n;
  }
  return *last_plan;
}

inline const pocketfft::detail::pocketfft_r<double>& get_r_plan(size_t n) {
  thread_local size_t last_n = 0;
  thread_local std::shared_ptr<pocketfft::detail::pocketfft_r<double>>
      last_plan;
  if (last_n != n || !last_plan) {
    last_plan =
        pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<double>>(n);
    last_n = n;
  }
  return *last_plan;
}

}  // namespace

POCKETFFT_EXPORT kiss_fft_cfg kiss_fft_alloc(
    int nfft,
    int inverse_fft,
    void* mem,
    size_t* lenmem) {
  if (nfft <= 0) {
    return nullptr;
  }
  try {
    (void)get_c_plan(static_cast<size_t>(nfft));
  } catch (...) {
    return nullptr;
  }

  const size_t memneeded = sizeof(struct kiss_fft_state);
  kiss_fft_cfg st = nullptr;
  if (lenmem == nullptr) {
    st = static_cast<kiss_fft_cfg>(std::malloc(memneeded));
  } else {
    if (mem != nullptr && *lenmem >= memneeded) {
      st = static_cast<kiss_fft_cfg>(mem);
    }
    *lenmem = memneeded;
  }
  if (st != nullptr) {
    st->nfft = nfft;
    st->inverse = inverse_fft ? 1 : 0;
  }
  return st;
}

POCKETFFT_EXPORT void kiss_fft_stride(
    kiss_fft_cfg cfg,
    const kiss_fft_cpx* fin,
    kiss_fft_cpx* fout,
    int fin_stride) {
  if (cfg == nullptr || fin == nullptr || fout == nullptr || cfg->nfft <= 0 ||
      fin_stride <= 0) {
    return;
  }
  try {
    const size_t n = static_cast<size_t>(cfg->nfft);
    const size_t stride = static_cast<size_t>(fin_stride);
    const auto& plan = get_c_plan(n);
    const bool forward = (cfg->inverse == 0);

    if (stride == 1) {
      if (fin != fout) {
        std::memmove(fout, fin, n * sizeof(kiss_fft_cpx));
      }
      plan.exec(
          reinterpret_cast<pocketfft::detail::cmplx<double>*>(fout),
          1.0,
          forward);
    } else if (fin != fout) {
      for (size_t i = 0; i < n; ++i) {
        fout[i] = fin[i * stride];
      }
      plan.exec(
          reinterpret_cast<pocketfft::detail::cmplx<double>*>(fout),
          1.0,
          forward);
    } else {
      pocketfft::detail::arr<pocketfft::detail::cmplx<double>> tmp(n);
      for (size_t i = 0; i < n; ++i) {
        tmp[i].r = fin[i * stride].r;
        tmp[i].i = fin[i * stride].i;
      }
      plan.exec(tmp.data(), 1.0, forward);
      std::memcpy(fout, tmp.data(), n * sizeof(kiss_fft_cpx));
    }
  } catch (...) {
  }
}

POCKETFFT_EXPORT void kiss_fft(
    kiss_fft_cfg cfg,
    const kiss_fft_cpx* fin,
    kiss_fft_cpx* fout) {
  kiss_fft_stride(cfg, fin, fout, 1);
}

POCKETFFT_EXPORT void kiss_fft_cleanup(void) {}

POCKETFFT_EXPORT int kiss_fft_next_fast_size(int n) {
  if (n <= 1) {
    return 1;
  }
  return static_cast<int>(
      pocketfft::detail::util::good_size_cmplx(static_cast<size_t>(n)));
}

POCKETFFT_EXPORT kiss_fftr_cfg kiss_fftr_alloc(
    int nfft,
    int inverse_fft,
    void* mem,
    size_t* lenmem) {
  if (nfft <= 0) {
    return nullptr;
  }
  try {
    (void)get_r_plan(static_cast<size_t>(nfft));
  } catch (...) {
    return nullptr;
  }

  const size_t memneeded = sizeof(struct kiss_fftr_state);
  kiss_fftr_cfg st = nullptr;
  if (lenmem == nullptr) {
    st = static_cast<kiss_fftr_cfg>(std::malloc(memneeded));
  } else {
    if (mem != nullptr && *lenmem >= memneeded) {
      st = static_cast<kiss_fftr_cfg>(mem);
    }
    *lenmem = memneeded;
  }
  if (st != nullptr) {
    st->nfft = nfft;
    st->inverse = inverse_fft ? 1 : 0;
  }
  return st;
}

POCKETFFT_EXPORT void kiss_fftr(
    kiss_fftr_cfg cfg,
    const double* timedata,
    kiss_fft_cpx* freqdata) {
  if (cfg == nullptr || timedata == nullptr || freqdata == nullptr ||
      cfg->nfft <= 0 || cfg->inverse != 0) {
    return;
  }
  try {
    const size_t n = static_cast<size_t>(cfg->nfft);
    const auto& plan = get_r_plan(n);

    // freqdata has (n / 2 + 1) complex elements = (2 * (n / 2) + 2) >= n + 1
    // doubles, so we can execute the in-place real transform directly in the
    // leading n doubles of freqdata and unpack backwards without extra heap
    // allocation.
    double* buf = reinterpret_cast<double*>(freqdata);
    std::memmove(buf, timedata, n * sizeof(double));
    plan.exec(buf, 1.0, true);

    if ((n & 1) == 0) {
      freqdata[n / 2].r = buf[n - 1];
      freqdata[n / 2].i = 0.0;
    }
    for (size_t k = (n - 1) / 2; k >= 1; --k) {
      const double re = buf[2 * k - 1];
      const double im = buf[2 * k];
      freqdata[k].r = re;
      freqdata[k].i = im;
    }
    freqdata[0].r = buf[0];
    freqdata[0].i = 0.0;
  } catch (...) {
  }
}

POCKETFFT_EXPORT void kiss_fftri(
    kiss_fftr_cfg cfg,
    const kiss_fft_cpx* freqdata,
    double* timedata) {
  if (cfg == nullptr || freqdata == nullptr || timedata == nullptr ||
      cfg->nfft <= 0 || cfg->inverse == 0) {
    return;
  }
  try {
    const size_t n = static_cast<size_t>(cfg->nfft);
    const auto& plan = get_r_plan(n);

    const double* freq_d = reinterpret_cast<const double*>(freqdata);
    const size_t freq_doubles = (n / 2 + 1) * 2;
    const bool overlaps =
        (timedata < freq_d + freq_doubles) && (freq_d < timedata + n);

    if (!overlaps || timedata == freq_d) {
      timedata[0] = freqdata[0].r;
      const size_t half = (n - 1) / 2;
      for (size_t k = 1; k <= half; ++k) {
        const double re = freqdata[k].r;
        const double im = freqdata[k].i;
        timedata[2 * k - 1] = re;
        timedata[2 * k] = im;
      }
      if ((n & 1) == 0) {
        timedata[n - 1] = freqdata[n / 2].r;
      }
      plan.exec(timedata, 1.0, false);
    } else {
      pocketfft::detail::arr<double> tmp(n);
      tmp[0] = freqdata[0].r;
      const size_t half = (n - 1) / 2;
      for (size_t k = 1; k <= half; ++k) {
        tmp[2 * k - 1] = freqdata[k].r;
        tmp[2 * k] = freqdata[k].i;
      }
      if ((n & 1) == 0) {
        tmp[n - 1] = freqdata[n / 2].r;
      }
      plan.exec(tmp.data(), 1.0, false);
      std::memcpy(timedata, tmp.data(), n * sizeof(double));
    }
  } catch (...) {
  }
}

POCKETFFT_EXPORT kiss_fftnd_cfg kiss_fftnd_alloc(
    const int* dims,
    int ndims,
    int inverse_fft,
    void* mem,
    size_t* lenmem) {
  if (dims == nullptr || ndims <= 0) {
    return nullptr;
  }
  for (int i = 0; i < ndims; ++i) {
    if (dims[i] <= 0) {
      return nullptr;
    }
  }

  try {
    for (int i = 0; i < ndims; ++i) {
      (void)get_c_plan(static_cast<size_t>(dims[i]));
    }
  } catch (...) {
    return nullptr;
  }

  const size_t memneeded = sizeof(struct kiss_fftnd_state) +
      static_cast<size_t>(ndims - 1) * sizeof(int);
  kiss_fftnd_cfg st = nullptr;
  if (lenmem == nullptr) {
    st = static_cast<kiss_fftnd_cfg>(std::malloc(memneeded));
  } else {
    if (mem != nullptr && *lenmem >= memneeded) {
      st = static_cast<kiss_fftnd_cfg>(mem);
    }
    *lenmem = memneeded;
  }
  if (st != nullptr) {
    st->inverse = inverse_fft ? 1 : 0;
    st->ndims = ndims;
    for (int i = 0; i < ndims; ++i) {
      st->dims[i] = dims[i];
    }
  }
  return st;
}

POCKETFFT_EXPORT void kiss_fftnd(
    kiss_fftnd_cfg cfg,
    const kiss_fft_cpx* fin,
    kiss_fft_cpx* fout) {
  if (cfg == nullptr || fin == nullptr || fout == nullptr || cfg->ndims <= 0) {
    return;
  }
  try {
    const size_t ndims = static_cast<size_t>(cfg->ndims);
    pocketfft::shape_t shape(ndims);
    pocketfft::stride_t stride(ndims);
    pocketfft::shape_t axes(ndims);

    for (size_t i = 0; i < ndims; ++i) {
      if (cfg->dims[i] <= 0) {
        return;
      }
      shape[i] = static_cast<size_t>(cfg->dims[i]);
      axes[i] = i;
    }

    ptrdiff_t cur_stride =
        static_cast<ptrdiff_t>(sizeof(std::complex<double>));
    for (size_t i = ndims; i > 0; --i) {
      stride[i - 1] = cur_stride;
      cur_stride *= static_cast<ptrdiff_t>(shape[i - 1]);
    }

    pocketfft::c2c<double>(
        shape,
        stride,
        stride,
        axes,
        cfg->inverse == 0 ? pocketfft::FORWARD : pocketfft::BACKWARD,
        reinterpret_cast<const std::complex<double>*>(fin),
        reinterpret_cast<std::complex<double>*>(fout),
        1.0);
  } catch (...) {
  }
}
