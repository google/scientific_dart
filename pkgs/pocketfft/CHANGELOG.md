## 0.2.0

- **Breaking — 64-Bit FFT Lengths, Strides, & Dimensions**: Upgraded `kiss_fft_alloc`, `kiss_fft_stride`, `kiss_fft_next_fast_size`, `kiss_fftr_alloc`, `kiss_fftnd_alloc`, and `PocketFftPlanCache` from 32-bit `int` (`ffi.Int`) to 64-bit `int64_t` (`ffi.Int64`) so transform sizes, element strides, and N-D dimensions do not truncate on 64-bit arrays.

## 0.1.0

- Initial release.
- Exposes native AOT FFI bindings for high-performance mixed-radix Fast Fourier Transforms (KissFFT).
- Implements multi-platform matrix C compilers build hooks (`hook/build.dart`) support.
