# ndarray

[![Pub Version](https://img.shields.io/pub/v/ndarray)](https://pub.dev/packages/ndarray)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**ndarray** is a high-performance N-dimensional array and numerical computing library for Dart, heavily inspired by NumPy. It supports a rich suite of mathematical, signal processing, and linear algebra operations, including:

- **Element-wise Mathematics**: Trigonometric (`sin`, `cos`, `tan`), hyperbolic (`sinh`, `cosh`, `tanh`), exponential (`exp`), and logarithmic (`log`) operations.
- **Linear Algebra (OpenBLAS/LAPACK)**: Matrix multiplications (`matmul`), determinants (`det`), inversion (`inv`), QR, SVD decompositions, and linear system solvers (`solve`).
- **Discrete Fourier Transforms**: Mixed-radix spectral mappings (`fft`, `ifft`) and spectrum shifts (`fftshift`, `ifftshift`).
- **Geometry & Manipulation**: Stride-based zero-copy slices, transposes, shape views (`reshape`, `expand_dims`, `squeeze`), mesh grid spacers (`mgrid`, `ogrid`), repeating/tiling (`repeat`, `tile`), and block splitting (`split`, `array_split`, `hsplit`, `vsplit`).
- **Statistics & Reductions**: Sum, product, mean, variance, standard deviation, and NaN-ignoring reductions (`nansum`, `nanmean`, `nanvar`, `nanstd`).
- **RNG Distributions**: Normal (Gaussian), Poisson, and Binomial sample generation.

---

## Key Features

- 🚀 **FFI Native Acceleration**: Directly offloads vector operations, transcendental functions, and reductions to optimized C kernels, bypassing JIT overhead.
- 🎛️ **Zero-Copy Slices & Transposes**: Supports stride-based multi-dimensional array views, transposes, dimensions expansion (`expand_dims`), and squeezing (`squeeze`) with zero memory copies.
- 🧮 **OpenBLAS Linear Algebra**: Accelerated determinant (`det`), matrix inversion (`inv`), QR, SVD, and solvers (`solve`) powered by raw LAPACK.
- 📶 **Mixed-Radix pocketfft**: Fast Fourier Transform (`fft`) and Inverse FFT (`ifft`) mapped to PocketFFT unmanaged kernels.
- 🎲 **RNG Distributions**: High-performance Normal (Gaussian), Poisson, and Binomial sample generation.
- 🔍 **Data Sanitation & Tolerance Comparisons**: Approximate floating-point equality comparisons (`isClose`/`allClose`) and NaN/Infinities cleaning (`nan_to_num`).
- 🧹 **Automatic Resource Management Scope**: Zone-based scoped lifetimes (`NDArray.scope()`) that automatically release raw unmanaged heap allocations of transient arrays. [Read the Tutorial](doc/memory_management.md).

---

## Getting Started: Scientific DSP Denoising Tutorial

Here is a comprehensive end-to-end example demonstrating how to generate a signal, inject RNG Gaussian noise, execute FFI Mixed-Radix FFT spectral mapping, apply a frequency low-pass filter, restore the time-domain signal via FFI IFFT, and verify reconstructed correctness using tolerance comparisons:

```dart
import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Master Scientific & ML Signal Denoising Tutorial ===\n');

  // Wrap operations inside a zone-based scope for automatic unmanaged heap resource cleanup!
  NDArray.scope(() {
    // 1. Generate pure time-domain signal
    // Sample a 10 Hz sine wave at 128 Hz for 1 second (128 points)
    const samplingRate = 128.0;
    const numPoints = 128;

    final time = NDArray.fromList(
      List.generate(numPoints, (i) => i / samplingRate),
      [numPoints],
      DType.float64,
    );
    final pureSignal = NDArray.fromList(
      List.generate(
        numPoints,
        (i) => 5.0 * math.sin(2.0 * math.pi * 10.0 * (i / samplingRate)),
      ),
      [numPoints],
      DType.float64,
    );
    print('1. Generated pure 10 Hz sine wave signal.');

    // 2. Inject RNG Gaussian Noise to simulate measurement sensors
    final noise = normal([numPoints], loc: 0.0, scale: 0.5, random: math.Random(42));
    
    // Combine pure signal + noise element-wise!
    final noisySignal = add(pureSignal, noise);
    print('2. Injected RNG Gaussian noise (loc = 0.0, scale = 0.5) element-wise.');

    // 3. Execute FFI-Accelerated mixed-radix FFT to map signal to frequency space
    final fftCoeffs = fft(noisySignal);
    print('3. Executed mixed-radix FFI FFT to transform signal to frequency space.');

    // 4. Low-pass Filter: Filter out high frequencies (noise) above 15 Hz!
    final halfLen = numPoints ~/ 2;
    var filteredBinsCount = 0;

    for (var i = 0; i <= halfLen; i++) {
      final freq = i * samplingRate / numPoints;
      if (freq > 15.0) {
        // Symmetrically zero out positive and corresponding negative high frequency bins!
        fftCoeffs[i] = Complex(0.0, 0.0);
        if (i > 0 && i < halfLen) {
          fftCoeffs[numPoints - i] = Complex(0.0, 0.0);
        }
        filteredBinsCount++;
      }
    }
    print('4. Low-pass filter applied: zeroed out $filteredBinsCount noise bins (> 15 Hz).');

    // 5. Restoration: Inverse Fourier Transform (IFFT) back to time domain!
    final restoredComplex = ifft(fftCoeffs);

    // Extract real component of the reconstructed signal
    final reconstructed = NDArray.fromList(
      restoredComplex.toList().map((c) => c.real).toList(),
      [numPoints],
      DType.float64,
    );
    print('5. Executed FFI IFFT to restore signal back to time-domain.');

    // 6. Compare reconstructed signal against original pure signal!
    final isApproximatelyClose = allClose(pureSignal, reconstructed, rtol: 0.3, atol: 0.5);
    print('\n=== Verification Results ===');
    print('Is restored signal approximately close to pure signal? $isApproximatelyClose');
  });
}
```

---

## Installation

Add `ndarray` to your `pubspec.yaml`:

```yaml
dependencies:
  ndarray: ^0.1.0
```

### Compiling Native Libraries

`ndarray` relies on native compiled shared libraries for PocketFFT and custom universal ufuncs kernels.

These are automatically compiled on the fly via the modern Dart Native Assets system (integrated seamlessly into build/run pipelines) when executing your application or running tests. No manual compilation steps are necessary!
This parallelizes compilation over available hardware CPU cores to complete setup in seconds.

### Limitations relative to numpy.

NdArray is not a drop in replacement for numpy.

NDArrays are strictly numerical, and only supports a fixed number of dtypes. 
The supported types are: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64, complex64, complex128.

Memory is manually allocated and released. You can use NDArray.scope() to automatically release memory, but you still have to be aware of what you are doing. See [memory management](doc/memory_management.md).

NDArrays support arbitrary non-contiguous strided layouts, including transposed column-major (Fortran-ordered) views.


---

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
