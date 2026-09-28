# pocketfft

Native AOT FFI bindings for high-performance, mixed-radix Fast Fourier Transforms (FFT) in Dart.

This package wraps the highly reliable **KissFFT** mixed-radix library in unmanaged C space, enabling Fourier transform calculations across arbitrary sequence lengths (with optimal $O(N \log N)$ scaling even for non-power-of-two signals).

## Features
- **Mixed-Radix FFT**: Supports arbitrary sequence lengths utilizing prime factorizations natively.
- **Automatic Compilations Build Hooks**: Packages KissFFT plain-C source codes and includes automatic AOT compilations build hooks (`hook/build.dart`) that compile shared libraries dynamically across platforms.
- **Real & Complex Transforms**: Exposes both standard complex-to-complex transforms (`kiss_fft`) and Hermitian real-to-complex / complex-to-real fast-paths (`kiss_fftr`, `kiss_fftri`).

## Usage Example

Exposes the raw `pocketfft` bindings generated via `ffigen`:

```dart
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:pocketfft/pocketfft.dart';

void main() {
  final targetLen = 4;
  
  // 1. Allocate raw FFT configuration plan
  final kiss_fft_cfg cfg = kiss_fft_alloc(targetLen, 0, ffi.nullptr, ffi.nullptr);
  
  // 2. Allocate unmanaged FFI C-heap input and output structures
  final pin = malloc<kiss_fft_cpx>(targetLen);
  final pout = malloc<kiss_fft_cpx>(targetLen);
  
  try {
    // 3. Populate input buffer
    for (var i = 0; i < targetLen; i++) {
      pin[i].r = i.toDouble();
      pin[i].i = 0.0;
    }
    
    // 4. Fire native FFT
    kiss_fft(cfg, pin, pout);
    
    // 5. Print frequency domain coefficients
    for (var i = 0; i < targetLen; i++) {
      print('Out[$i]: ${pout[i].r} + ${pout[i].i}i');
    }
  } finally {
    // 6. Strictly release C-heap allocations and planners
    free(cfg.cast<ffi.Void>());
    malloc.free(pin);
    malloc.free(pout);
  }
}
```

## License
This package is licensed under the [Apache License, Version 2.0](LICENSE).
