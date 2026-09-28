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

/// Fast Fourier Transform (FFT) operations for [GpuArray] tensors.
///
/// Provides 1D and 2D complex and real discrete Fourier transforms ([fft],
/// [ifft], [rfft], [irfft], [fft2], [ifft2]), frequency bin generators
/// ([fftfreq], [rfftfreq]), spectrum shifting utilities ([fftshift],
/// [ifftshift]), and normalization modes ([FftNorm]).
library;

export 'src/fft/fft.dart'
    show
        FftNorm,
        fft,
        fft2,
        fftfreq,
        fftshift,
        ifft,
        ifft2,
        ifftshift,
        irfft,
        rfft,
        rfftfreq;
