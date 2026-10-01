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

/// Fast Fourier Transform operations (`fft`, `ifft`, `rfft`, `irfft`, `fft2`,
/// `ifft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn`, `hfft`,
/// `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`) executed on WebGPU.
library;

export 'src/fft/fft.dart'
    show
        FftNorm,
        fft,
        fft2,
        fftfreq,
        fftn,
        fftshift,
        hfft,
        ifft,
        ifft2,
        ifftn,
        ifftshift,
        ihfft,
        irfft,
        irfft2,
        irfftn,
        rfft,
        rfft2,
        rfftfreq,
        rfftn;
