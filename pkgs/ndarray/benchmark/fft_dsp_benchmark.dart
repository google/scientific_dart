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

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);

  await criterion(
    'NDArray Real FFT, 2D Transform & Window Functions Benchmark Suite',
    (c) {
      c.group('1. Real-Valued 1D Transforms (rfft & irfft)', () {
        for (final length in [1024, 4096, 16384, 65536]) {
          final realSignal = linspace(0.0, 100.0, length, dtype: DType.float64);

          c.bench('rfft(realSignal) [length=$length]', () {
            final spec = rfft((realSignal as NDArray<AnySpec>));
            blackhole(spec);
            spec.dispose();
          }, throughput: Throughput.elements(length));

          final specInput = rfft((realSignal as NDArray<AnySpec>));
          c.bench('irfft(spec) [length=$length]', () {
            final recovered = irfft((specInput as NDArray<AnySpec>), n: length);
            blackhole(recovered);
            recovered.dispose();
          }, throughput: Throughput.elements(length));
        }
      });

      c.group('2. 2D Complex Fourier Transforms (fft2 & ifft2)', () {
        for (final dim in [256, 512]) {
          final img2d = NDArray<AnySpec>.zeros([dim, dim], DType.float64);
          for (var i = 0; i < dim; i++) {
            img2d.setCell([i, i], 1.0);
          }

          c.bench('fft2 [${dim}x$dim]', () {
            final res = fft2(img2d);
            blackhole(res);
            res.dispose();
          }, throughput: Throughput.elements(dim * dim));

          final imgSpec = fft2(img2d);
          c.bench('ifft2 [${dim}x$dim]', () {
            final res = ifft2((imgSpec as NDArray<AnySpec>));
            blackhole(res);
            res.dispose();
          }, throughput: Throughput.elements(dim * dim));
        }
      });

      c.group('3. DSP Window Functions', () {
        const windowSize = 100000;

        c.bench('hanning($windowSize)', () {
          final w = hanning(windowSize);
          blackhole(w);
          w.dispose();
        }, throughput: Throughput.elements(windowSize));

        c.bench('hamming($windowSize)', () {
          final w = hamming(windowSize);
          blackhole(w);
          w.dispose();
        }, throughput: Throughput.elements(windowSize));
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/fft_dsp',
    ),
  );
}
