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

import 'dart:ffi' as ffi;

import 'package:criterion/criterion.dart';
import 'package:ffi/ffi.dart';
import 'package:pocketfft/pocketfft.dart';

void main() async {
  final lengths = [256, 1024, 4096, 16384, 65536];

  await criterion(
    'Native PocketFFT (kiss_fft) C Bindings Benchmarks',
    (c) {
      c.group('1D Complex Transform (Preallocated Plan)', () {
        for (final nfft in lengths) {
          final cfg = kiss_fft_alloc(nfft, 0, ffi.nullptr, ffi.nullptr);
          final fin = malloc<kiss_fft_cpx>(nfft);
          final fout = malloc<kiss_fft_cpx>(nfft);

          // Fill with sinusoid signal
          for (var i = 0; i < nfft; i++) {
            fin[i].r = (i * 0.1);
            fin[i].i = 0.0;
          }

          c.bench('kiss_fft forward (1D complex) [$nfft]', () {
            kiss_fft(cfg, fin, fout);
            blackhole(fout[0].r);
          }, throughput: Throughput.elements(nfft));

          // Clean up in a finally block after criterion finishes or register benchmark teardown
          // Note: We can also benchmark inverse transform
          final cfgInv = kiss_fft_alloc(nfft, 1, ffi.nullptr, ffi.nullptr);
          c.bench('kiss_fft inverse (1D complex) [$nfft]', () {
            kiss_fft(cfgInv, fout, fin);
            blackhole(fin[0].r);
          }, throughput: Throughput.elements(nfft));
        }
      });

      c.group('FFT Plan Allocation Overhead', () {
        c.benchWith<void, int>('kiss_fft_alloc + free', [256, 1024, 4096], (
          nfft,
        ) {
          final cfg = kiss_fft_alloc(nfft, 0, ffi.nullptr, ffi.nullptr);
          blackhole(cfg.address);
          malloc.free(cfg);
        });
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/pocketfft',
    ),
  );
}
