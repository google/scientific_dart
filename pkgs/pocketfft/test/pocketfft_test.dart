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

import 'package:ffi/ffi.dart';
import 'package:pocketfft/pocketfft.dart';
import 'package:test/test.dart';

void main() {
  group('package:pocketfft Raw FFI Bindings Tests', () {
    test(
      'Perform basic raw FFI kiss_fft transform on unmanaged heap structs',
      () {
        const nfft = 4;
        // Allocate forward transform plan (0 for forward)
        final cfg = kiss_fft_alloc(nfft, 0, ffi.nullptr, ffi.nullptr);
        expect(cfg.address != 0, true);

        final fin = malloc<kiss_fft_cpx>(nfft);
        final fout = malloc<kiss_fft_cpx>(nfft);

        try {
          // Initialize flat Kronecker delta spike input: [1.0 + 0i, 0i, 0i, 0i]
          fin[0].r = 1.0;
          fin[0].i = 0.0;
          fin[1].r = 0.0;
          fin[1].i = 0.0;
          fin[2].r = 0.0;
          fin[2].i = 0.0;
          fin[3].r = 0.0;
          fin[3].i = 0.0;

          // Execute native FFT
          kiss_fft(cfg, fin, fout);

          // Kronecker delta spike FFT is a perfectly flat spectrum: [1.0 + 0i, 1.0 + 0i, 1.0 + 0i, 1.0 + 0i]!
          for (var i = 0; i < nfft; i++) {
            expect(fout[i].r, closeTo(1.0, 1e-9));
            expect(fout[i].i, closeTo(0.0, 1e-9));
          }
        } finally {
          // Release heap configurations and buffers
          malloc.free(cfg);
          malloc.free(fin);
          malloc.free(fout);
        }
      },
    );
    test('Can allocate and free ND FFT config', () {
      final dims = malloc<ffi.Int>(2);
      dims[0] = 2;
      dims[1] = 2;
      final cfg = kiss_fftnd_alloc(dims, 2, 0, ffi.nullptr, ffi.nullptr);
      expect(cfg.address != 0, true);
      malloc.free(cfg);
      malloc.free(dims);
    });
  });

  group('PocketFFTPlanCache Tests', () {
    late PocketFFTPlanCache cache;

    setUp(() {
      cache = PocketFFTPlanCache(maxCapacity: 4);
    });

    tearDown(() {
      cache.dispose();
    });

    test('Caches and reuses 1D complex FFT plans', () {
      final plan1 = cache.getPlan(16, isInverse: false);
      expect(plan1.address, isNot(0));
      expect(cache.size, equals(1));

      // Same plan should return identical cached pointer
      final plan2 = cache.getPlan(16, isInverse: false);
      expect(plan2.address, equals(plan1.address));
      expect(cache.size, equals(1));

      // Inverse plan has different key
      final planInv = cache.getPlan(16, isInverse: true);
      expect(planInv.address, isNot(0));
      expect(planInv.address, isNot(plan1.address));
      expect(cache.size, equals(2));
    });

    test('Caches and reuses 1D real FFT plans', () {
      final plan1 = cache.getRealPlan(32, isInverse: false);
      expect(plan1.address, isNot(0));
      expect(cache.size, equals(1));

      final plan2 = cache.getRealPlan(32, isInverse: false);
      expect(plan2.address, equals(plan1.address));
      expect(cache.size, equals(1));

      final planInv = cache.getRealPlan(32, isInverse: true);
      expect(planInv.address, isNot(0));
      expect(planInv.address, isNot(plan1.address));
      expect(cache.size, equals(2));
    });

    test('Caches and reuses ND complex FFT plans', () {
      final plan1 = cache.getNDPlan([8, 8], isInverse: false);
      expect(plan1.address, isNot(0));
      expect(cache.size, equals(1));

      // Equal list should match cached key
      final plan2 = cache.getNDPlan([8, 8], isInverse: false);
      expect(plan2.address, equals(plan1.address));
      expect(cache.size, equals(1));

      // Different dimensions
      final plan3 = cache.getNDPlan([8, 16], isInverse: false);
      expect(plan3.address, isNot(0));
      expect(plan3.address, isNot(plan1.address));
      expect(cache.size, equals(2));
    });

    test('LRU eviction works correctly when maxCapacity is reached', () {
      final p1 = cache.getPlan(8);
      final p2 = cache.getPlan(16);
      final p3 = cache.getPlan(32);
      final p4 = cache.getPlan(64);
      expect(p1.address, isNot(0));
      expect(p2.address, isNot(0));
      expect(p3.address, isNot(0));
      expect(p4.address, isNot(0));
      expect(cache.size, equals(4));

      // Touch p1 so p2 becomes least recently used
      final p1Touch = cache.getPlan(8);
      expect(p1Touch.address, equals(p1.address));

      // Insert 5th element -> p2 should be evicted and freed
      final p5 = cache.getPlan(128);
      expect(p5.address, isNot(0));
      expect(cache.size, equals(4));

      // p1, p3, p4, p5 should be in cache; p2 was evicted
      // Requesting 16 will allocate a new plan (or reallocate)
      final p2New = cache.getPlan(16);
      expect(p2New.address, isNot(0));
      expect(cache.size, equals(4));
    });

    test('clear() frees all plans and resets size to 0', () {
      cache.getPlan(8);
      cache.getRealPlan(16);
      cache.getNDPlan([4, 4]);
      expect(cache.size, equals(3));

      cache.clear();
      expect(cache.size, equals(0));
    });

    test('Changing maxCapacity evicts oldest items if reduced', () {
      cache.getPlan(8);
      cache.getPlan(16);
      cache.getPlan(32);
      expect(cache.size, equals(3));

      cache.maxCapacity = 2;
      expect(cache.size, equals(2));
      expect(cache.maxCapacity, equals(2));
    });

    test('Preconditions and error handling', () {
      expect(() => PocketFFTPlanCache(maxCapacity: 0), throwsArgumentError);
      expect(() => PocketFFTPlanCache(maxCapacity: -1), throwsArgumentError);

      expect(() => cache.getPlan(0), throwsArgumentError);
      expect(() => cache.getPlan(-5), throwsArgumentError);

      expect(() => cache.getRealPlan(0), throwsArgumentError);
      expect(() => cache.getRealPlan(7), throwsArgumentError); // Odd length

      expect(() => cache.getNDPlan([]), throwsArgumentError);
      expect(() => cache.getNDPlan([4, 0]), throwsArgumentError);
      expect(() => cache.getNDPlan([4, -2]), throwsArgumentError);

      cache.dispose();
      expect(cache.isDisposed, isTrue);
      expect(() => cache.getPlan(8), throwsStateError);
      expect(() => cache.getRealPlan(8), throwsStateError);
      expect(() => cache.getNDPlan([4, 4]), throwsStateError);
      expect(() => cache.clear(), throwsStateError);
      expect(() => cache.maxCapacity = 10, throwsStateError);
    });

    test('Global helpers use singleton default cache', () {
      clearPocketFFTPlanCache();
      final p1 = getCachedKissFFTPlan(16);
      final p2 = getCachedKissFFTPlan(16);
      expect(p2.address, equals(p1.address));

      final pr1 = getCachedKissFFTRPlan(16);
      final pr2 = getCachedKissFFTRPlan(16);
      expect(pr2.address, equals(pr1.address));

      final pnd1 = getCachedKissFFTNDPlan([4, 4]);
      final pnd2 = getCachedKissFFTNDPlan([4, 4]);
      expect(pnd2.address, equals(pnd1.address));

      clearPocketFFTPlanCache();
      expect(PocketFFTPlanCache.instance.size, equals(0));
    });
  });
}
