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

import 'package:ndarray/ndarray.dart';

/// A class containing various implementations of 2D convolution.
final class ImageConvolution {
  /// Generates a synthetic 2D image of shape [height, width] with a square in the middle.
  ///
  /// The background is 0.0 and the square is 1.0.
  static NDArray<Float64> generateSyntheticImage(int height, int width) {
    final image = NDArray<Float64>.zeros([height, width], DType.float64);

    // Define square boundaries (middle 40%)
    final startY = (height * 0.3).toInt();
    final endY = (height * 0.7).toInt();
    final startX = (width * 0.3).toInt();
    final endX = (width * 0.7).toInt();

    final subView = image.slice([
      Slice(start: startY, stop: endY),
      Slice(start: startX, stop: endX),
    ]);
    subView.fill(Float64(1.0));

    return image;
  }

  /// Naive 2D convolution implementation using nested loops in Dart.
  ///
  /// This is used as a baseline for performance and correctness.
  static NDArray<Float64> convolve2DNaive(
    NDArray<Float64> image,
    NDArray<Float64> kernel,
  ) {
    if (image.shape.length != 2 || kernel.shape.length != 2) {
      throw ArgumentError('Both image and kernel must be 2D arrays.');
    }

    final imShape = image.shape;
    final kShape = kernel.shape;
    final H = imShape[0];
    final W = imShape[1];
    final kH = kShape[0];
    final kW = kShape[1];

    if (H < kH || W < kW) {
      throw ArgumentError(
        'Kernel dimensions must be smaller than image dimensions.',
      );
    }

    final outH = H - kH + 1;
    final outW = W - kW + 1;

    final result = NDArray<Float64>.zeros([outH, outW], DType.float64);

    for (var i = 0; i < outH; i++) {
      for (var j = 0; j < outW; j++) {
        var sum = 0.0;
        for (var ki = 0; ki < kH; ki++) {
          for (var kj = 0; kj < kW; kj++) {
            final imVal = image.getCell([i + ki, j + kj]);
            final kVal = kernel.getCell([ki, kj]);
            sum += imVal * kVal;
          }
        }
        result.setCell([i, j], Float64(sum));
      }
    }
    return result;
  }

  /// Optimized generic 2D convolution (correlation) using vectorized slicing.
  ///
  /// Loops over the kernel elements and performs vectorized operations on image slices.
  /// Reuses temporary buffers to minimize allocations.
  static NDArray<Float64> convolve2DGeneric(
    NDArray<Float64> image,
    NDArray<Float64> kernel, {
    NDArray<Float64>? out,
  }) {
    if (image.shape.length != 2 || kernel.shape.length != 2) {
      throw ArgumentError('Both image and kernel must be 2D arrays.');
    }

    final imShape = image.shape;
    final kShape = kernel.shape;
    final H = imShape[0];
    final W = imShape[1];
    final kH = kShape[0];
    final kW = kShape[1];

    if (H < kH || W < kW) {
      throw ArgumentError(
        'Kernel dimensions must be smaller than image dimensions.',
      );
    }

    final outH = H - kH + 1;
    final outW = W - kW + 1;

    final NDArray<Float64> result =
        out ?? NDArray<Float64>.zeros([outH, outW], DType.float64);
    if (out != null) {
      result.fill(Float64(0.0));
    }

    // Pre-allocate temp buffer for scaled slice.
    final temp = NDArray<Float64>.zeros([outH, outW], DType.float64);
    // Pre-allocate 1D array of size 1 for factor (to allow setCell and broadcasting)
    final factor = NDArray<Float64>.zeros([1], DType.float64);

    for (var i = 0; i < kH; i++) {
      for (var j = 0; j < kW; j++) {
        final kVal = kernel.getCell([i, j]);
        if (kVal == 0.0) continue;

        factor.setCell([0], Float64(kVal));

        final imSlice = image.slice([
          Slice(start: i, stop: outH + i),
          Slice(start: j, stop: outW + j),
        ]);

        // temp = imSlice * factor
        multiply<Float64, Float64, Float64>(imSlice, factor, out: temp);

        // result = result + temp
        add<Float64, Float64, Float64>(result, temp, out: result);
      }
    }

    temp.dispose();
    factor.dispose();

    return result;
  }

  /// Highly optimized Sobel edge detection (3x3 kernel) using hardcoded 2D slices.
  ///
  /// This version avoids the kernel loop entirely and pre-allocates all intermediate buffers.
  /// It only works for 3x3 Sobel-like operations.
  static void sobel3X3Optimized(
    NDArray<Float64> image, {
    required NDArray<Float64> outGx,
    required NDArray<Float64> outGy,
    required NDArray<Float64> outMag,
    required NDArray<Float64> temp1,
    required NDArray<Float64> temp2,
    required NDArray<Float64> constantTwo,
  }) {
    final h = image.shape[0];
    final w = image.shape[1];

    // Slices (views, zero-copy)
    final topLeft = image.slice([
      Slice(start: 0, stop: h - 2),
      Slice(start: 0, stop: w - 2),
    ]);
    final topMid = image.slice([
      Slice(start: 0, stop: h - 2),
      Slice(start: 1, stop: w - 1),
    ]);
    final topRight = image.slice([
      Slice(start: 0, stop: h - 2),
      Slice(start: 2, stop: w),
    ]);

    final midLeft = image.slice([
      Slice(start: 1, stop: h - 1),
      Slice(start: 0, stop: w - 2),
    ]);
    final midRight = image.slice([
      Slice(start: 1, stop: h - 1),
      Slice(start: 2, stop: w),
    ]);

    final botLeft = image.slice([
      Slice(start: 2, stop: h),
      Slice(start: 0, stop: w - 2),
    ]);
    final botMid = image.slice([
      Slice(start: 2, stop: h),
      Slice(start: 1, stop: w - 1),
    ]);
    final botRight = image.slice([
      Slice(start: 2, stop: h),
      Slice(start: 2, stop: w),
    ]);

    // Gx = (topRight - topLeft) + 2 * (midRight - midLeft) + (botRight - botLeft)
    subtract<Float64, Float64, Float64>(topRight, topLeft, out: outGx);

    subtract<Float64, Float64, Float64>(midRight, midLeft, out: temp1);
    multiply<Float64, Float64, Float64>(temp1, constantTwo, out: temp2);
    add<Float64, Float64, Float64>(outGx, temp2, out: outGx);

    subtract<Float64, Float64, Float64>(botRight, botLeft, out: temp1);
    add<Float64, Float64, Float64>(outGx, temp1, out: outGx);

    // Gy = (botLeft - topLeft) + 2 * (botMid - topMid) + (botRight - topRight)
    subtract<Float64, Float64, Float64>(botLeft, topLeft, out: outGy);

    subtract<Float64, Float64, Float64>(botMid, topMid, out: temp1);
    multiply<Float64, Float64, Float64>(temp1, constantTwo, out: temp2);
    add<Float64, Float64, Float64>(outGy, temp2, out: outGy);

    subtract<Float64, Float64, Float64>(botRight, topRight, out: temp1);
    add<Float64, Float64, Float64>(outGy, temp1, out: outGy);

    // Magnitude: outMag = sqrt(Gx*Gx + Gy*Gy)
    multiply<Float64, Float64, Float64>(outGx, outGx, out: temp1);
    multiply<Float64, Float64, Float64>(outGy, outGy, out: temp2);
    add<Float64, Float64, Float64>(temp1, temp2, out: temp1);
    sqrt<Float64, Float64>(temp1, out: outMag);
  }
}

void main() {
  NDArray.scope(() {
    print('=== Image Convolution Example (Sobel Edge Detection) ===\n');

    const height = 512;
    const width = 512;
    const iterations = 20;
    print('Generating synthetic $height x $width image...');
    final image = ImageConvolution.generateSyntheticImage(height, width);

    print('Defining Sobel kernels...');
    final sobelX = NDArray<Float64>.fromList(
      <Float64>[
        Float64(-1.0),
        Float64(0.0),
        Float64(1.0),
        Float64(-2.0),
        Float64(0.0),
        Float64(2.0),
        Float64(-1.0),
        Float64(0.0),
        Float64(1.0),
      ],
      [3, 3],
      DType.float64,
    );

    final sobelY = NDArray<Float64>.fromList(
      <Float64>[
        Float64(-1.0),
        Float64(-2.0),
        Float64(-1.0),
        Float64(0.0),
        Float64(0.0),
        Float64(0.0),
        Float64(1.0),
        Float64(2.0),
        Float64(1.0),
      ],
      [3, 3],
      DType.float64,
    );

    final constantTwo = NDArray<Float64>.fromList(
      <Float64>[Float64(2.0)],
      [1],
      DType.float64,
    );

    // Warmup
    print('Warming up...');
    for (var i = 0; i < 5; i++) {
      final dummy = ImageConvolution.convolve2DGeneric(image, sobelX);
      dummy.dispose();
    }
    print('Warmup complete.\n');

    // 1. Naive Dart
    print('Benchmarking Naive Dart...');
    final stopwatchNaive = Stopwatch()..start();
    late NDArray<Float64> gradXNaive;
    late NDArray<Float64> gradYNaive;
    for (var i = 0; i < iterations; i++) {
      gradXNaive = ImageConvolution.convolve2DNaive(image, sobelX);
      gradYNaive = ImageConvolution.convolve2DNaive(image, sobelY);
      if (i < iterations - 1) {
        gradXNaive.dispose();
        gradYNaive.dispose();
      }
    }
    stopwatchNaive.stop();
    final timeNaive = stopwatchNaive.elapsedMilliseconds / iterations;
    print('Naive Dart (Gx + Gy): ${timeNaive.toStringAsFixed(2)} ms/iter');

    // Naive Magnitude (using hypot on naive results)
    final stopwatchNaiveMag = Stopwatch()..start();
    final magnitudeNaive = hypot(gradXNaive, gradYNaive);
    stopwatchNaiveMag.stop();
    print(
      'Naive Magnitude (hypot): ${stopwatchNaiveMag.elapsedMicroseconds / 1000.0} ms',
    );

    // 2. Generic Vectorized
    print('\nBenchmarking Generic Vectorized...');
    final stopwatchGeneric = Stopwatch()..start();
    late NDArray<Float64> gradXGeneric;
    late NDArray<Float64> gradYGeneric;
    for (var i = 0; i < iterations; i++) {
      gradXGeneric = ImageConvolution.convolve2DGeneric(image, sobelX);
      gradYGeneric = ImageConvolution.convolve2DGeneric(image, sobelY);
      if (i < iterations - 1) {
        gradXGeneric.dispose();
        gradYGeneric.dispose();
      }
    }
    stopwatchGeneric.stop();
    final timeGeneric = stopwatchGeneric.elapsedMilliseconds / iterations;
    print(
      'Generic Vectorized (Gx + Gy): ${timeGeneric.toStringAsFixed(2)} ms/iter',
    );

    // Generic Magnitude - Hypot
    final stopwatchGenericHypot = Stopwatch()..start();
    late NDArray<double> magnitudeGenericHypot;
    for (var i = 0; i < iterations; i++) {
      magnitudeGenericHypot = hypot(gradXGeneric, gradYGeneric);
      if (i < iterations - 1) {
        magnitudeGenericHypot.dispose();
      }
    }
    stopwatchGenericHypot.stop();
    final timeGenericHypot =
        (stopwatchGenericHypot.elapsedMicroseconds / 1000.0) / iterations;
    print(
      'Generic Magnitude (hypot): ${timeGenericHypot.toStringAsFixed(2)} ms/iter',
    );

    // Generic Magnitude - Allocated (sqrt(add(mul, mul)))
    final stopwatchGenericAllocMag = Stopwatch()..start();
    late NDArray<Float64> magnitudeGenericAlloc;
    for (var i = 0; i < iterations; i++) {
      final temp1 = multiply<Float64, Float64, Float64>(
        gradXGeneric,
        gradXGeneric,
      );
      final temp2 = multiply<Float64, Float64, Float64>(
        gradYGeneric,
        gradYGeneric,
      );
      final temp3 = add<Float64, Float64, Float64>(temp1, temp2);
      magnitudeGenericAlloc = sqrt<Float64, Float64>(temp3);
      temp1.dispose();
      temp2.dispose();
      temp3.dispose();
      if (i < iterations - 1) {
        magnitudeGenericAlloc.dispose();
      }
    }
    stopwatchGenericAllocMag.stop();
    final timeGenericAllocMag =
        (stopwatchGenericAllocMag.elapsedMicroseconds / 1000.0) / iterations;
    print(
      'Generic Magnitude (sqrt(add(mul,mul)) allocated): ${timeGenericAllocMag.toStringAsFixed(2)} ms/iter',
    );

    // Generic Magnitude - Pre-allocated (sqrt(add(mul, mul)))
    final outH = height - 2;
    final outW = width - 2;
    final mTemp1 = NDArray<Float64>.zeros([outH, outW], DType.float64);
    final mTemp2 = NDArray<Float64>.zeros([outH, outW], DType.float64);
    final magnitudeGenericPre = NDArray<Float64>.zeros([
      outH,
      outW,
    ], DType.float64);

    final stopwatchGenericPreMag = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      multiply<Float64, Float64, Float64>(
        gradXGeneric,
        gradXGeneric,
        out: mTemp1,
      );
      multiply<Float64, Float64, Float64>(
        gradYGeneric,
        gradYGeneric,
        out: mTemp2,
      );
      add<Float64, Float64, Float64>(mTemp1, mTemp2, out: mTemp1);
      sqrt<Float64, Float64>(mTemp1, out: magnitudeGenericPre);
    }
    stopwatchGenericPreMag.stop();
    final timeGenericPreMag =
        (stopwatchGenericPreMag.elapsedMicroseconds / 1000.0) / iterations;
    print(
      'Generic Magnitude (sqrt(add(mul,mul)) pre-allocated): ${timeGenericPreMag.toStringAsFixed(2)} ms/iter',
    );

    // 3. Specialized Sobel Optimized
    print('\nBenchmarking Specialized Sobel Optimized...');
    final outGx = NDArray<Float64>.zeros([outH, outW], DType.float64);
    final outGy = NDArray<Float64>.zeros([outH, outW], DType.float64);
    final outMag = NDArray<Float64>.zeros([outH, outW], DType.float64);
    final sTemp1 = NDArray<Float64>.zeros([outH, outW], DType.float64);
    final sTemp2 = NDArray<Float64>.zeros([outH, outW], DType.float64);

    final stopwatchSpecialized = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      ImageConvolution.sobel3X3Optimized(
        image,
        outGx: outGx,
        outGy: outGy,
        outMag: outMag,
        temp1: sTemp1,
        temp2: sTemp2,
        constantTwo: constantTwo,
      );
    }
    stopwatchSpecialized.stop();
    final timeSpecialized =
        stopwatchSpecialized.elapsedMilliseconds / iterations;
    print(
      'Specialized Sobel (Gx + Gy + Mag): ${timeSpecialized.toStringAsFixed(2)} ms/iter',
    );

    // Verify Correctness
    print('\nVerifying correctness...');
    var correct = true;
    for (var i = 0; i < outMag.shape[0]; i++) {
      for (var j = 0; j < outMag.shape[1]; j++) {
        final valNaive = magnitudeNaive.getCell([i, j]);
        final valGeneric = magnitudeGenericPre.getCell([i, j]);
        final valSpecial = outMag.getCell([i, j]);
        if ((valNaive - valGeneric).abs() > 1e-5 ||
            (valNaive - valSpecial).abs() > 1e-5) {
          correct = false;
          print(
            'Mismatch at [$i, $j]: Naive $valNaive, Generic $valGeneric, Special $valSpecial',
          );
          break;
        }
      }
      if (!correct) break;
    }
    if (correct) {
      print('All implementation results match.');
    } else {
      print('WARNING: Results mismatch!');
    }

    // Clean up
    magnitudeNaive.dispose();
    gradXNaive.dispose();
    gradYNaive.dispose();
    gradXGeneric.dispose();
    gradYGeneric.dispose();
    magnitudeGenericHypot.dispose();
    magnitudeGenericAlloc.dispose();
    mTemp1.dispose();
    mTemp2.dispose();
    magnitudeGenericPre.dispose();

    outGx.dispose();
    outGy.dispose();
    outMag.dispose();
    sTemp1.dispose();
    sTemp2.dispose();
    constantTwo.dispose();
  });
}
