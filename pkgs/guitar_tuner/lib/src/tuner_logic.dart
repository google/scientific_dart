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

import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';

final class TunerResult {
  final String note;
  final double frequency;
  final double targetFrequency;
  final double cents;
  final double rms;
  final String spectrumLine;

  TunerResult({
    required this.note,
    required this.frequency,
    required this.targetFrequency,
    required this.cents,
    required this.rms,
    required this.spectrumLine,
  });

  @override
  String toString() {
    final status = cents.abs() < 5
        ? 'IN TUNE'
        : (cents > 0 ? 'TOO HIGH' : 'TOO LOW');
    final visual = _generateVisual();
    final vol = _generateVolumeBar();
    return 'Vol: $vol | $note |$visual| (${frequency.toStringAsFixed(1)} Hz) - $status (${cents.toStringAsFixed(1)} cents)';
  }

  String _generateVolumeBar() {
    const width = 10;
    // Map RMS (approx 0 to 0.5 for loud signal) to bar width
    final int level = (rms * 20).round().clamp(0, width);
    return '[${'#' * level}${' ' * (width - level)}]';
  }

  String _generateVisual() {
    const width = 21; // odd number
    const center = width ~/ 2;
    final chars = List.filled(width, '-');
    chars[center] = '|';

    // Map cents (-50 to 50) to index (0 to 20)
    int needlePos = (center + (cents / 50.0 * center)).round();
    needlePos = needlePos.clamp(0, width - 1);

    if (cents.abs() < 5) {
      chars[needlePos] = '★'; // Special char for "in tune"
    } else {
      chars[needlePos] = '▲';
    }
    return chars.join();
  }
}

final class TunerLogic {
  final int sampleRate;
  final int bufferSize;
  final NDArray<Float32> _window;
  double? _smoothedFrequency;

  static const Map<String, double> guitarNotes = {
    'E2': 82.41,
    'A2': 110.00,
    'D3': 146.83,
    'G3': 196.00,
    'B3': 246.94,
    'E4': 329.63,
  };

  TunerLogic({required this.sampleRate, required this.bufferSize})
    : _window = hanning(bufferSize, dtype: DType.float32);

  TunerResult process(NDArray<Float32> input, double rms) {
    return NDArray.scope(() {
      // 2. Apply Hanning window to reduce spectral leakage
      final NDArray<Float32> windowedInput = multiply(input, _window);

      // 3. Perform FFT
      final spectrum = fft(windowedInput);

      // 4. Calculate magnitude spectrum
      final magnitudes = abs(spectrum);

      // 5. Generate spectrum line for waterfall plot (always visible)
      final String spectrumLine = _generateSpectrumLine(magnitudes);

      if (rms < 0.005) {
        _smoothedFrequency = null;
        return TunerResult(
          note: '--',
          frequency: 0.0,
          targetFrequency: 0.0,
          cents: 0.0,
          rms: rms,
          spectrumLine: spectrumLine,
        );
      }

      // 6. Find peak in the spectrum (excluding DC and higher half due to symmetry)
      final minIdx = (50 * bufferSize / sampleRate).floor();
      final maxIdx = (1000 * bufferSize / sampleRate).ceil();

      final searchRange = magnitudes.slice([
        Slice(start: minIdx, stop: maxIdx),
      ]);
      final peakIdxInSlice = argmax(searchRange) as int;
      final peakIdx = minIdx + peakIdxInSlice;

      // Denoising: SNR (Signal-to-Noise Ratio) & Sharpness Thresholds
      // Guitar plucks produce very narrow, high-energy spectral peaks compared to broadband room noise.
      final double meanMag = mean(searchRange).scalar as double;
      final magsList = magnitudes.toList();
      final double peakMag = magsList[peakIdx];

      bool isSharp = true;
      if (peakIdx > minIdx && peakIdx < maxIdx - 1) {
        final double leftMag = magsList[peakIdx - 1];
        final double rightMag = magsList[peakIdx + 1];
        if (peakMag < 1.5 * (leftMag + rightMag) / 2.0) {
          isSharp = false;
        }
      }

      if (peakMag < 4.0 * meanMag || !isSharp) {
        _smoothedFrequency = null;
        return TunerResult(
          note: '--',
          frequency: 0.0,
          targetFrequency: 0.0,
          cents: 0.0,
          rms: rms,
          spectrumLine: spectrumLine,
        );
      }

      // 7. Parabolic interpolation for more accurate peak frequency
      double refinedPeakIdx = peakIdx.toDouble();
      if (peakIdx > 0 && peakIdx < magnitudes.shape[0] - 1) {
        final magsList = magnitudes.toList();
        final y1 = magsList[peakIdx - 1];
        final y2 = magsList[peakIdx];
        final y3 = magsList[peakIdx + 1];

        final denom = y1 - 2 * y2 + y3;
        if (denom != 0) {
          refinedPeakIdx = peakIdx + 0.5 * (y1 - y3) / denom;
        }
      }

      final freq = refinedPeakIdx * sampleRate / bufferSize;

      // Exponential Moving Average (EMA) frequency smoothing (filters out measurement jitter)
      if (_smoothedFrequency == null) {
        _smoothedFrequency = freq;
      } else {
        const alpha = 0.25; // Smoothness vs response delay factor
        _smoothedFrequency = alpha * freq + (1.0 - alpha) * _smoothedFrequency!;
      }
      final displayFreq = _smoothedFrequency!;

      // 8. Find closest note
      String closestNote = 'Unknown';
      double minDiff = double.infinity;
      double targetFreq = 0.0;

      guitarNotes.forEach((note, noteFreq) {
        final diff = (displayFreq - noteFreq).abs();
        if (diff < minDiff) {
          minDiff = diff;
          closestNote = note;
          targetFreq = noteFreq;
        }
      });

      // 9. Calculate cents difference
      final cents = 1200 * math.log(displayFreq / targetFreq) / math.log(2);

      return TunerResult(
        note: closestNote,
        frequency: displayFreq,
        targetFrequency: targetFreq,
        cents: cents,
        rms: rms,
        spectrumLine: spectrumLine,
      );
    });
  }

  String _generateSpectrumLine(NDArray magnitudes) {
    const plotWidth = 80;
    const chars = ' .:-=+*#%@';

    // Use frequencies from 50Hz to 1000Hz
    final minIdx = (50 * bufferSize / sampleRate).floor();
    final maxIdx = (1000 * bufferSize / sampleRate).ceil();
    final range = maxIdx - minIdx;

    final result = StringBuffer();
    final magnitudesList = magnitudes.toList();
    for (var i = 0; i < plotWidth; i++) {
      final start = minIdx + (i * range / plotWidth).floor();
      final end = minIdx + ((i + 1) * range / plotWidth).floor();

      double sum = 0.0;
      int count = 0;
      for (var j = start; j < end && j < magnitudesList.length; j++) {
        sum += magnitudesList[j] as num;
        count++;
      }

      final avg = count > 0 ? sum / count : 0.0;
      // Heuristic scaling for better visual contrast
      final int level = (avg * 50).round().clamp(0, chars.length - 1);
      result.write(chars[level]);
    }
    return result.toString();
  }

  void dispose() {
    _window.dispose();
  }
}
