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

// ignore_for_file: non_constant_identifier_names

@ffi.DefaultAsset('package:guitar_tuner/tuner_bridge')
library;

import 'dart:ffi' as ffi;
import 'package:ndarray/ndarray.dart';

@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.Int32, ffi.Int32)>()
external ffi.Pointer<ffi.Void> tuner_init(int sampleRate, int ringBufferSize);

@ffi.Native<
  ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>, ffi.Int32)
>()
external void tuner_get_samples(
  ffi.Pointer<ffi.Void> context,
  ffi.Pointer<ffi.Float> output,
  int count,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void tuner_close(ffi.Pointer<ffi.Void> context);

final class AudioCapture {
  final int sampleRate;
  final int bufferSize;

  ffi.Pointer<ffi.Void> _context = ffi.nullptr;

  AudioCapture({this.sampleRate = 44100, this.bufferSize = 4096});

  void open() {
    // Initialize with a ring buffer 4x the size of the request buffer
    _context = tuner_init(sampleRate, bufferSize * 4);
    if (_context == ffi.nullptr) {
      throw Exception('Failed to initialize audio capture via miniaudio');
    }
  }

  /// Reads the current audio samples into the provided [buffer].
  ///
  /// The [buffer] must have [DType.float32] and its first dimension must be at least [bufferSize].
  void read(NDArray<Float32> buffer) {
    if (_context == ffi.nullptr) return;

    if (buffer.dtype != DType.float32) {
      throw ArgumentError('Buffer must be DType.float32');
    }
    if (buffer.shape[0] < bufferSize) {
      throw ArgumentError(
        'Buffer is too small (expected at least $bufferSize)',
      );
    }

    tuner_get_samples(_context, buffer.pointer.cast(), bufferSize);
  }

  void close() {
    if (_context != ffi.nullptr) {
      tuner_close(_context);
      _context = ffi.nullptr;
    }
  }
}
