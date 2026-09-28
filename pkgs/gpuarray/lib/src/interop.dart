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

import 'package:ndarray/ndarray.dart' as nd;

import 'device.dart';
import 'gpu_array.dart';

/// Extension on [nd.NDArray] for transferring host tensors to a [GpuDevice].
extension NDArrayGpuInterop<T extends DTypeTag> on nd.NDArray<T> {
  /// Transfers this host [nd.NDArray] to a contiguous [GpuArray] on [device].
  ///
  /// Uses a single contiguous bulk memory transfer when this array is already
  /// C-contiguous, or materializes a temporary C-contiguous copy via native C
  /// strided copy before uploading.
  ///
  /// It is an error if this array has been disposed.
  GpuArray<T> toGpu({GpuDevice? device, bool requiresGrad = false}) {
    if (isDisposed) {
      throw StateError('Cannot transfer a disposed NDArray to GPU.');
    }
    return GpuArray<T>.fromNDArray(
      this,
      device: device,
      requiresGrad: requiresGrad,
    );
  }
}

/// Extension on [GpuArray] for downloading GPU tensors to host [nd.NDArray] memory.
extension GpuArrayNDArrayInterop<T extends DTypeTag> on GpuArray<T> {
  /// Transfers this [GpuArray] to a newly allocated C-contiguous host [nd.NDArray].
  nd.NDArray<T> toHostNDArray() => toNDArray();
}
