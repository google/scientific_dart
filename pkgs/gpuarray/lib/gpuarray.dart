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

/// GPU-accelerated N-dimensional array computing for Dart.
///
/// This package provides high-performance tensor computing on GPU devices
/// (`GpuArray`), device and buffer memory management (`GpuDevice`, `GpuBuffer`),
/// and seamless, zero-intermediate-copy interoperability with `package:ndarray`.
library;

export 'src/exceptions.dart';
export 'src/dtype.dart' hide Float16Utils;
export 'src/buffer.dart';
export 'src/device.dart';
export 'src/gpu_array.dart' hide ResourceScope, ScopedResource, tanh;
export 'src/slice.dart';
export 'src/operations/indexing.dart';
export 'src/operations/manipulation.dart';
export 'src/autograd/autograd_core.dart'
    show GradFn, LossReduction, enableGrad, isGradEnabled, noGrad;
export 'src/interop.dart';
export 'src/backend/backend.dart' show GpuBackend, GpuDeviceType;
export 'src/backend/memory_pool.dart' show GpuMemoryPool;
export 'src/backend/webgpu_backend.dart';
