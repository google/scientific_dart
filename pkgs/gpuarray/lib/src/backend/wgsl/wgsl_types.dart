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

import '../../dtype.dart';

/// WGSL data types supported in compute shaders.
enum WgslDType {
  float32('f32', 4),
  float16('f16', 2),
  int32('i32', 4),
  uint32('u32', 4),
  boolean('bool', 4);

  final String wgslType;
  final int byteSize;

  const WgslDType(this.wgslType, this.byteSize);

  /// Converts a [DType] to its corresponding [WgslDType].
  static WgslDType fromDType(DType dtype) {
    switch (dtype) {
      case DType.float32:
        return WgslDType.float32;
      case DType.float16:
      case DType.bfloat16:
        return WgslDType.float16;
      case DType.int32:
      case DType.int16:
      case DType.int8:
        return WgslDType.int32;
      case DType.uint32:
      case DType.uint16:
      case DType.uint8:
        return WgslDType.uint32;
      case DType.boolean:
        return WgslDType
            .uint32; // Standard WebGPU storage representation for bool
      case DType.float64:
      case DType.int64:
      case DType.uint64:
      case DType.complex64:
      case DType.complex128:
        // WebGPU standard currently primarily targets 32-bit floats/ints, fallback to f32
        return WgslDType.float32;
    }
  }
}

/// Buffer access qualifier for storage buffers in WGSL.
enum WgslBufferAccess {
  read('read'),
  readWrite('read_write');

  final String qualifier;
  const WgslBufferAccess(this.qualifier);
}

/// Represents a resource binding in a WGSL compute shader.
final class WgslBinding {
  final int group;
  final int binding;
  final String name;
  final WgslDType dtype;
  final WgslBufferAccess access;
  final bool isUniform;
  final bool isArray;
  final String? customTypeName;

  const WgslBinding({
    required this.group,
    required this.binding,
    required this.name,
    this.dtype = WgslDType.float32,
    this.access = WgslBufferAccess.read,
    this.isUniform = false,
    this.isArray = true,
    this.customTypeName,
  });

  /// Generates the WGSL variable declaration line.
  String toWgslDeclaration() {
    if (isUniform) {
      final typeStr = customTypeName ?? dtype.wgslType;
      return '@group($group) @binding($binding) var<uniform> $name: $typeStr;';
    }
    final typeStr =
        customTypeName ??
        (isArray ? 'array<${dtype.wgslType}>' : dtype.wgslType);
    return '@group($group) @binding($binding) var<storage, ${access.qualifier}> $name: $typeStr;';
  }

  @override
  String toString() =>
      'WgslBinding(group: $group, binding: $binding, name: "$name", type: ${customTypeName ?? dtype.wgslType})';
}

/// Workgroup size dimensions for a compute shader.
final class WgslWorkgroupSize {
  final int x;
  final int y;
  final int z;

  const WgslWorkgroupSize(this.x, [this.y = 1, this.z = 1]);

  static const WgslWorkgroupSize linear1D = WgslWorkgroupSize(256, 1, 1);
  static const WgslWorkgroupSize linear64 = WgslWorkgroupSize(64, 1, 1);
  static const WgslWorkgroupSize tiled2D = WgslWorkgroupSize(16, 16, 1);
  static const WgslWorkgroupSize tiled8x8 = WgslWorkgroupSize(8, 8, 1);

  int get totalThreads => x * y * z;

  /// Returns the `@workgroup_size(...)` attribute string.
  String toAttribute() {
    if (z == 1 && y == 1) {
      return '@workgroup_size($x)';
    } else if (z == 1) {
      return '@workgroup_size($x, $y)';
    }
    return '@workgroup_size($x, $y, $z)';
  }

  @override
  String toString() => 'WgslWorkgroupSize($x, $y, $z)';
}

/// Dispatch dimensions for launching a compute shader.
final class WgslDispatch {
  final int workgroupsX;
  final int workgroupsY;
  final int workgroupsZ;

  const WgslDispatch({
    required this.workgroupsX,
    this.workgroupsY = 1,
    this.workgroupsZ = 1,
  });

  @override
  String toString() => 'WgslDispatch($workgroupsX, $workgroupsY, $workgroupsZ)';
}

/// Represents a compiled WGSL compute shader module with metadata.
final class WgslShaderModule {
  final String name;
  final String code;
  final String entryPoint;
  final WgslWorkgroupSize workgroupSize;
  final List<WgslBinding> bindings;
  final Map<String, dynamic> metadata;

  const WgslShaderModule({
    required this.name,
    required this.code,
    this.entryPoint = 'main',
    this.workgroupSize = WgslWorkgroupSize.linear1D,
    this.bindings = const [],
    this.metadata = const {},
  });

  /// Calculates the 1D dispatch configuration for a given number of [totalElements].
  WgslDispatch calculateDispatch1D(int totalElements) {
    final count = (totalElements + workgroupSize.x - 1) ~/ workgroupSize.x;
    return WgslDispatch(workgroupsX: count > 0 ? count : 1);
  }

  /// Calculates the 2D dispatch configuration for matrix dimensions [width] x [height].
  WgslDispatch calculateDispatch2D(int width, int height) {
    final countX = (width + workgroupSize.x - 1) ~/ workgroupSize.x;
    final countY = (height + workgroupSize.y - 1) ~/ workgroupSize.y;
    return WgslDispatch(
      workgroupsX: countX > 0 ? countX : 1,
      workgroupsY: countY > 0 ? countY : 1,
    );
  }

  /// Calculates 3D dispatch for convolution or batched operations.
  WgslDispatch calculateDispatch3D(int dimX, int dimY, int dimZ) {
    final countX = (dimX + workgroupSize.x - 1) ~/ workgroupSize.x;
    final countY = (dimY + workgroupSize.y - 1) ~/ workgroupSize.y;
    final countZ = (dimZ + workgroupSize.z - 1) ~/ workgroupSize.z;
    return WgslDispatch(
      workgroupsX: countX > 0 ? countX : 1,
      workgroupsY: countY > 0 ? countY : 1,
      workgroupsZ: countZ > 0 ? countZ : 1,
    );
  }

  @override
  String toString() =>
      'WgslShaderModule(name: "$name", entryPoint: "$entryPoint", workgroup: $workgroupSize, bindings: ${bindings.length})';
}
