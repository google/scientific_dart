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

import '../../dtype.dart';

/// WGSL data types supported in compute shaders.
enum WgslDType {
  /// 32-bit IEEE-754 floating-point (`f32`).
  float32('f32', 4),

  /// 16-bit IEEE-754 half-precision floating-point (`f16`).
  float16('f16', 2),

  /// 32-bit signed two's-complement integer (`i32`).
  int32('i32', 4),

  /// 32-bit unsigned integer (`u32`).
  uint32('u32', 4),

  /// WGSL logical boolean (`bool`).
  boolean('bool', 4);

  /// WGSL type identifier string.
  final String wgslType;

  /// Storage size in bytes per element.
  final int byteSize;

  const WgslDType(this.wgslType, this.byteSize);

  /// Whether [dtype] has a direct 32-bit storage representation in standard
  /// WebGPU storage buffers (`f32`, `i32`, `u32`).
  static bool isNativelySupportedStorageDType(DType dtype) => switch (dtype) {
    DType.float32 || DType.int32 || DType.uint32 => true,
    _ => false,
  };

  /// Converts a [DType] to its corresponding [WgslDType].
  static WgslDType fromDType(DType dtype) => switch (dtype) {
    DType.float32 => WgslDType.float32,
    DType.float16 || DType.bfloat16 => WgslDType.float16,
    DType.int32 || DType.int16 || DType.int8 => WgslDType.int32,
    DType.uint32 ||
    DType.uint16 ||
    DType.uint8 ||
    DType.boolean => WgslDType.uint32,
    DType.float64 ||
    DType.int64 ||
    DType.uint64 ||
    DType.complex64 ||
    DType.complex128 => WgslDType.float32,
  };
}

/// Buffer access qualifier for storage buffers in WGSL.
enum WgslBufferAccess {
  /// Read-only storage buffer (`var<storage, read>`).
  read('read'),

  /// Read-write storage buffer (`var<storage, read_write>`).
  readWrite('read_write');

  /// WGSL storage access qualifier string.
  final String qualifier;

  const WgslBufferAccess(this.qualifier);
}

/// Represents a resource binding in a WGSL compute shader.
final class WgslBinding {
  /// Bind group index (`@group(...)`).
  final int group;

  /// Binding slot index (`@binding(...)`).
  final int binding;

  /// Variable identifier in the WGSL shader.
  final String name;

  /// Element data type of the bound buffer.
  final WgslDType dtype;

  /// Storage access mode (`read` or `read_write`).
  final WgslBufferAccess access;

  /// Whether this binding is a `var<uniform>` buffer rather than `var<storage>`.
  final bool isUniform;

  /// Whether the storage buffer holds a runtime-sized `array<T>`.
  final bool isArray;

  /// Optional custom WGSL struct type name.
  final String? customTypeName;

  /// Creates a [WgslBinding].
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
  /// Threads along the X dimension.
  final int x;

  /// Threads along the Y dimension.
  final int y;

  /// Threads along the Z dimension.
  final int z;

  /// Creates a [WgslWorkgroupSize].
  const WgslWorkgroupSize(this.x, [this.y = 1, this.z = 1]);

  /// Standard 1D workgroup of 256 threads.
  static const WgslWorkgroupSize linear1D = WgslWorkgroupSize(256, 1, 1);

  /// Compact 1D workgroup of 64 threads.
  static const WgslWorkgroupSize linear64 = WgslWorkgroupSize(64, 1, 1);

  /// Standard 2D tiled workgroup of 16x16 threads (256 total).
  static const WgslWorkgroupSize tiled2D = WgslWorkgroupSize(16, 16, 1);

  /// Compact 2D tiled workgroup of 8x8 threads (64 total).
  static const WgslWorkgroupSize tiled8x8 = WgslWorkgroupSize(8, 8, 1);

  /// Total number of threads per workgroup (`x * y * z`).
  int get totalThreads => x * y * z;

  /// Formats the `@workgroup_size(...)` WGSL attribute string.
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
  /// Maximum workgroups allowed per dimension by the WebGPU specification (`65535`).
  static const int maxWorkgroupsPerDimension = 65535;

  /// Number of workgroups along the X dimension.
  final int workgroupsX;

  /// Number of workgroups along the Y dimension.
  final int workgroupsY;

  /// Number of workgroups along the Z dimension.
  final int workgroupsZ;

  /// Creates a [WgslDispatch].
  const WgslDispatch({
    required this.workgroupsX,
    this.workgroupsY = 1,
    this.workgroupsZ = 1,
  });

  /// Total number of workgroups dispatched across all dimensions.
  int get totalWorkgroups => workgroupsX * workgroupsY * workgroupsZ;

  @override
  String toString() => 'WgslDispatch($workgroupsX, $workgroupsY, $workgroupsZ)';
}

/// Represents a compiled WGSL compute shader module with metadata.
final class WgslShaderModule {
  /// Descriptive kernel name.
  final String name;

  /// WGSL shader source code.
  final String code;

  /// Compute entry point function name.
  final String entryPoint;

  /// Workgroup size configuration.
  final WgslWorkgroupSize workgroupSize;

  /// Unmodifiable resource bindings declared by this shader.
  final List<WgslBinding> bindings;

  /// Unmodifiable compiler/dispatch metadata.
  final Map<String, Object?> metadata;

  /// Creates a [WgslShaderModule] with unmodifiable copies of [bindings] and [metadata].
  WgslShaderModule({
    required this.name,
    required this.code,
    this.entryPoint = 'main',
    this.workgroupSize = WgslWorkgroupSize.linear1D,
    List<WgslBinding> bindings = const [],
    Map<String, Object?> metadata = const {},
  }) : bindings = List<WgslBinding>.unmodifiable(bindings),
       metadata = Map<String, Object?>.unmodifiable(metadata);

  /// Calculates the dispatch grid for [totalElements] elements, folding across
  /// `workgroupsY` when the required workgroup count exceeds
  /// [WgslDispatch.maxWorkgroupsPerDimension] (`65535`).
  WgslDispatch calculateDispatch1D(int totalElements) {
    if (totalElements <= 0) {
      return const WgslDispatch(workgroupsX: 1);
    }
    final totalGroups =
        (totalElements + workgroupSize.x - 1) ~/ workgroupSize.x;
    if (totalGroups <= WgslDispatch.maxWorkgroupsPerDimension) {
      return WgslDispatch(workgroupsX: totalGroups);
    }
    const maxDim = WgslDispatch.maxWorkgroupsPerDimension;
    final groupsY = math.min(maxDim, (totalGroups + maxDim - 1) ~/ maxDim);
    final groupsX = math.min(maxDim, (totalGroups + groupsY - 1) ~/ groupsY);
    return WgslDispatch(workgroupsX: groupsX, workgroupsY: groupsY);
  }

  /// Calculates the 2D dispatch configuration for matrix dimensions [width] x [height].
  WgslDispatch calculateDispatch2D(int width, int height) {
    final countX = math.min(
      WgslDispatch.maxWorkgroupsPerDimension,
      math.max(1, (width + workgroupSize.x - 1) ~/ workgroupSize.x),
    );
    final countY = math.min(
      WgslDispatch.maxWorkgroupsPerDimension,
      math.max(1, (height + workgroupSize.y - 1) ~/ workgroupSize.y),
    );
    return WgslDispatch(workgroupsX: countX, workgroupsY: countY);
  }

  /// Calculates 3D dispatch for convolution or batched operations.
  WgslDispatch calculateDispatch3D(int dimX, int dimY, int dimZ) {
    final countX = math.min(
      WgslDispatch.maxWorkgroupsPerDimension,
      math.max(1, (dimX + workgroupSize.x - 1) ~/ workgroupSize.x),
    );
    final countY = math.min(
      WgslDispatch.maxWorkgroupsPerDimension,
      math.max(1, (dimY + workgroupSize.y - 1) ~/ workgroupSize.y),
    );
    final countZ = math.min(
      WgslDispatch.maxWorkgroupsPerDimension,
      math.max(1, (dimZ + workgroupSize.z - 1) ~/ workgroupSize.z),
    );
    return WgslDispatch(
      workgroupsX: countX,
      workgroupsY: countY,
      workgroupsZ: countZ,
    );
  }

  @override
  String toString() =>
      'WgslShaderModule(name: "$name", entryPoint: "$entryPoint", workgroup: $workgroupSize, bindings: ${bindings.length})';
}
