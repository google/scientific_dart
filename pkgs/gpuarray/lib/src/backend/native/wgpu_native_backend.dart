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
import 'dart:math' as math;
import 'package:ffi/ffi.dart';
import '../../buffer.dart';
import '../../device.dart';
import '../../exceptions.dart';
import '../backend.dart';
import '../wgsl/wgsl_types.dart';
import 'wgpu_bindings.dart';

/// Record of a dispatched compute shader pass on a [WgpuNativeBackend].
final class MockDispatchRecord {
  /// The compiled WGSL shader module executed in this dispatch.
  final WgslShaderModule shaderModule;

  /// Unmodifiable list of GPU buffers bound to the compute pass.
  final List<GpuBuffer> buffers;

  /// Optional 32-bit uniform words uploaded for this dispatch.
  final List<int>? uniforms;

  /// Number of workgroups dispatched along the X dimension.
  final int workgroupsX;

  /// Number of workgroups dispatched along the Y dimension.
  final int workgroupsY;

  /// Number of workgroups dispatched along the Z dimension.
  final int workgroupsZ;

  /// Timestamp when the dispatch was recorded.
  final DateTime timestamp;

  /// Creates a [MockDispatchRecord] for a compute pipeline dispatch.
  const MockDispatchRecord({
    required this.shaderModule,
    required this.buffers,
    this.uniforms,
    required this.workgroupsX,
    this.workgroupsY = 1,
    this.workgroupsZ = 1,
    required this.timestamp,
  });

  @override
  String toString() =>
      'MockDispatchRecord(shader: "${shaderModule.name}", workgroups: ($workgroupsX, $workgroupsY, $workgroupsZ), buffers: ${buffers.length})';
}

/// Native WebGPU hardware driver backend backed by `libwgpu_native` (Vulkan, Metal, DirectX 12).
final class WgpuNativeBackend extends GpuBackend {
  /// High-level `@ffi.Native` WebGPU bindings helper.
  final WgpuNativeBindings bindings;

  /// Native `WGPUInstance` handle.
  final ffi.Pointer<ffi.Void> instance;

  /// Native `WGPUAdapter` handle.
  final ffi.Pointer<ffi.Void> adapter;

  /// Native `WGPUDevice` handle.
  final ffi.Pointer<ffi.Void> device;

  /// Native `WGPUQueue` handle.
  final ffi.Pointer<ffi.Void> queue;

  static const int _maxDispatchHistory = 1024;

  final Map<int, int> _allocatedBufferSizes = {};
  final Map<String, ffi.Pointer<ffi.Void>> _shaderModules = {};
  final Map<String, ffi.Pointer<ffi.Void>> _pipelines = {};
  final Map<String, ffi.Pointer<ffi.Void>> _bindGroupLayouts = {};
  final Map<String, bool> _pipelineHasUniform = {};
  final List<MockDispatchRecord> _dispatches = [];

  ffi.Pointer<ffi.Void> _uniformBuffer = ffi.nullptr;
  int _uniformBufferCapacity = 0;
  ffi.Pointer<ffi.Uint32> _uniformHostMemory = ffi.nullptr;

  ffi.Pointer<ffi.Void> _stagingBuffer = ffi.nullptr;
  int _stagingBufferCapacity = 0;

  bool _isDisposed = false;

  WgpuNativeBackend._({
    required this.bindings,
    required this.instance,
    required this.adapter,
    required this.device,
    required this.queue,
  });

  /// Synchronously initializes a native WebGPU device backend.
  factory WgpuNativeBackend({
    WgpuNativeBindings bindings = const WgpuNativeBindings(),
  }) => WgpuNativeBackend.createSync(bindings: bindings);

  /// Synchronously creates and initializes a native WebGPU device backend.
  ///
  /// Throws a [GpuDeviceException] if no WebGPU adapter or device can be acquired.
  static WgpuNativeBackend createSync({
    WgpuNativeBindings bindings = const WgpuNativeBindings(),
  }) {
    if (!bindings.isAvailable) {
      throw const GpuDeviceException(
        'WebGPU native bindings (libwgpu_native) are not available.',
      );
    }
    final instance = bindings.createInstance();
    if (instance == ffi.nullptr) {
      throw const GpuDeviceException('Failed to create WGPUInstance.');
    }
    final adapter = bindings.requestAdapterSync(instance);
    final device = bindings.requestDeviceSync(
      instance,
      adapter,
      label: 'ScientificDart_WGPUDevice',
    );
    final queue = bindings.deviceGetQueue(device);

    return WgpuNativeBackend._(
      bindings: bindings,
      instance: instance,
      adapter: adapter,
      device: device,
      queue: queue,
    );
  }

  /// Creates and initializes a native WebGPU device backend.
  ///
  /// Throws a [GpuDeviceException] if no WebGPU adapter or device can be acquired.
  static Future<WgpuNativeBackend> create({
    WgpuNativeBindings bindings = const WgpuNativeBindings(),
  }) async {
    return createSync(bindings: bindings);
  }

  @override
  GpuDeviceType get deviceType => GpuDeviceType.webgpu;

  /// Whether this driver backend has been disposed.
  bool get isDisposed => _isDisposed;

  /// The number of compiled and cached compute pipelines.
  int get pipelineCacheSize => _pipelines.length;

  /// Formatted log of all dispatched compute shader operations.
  List<String> get dispatchLog => _dispatches
      .map(
        (d) =>
            '${d.shaderModule.name}(${d.workgroupsX}, ${d.workgroupsY}, ${d.workgroupsZ})',
      )
      .toList();

  /// Dispatched kernel calls recorded during execution.
  List<MockDispatchRecord> get dispatches => List.unmodifiable(_dispatches);

  /// Clears recorded dispatch history.
  void clearDispatches() => _dispatches.clear();

  /// Total count of active low-level `WGPUBuffer` allocations.
  int get activeAllocationCount => _allocatedBufferSizes.length;

  @override
  ffi.Pointer<ffi.Void> allocateBuffer(
    int sizeInBytes, {
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
  }) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot allocate buffer on disposed WgpuNativeBackend.',
      );
    }
    RangeError.checkNotNegative(sizeInBytes, 'sizeInBytes');
    if (sizeInBytes == 0) return ffi.nullptr;

    final alignedSize = math.max(16, (sizeInBytes + 3) & ~3);
    final gpuBuffer = bindings.createBuffer(
      device,
      size: alignedSize,
      usage:
          WGPUBufferUsage.storage |
          WGPUBufferUsage.copySrc |
          WGPUBufferUsage.copyDst |
          WGPUBufferUsage.uniform |
          usage.value,
      label: 'GpuBuffer',
    );
    _allocatedBufferSizes[gpuBuffer.address] = alignedSize;
    return gpuBuffer;
  }

  @override
  void freeBuffer(ffi.Pointer<ffi.Void> handle, int sizeInBytes) {
    if (handle == ffi.nullptr) return;
    if (_allocatedBufferSizes.remove(handle.address) != null) {
      bindings.bufferDestroy(handle);
      bindings.bufferRelease(handle);
    }
  }

  void _readAlignedFromGpu(
    ffi.Pointer<ffi.Void> gpuBuffer,
    int alignedOffset,
    int alignedSize,
    ffi.Pointer<ffi.Uint8> destination,
  ) {
    if (_stagingBuffer == ffi.nullptr || _stagingBufferCapacity < alignedSize) {
      if (_stagingBuffer != ffi.nullptr) {
        bindings.bufferDestroy(_stagingBuffer);
        bindings.bufferRelease(_stagingBuffer);
      }
      _stagingBufferCapacity = math.max(alignedSize, 4096);
      _stagingBuffer = bindings.createBuffer(
        device,
        size: _stagingBufferCapacity,
        usage: WGPUBufferUsage.mapRead | WGPUBufferUsage.copyDst,
        label: 'shared_staging_readback',
      );
    }

    final encoder = bindings.createCommandEncoder(device);
    bindings.commandEncoderCopyBufferToBuffer(
      encoder,
      gpuBuffer,
      alignedOffset,
      _stagingBuffer,
      0,
      alignedSize,
    );
    final commandBuffer = bindings.commandEncoderFinish(encoder);
    bindings.queueSubmit(queue, [commandBuffer]);
    bindings.commandBufferRelease(commandBuffer);
    bindings.commandEncoderRelease(encoder);

    bindings.bufferMapSync(
      instance,
      _stagingBuffer,
      device: device,
      mode: WGPUMapMode.read,
      offset: 0,
      size: alignedSize,
    );
    bindings.devicePoll(device, wait: true);

    final mappedPointer = bindings.bufferGetMappedRange(
      _stagingBuffer,
      offset: 0,
      size: alignedSize,
    );
    if (mappedPointer != ffi.nullptr) {
      destination
          .asTypedList(alignedSize)
          .setAll(0, mappedPointer.cast<ffi.Uint8>().asTypedList(alignedSize));
      bindings.bufferUnmap(_stagingBuffer);
    }
  }

  @override
  void clearBuffer(GpuBuffer buffer, {int offset = 0, int? bytes}) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot clear buffer on disposed WgpuNativeBackend.',
      );
    }
    super.clearBuffer(buffer, offset: offset, bytes: bytes);
    final resolvedBytes = bytes ?? (buffer.sizeInBytes - offset);
    if (resolvedBytes == 0 || buffer.rawNativeHandle == ffi.nullptr) return;

    if (offset % 4 == 0 && resolvedBytes % 4 == 0) {
      final encoder = bindings.createCommandEncoder(device);
      bindings.commandEncoderClearBuffer(
        encoder,
        buffer.rawNativeHandle,
        offset,
        resolvedBytes,
      );
      final commandBuffer = bindings.commandEncoderFinish(encoder);
      bindings.queueSubmit(queue, [commandBuffer]);
      bindings.commandBufferRelease(commandBuffer);
      bindings.commandEncoderRelease(encoder);
      bindings.devicePoll(device, wait: true);
      return;
    }

    using((arena) {
      final zeros = arena<ffi.Uint8>(resolvedBytes);
      copyHostToBuffer(zeros, buffer, resolvedBytes, offset: offset);
    });
  }

  @override
  void copyHostToBuffer(
    ffi.Pointer<ffi.Uint8> src,
    GpuBuffer dst,
    int bytes, {
    int offset = 0,
  }) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot copy to buffer on disposed WgpuNativeBackend.',
      );
    }
    super.copyHostToBuffer(src, dst, bytes, offset: offset);
    if (bytes == 0 || dst.nativeHandle == ffi.nullptr) return;

    if (offset % 4 == 0 && bytes % 4 == 0) {
      bindings.queueWriteBuffer(
        queue,
        dst.nativeHandle,
        bufferOffset: offset,
        data: src.cast<ffi.Void>(),
        size: bytes,
      );
      bindings.devicePoll(device, wait: true);
      return;
    }

    final alignedOffset = offset & ~3;
    final alignedEnd = (offset + bytes + 3) & ~3;
    final alignedSize = alignedEnd - alignedOffset;
    final prefixLength = offset - alignedOffset;
    final suffixLength = alignedEnd - (offset + bytes);

    using((arena) {
      final staging = arena<ffi.Uint8>(alignedSize);
      if (prefixLength > 0) {
        _readAlignedFromGpu(dst.nativeHandle, alignedOffset, 4, staging);
      }
      if (suffixLength > 0 && (alignedSize > 4 || prefixLength == 0)) {
        _readAlignedFromGpu(
          dst.nativeHandle,
          alignedEnd - 4,
          4,
          staging + (alignedSize - 4),
        );
      }
      (staging + prefixLength)
          .asTypedList(bytes)
          .setAll(0, src.asTypedList(bytes));
      bindings.queueWriteBuffer(
        queue,
        dst.nativeHandle,
        bufferOffset: alignedOffset,
        data: staging.cast<ffi.Void>(),
        size: alignedSize,
      );
      bindings.devicePoll(device, wait: true);
    });
  }

  @override
  void copyBufferToHost(
    GpuBuffer src,
    ffi.Pointer<ffi.Uint8> dst,
    int bytes, {
    int offset = 0,
  }) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot copy from buffer on disposed WgpuNativeBackend.',
      );
    }
    super.copyBufferToHost(src, dst, bytes, offset: offset);
    if (bytes == 0 || src.nativeHandle == ffi.nullptr) return;

    final alignedOffset = offset & ~3;
    final alignedEnd = (offset + bytes + 3) & ~3;
    final alignedSize = alignedEnd - alignedOffset;
    final prefixLength = offset - alignedOffset;

    if (prefixLength == 0 && alignedSize == bytes) {
      _readAlignedFromGpu(src.nativeHandle, alignedOffset, alignedSize, dst);
      return;
    }

    using((arena) {
      final staging = arena<ffi.Uint8>(alignedSize);
      _readAlignedFromGpu(
        src.nativeHandle,
        alignedOffset,
        alignedSize,
        staging,
      );
      dst
          .asTypedList(bytes)
          .setAll(0, (staging + prefixLength).asTypedList(bytes));
    });
  }

  @override
  void copyBufferToBuffer(
    GpuBuffer src,
    GpuBuffer dst,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot copy between buffers on disposed WgpuNativeBackend.',
      );
    }
    super.copyBufferToBuffer(
      src,
      dst,
      bytes,
      srcOffset: srcOffset,
      dstOffset: dstOffset,
    );
    if (bytes == 0 ||
        src.nativeHandle == ffi.nullptr ||
        dst.nativeHandle == ffi.nullptr) {
      return;
    }

    if (srcOffset % 4 == 0 &&
        dstOffset % 4 == 0 &&
        bytes % 4 == 0 &&
        src.nativeHandle != dst.nativeHandle) {
      final encoder = bindings.createCommandEncoder(device);
      bindings.commandEncoderCopyBufferToBuffer(
        encoder,
        src.nativeHandle,
        srcOffset,
        dst.nativeHandle,
        dstOffset,
        bytes,
      );
      final commandBuffer = bindings.commandEncoderFinish(encoder);
      bindings.queueSubmit(queue, [commandBuffer]);
      bindings.commandBufferRelease(commandBuffer);
      bindings.commandEncoderRelease(encoder);
      bindings.devicePoll(device, wait: true);
      return;
    }

    using((arena) {
      final hostBounce = arena<ffi.Uint8>(bytes);
      copyBufferToHost(src, hostBounce, bytes, offset: srcOffset);
      copyHostToBuffer(hostBounce, dst, bytes, offset: dstOffset);
    });
  }

  @override
  void dispatchComputePipeline({
    required WgslShaderModule shaderModule,
    required List<GpuBuffer> buffers,
    List<int>? uniforms,
    required int workgroupsX,
    int workgroupsY = 1,
    int workgroupsZ = 1,
  }) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot dispatch compute pipeline on disposed backend.',
      );
    }
    if (buffers.any((b) => b.isDisposed)) {
      throw const GpuMemoryException(
        'Cannot dispatch pipeline with disposed buffers.',
      );
    }
    if (workgroupsX <= 0 ||
        workgroupsX > WgslDispatch.maxWorkgroupsPerDimension) {
      throw ArgumentError.value(
        workgroupsX,
        'workgroupsX',
        'Must be between 1 and ${WgslDispatch.maxWorkgroupsPerDimension}.',
      );
    }
    if (workgroupsY <= 0 ||
        workgroupsY > WgslDispatch.maxWorkgroupsPerDimension) {
      throw ArgumentError.value(
        workgroupsY,
        'workgroupsY',
        'Must be between 1 and ${WgslDispatch.maxWorkgroupsPerDimension}.',
      );
    }
    if (workgroupsZ <= 0 ||
        workgroupsZ > WgslDispatch.maxWorkgroupsPerDimension) {
      throw ArgumentError.value(
        workgroupsZ,
        'workgroupsZ',
        'Must be between 1 and ${WgslDispatch.maxWorkgroupsPerDimension}.',
      );
    }

    if (_dispatches.length >= _maxDispatchHistory) {
      _dispatches.removeRange(0, _maxDispatchHistory ~/ 2);
    }
    _dispatches.add(
      MockDispatchRecord(
        shaderModule: shaderModule,
        buffers: List.unmodifiable(buffers),
        uniforms: uniforms != null ? List.unmodifiable(uniforms) : null,
        workgroupsX: workgroupsX,
        workgroupsY: workgroupsY,
        workgroupsZ: workgroupsZ,
        timestamp: DateTime.now(),
      ),
    );

    final module = _shaderModules[shaderModule.code] ??= bindings
        .createShaderModule(
          device,
          shaderModule.code,
          label: shaderModule.name,
        );

    final pipelineKey =
        '${shaderModule.name}_${shaderModule.entryPoint}_${shaderModule.code.hashCode}';
    final pipeline = _pipelines[pipelineKey] ??= bindings.createComputePipeline(
      device,
      shaderModule: module,
      entryPoint: shaderModule.entryPoint,
      label: '${shaderModule.name}_pipeline',
    );

    final bindGroupLayout = _bindGroupLayouts[pipelineKey] ??= bindings
        .pipelineGetBindGroupLayout(pipeline, 0);
    final encoder = bindings.createCommandEncoder(device);

    final entries = <WgpuBindGroupEntryData>[];
    final clonedBuffers = <ffi.Pointer<ffi.Void>>[];

    try {
      for (var i = 0; i < buffers.length; i++) {
        final buffer = buffers[i];
        final handle = buffer.nativeHandle;
        if (handle == ffi.nullptr) {
          final emptyHandle = bindings.createBuffer(
            device,
            size: 16,
            usage:
                WGPUBufferUsage.storage |
                WGPUBufferUsage.copySrc |
                WGPUBufferUsage.copyDst |
                WGPUBufferUsage.uniform,
          );
          clonedBuffers.add(emptyHandle);
          entries.add(
            WgpuBindGroupEntryData(binding: i, buffer: emptyHandle, size: 16),
          );
          continue;
        }

        final alignedSize =
            _allocatedBufferSizes[handle.address] ??
            math.max(16, (buffer.allocatedBytes + 3) & ~3);

        var isAliasedLater = false;
        for (var j = i + 1; j < buffers.length; j++) {
          if (buffers[j].nativeHandle == handle) {
            isAliasedLater = true;
            break;
          }
        }

        if (isAliasedLater) {
          final clonedHandle = bindings.createBuffer(
            device,
            size: alignedSize,
            usage:
                WGPUBufferUsage.storage |
                WGPUBufferUsage.copySrc |
                WGPUBufferUsage.copyDst |
                WGPUBufferUsage.uniform,
          );
          clonedBuffers.add(clonedHandle);
          bindings.commandEncoderCopyBufferToBuffer(
            encoder,
            handle,
            0,
            clonedHandle,
            0,
            alignedSize,
          );
          entries.add(
            WgpuBindGroupEntryData(
              binding: i,
              buffer: clonedHandle,
              size: alignedSize,
            ),
          );
        } else {
          entries.add(
            WgpuBindGroupEntryData(
              binding: i,
              buffer: handle,
              size: alignedSize,
            ),
          );
        }
      }

      final uniformBindingIndex = buffers.length;
      final uniformCheckKey = '${pipelineKey}_$uniformBindingIndex';
      final hasBindingInCode = _pipelineHasUniform[uniformCheckKey] ??=
          shaderModule.code.contains('@binding($uniformBindingIndex)') ||
          shaderModule.bindings.any(
            (b) => b.isUniform && b.binding == uniformBindingIndex,
          );
      final hasUniformBinding =
          hasBindingInCode &&
          ((uniforms != null && uniforms.isNotEmpty) ||
              shaderModule.bindings.any(
                (b) => b.isUniform && b.binding == uniformBindingIndex,
              ));
      if (hasUniformBinding) {
        final uniformWords = uniforms ?? const <int>[];
        final uniformBytes = math.max(
          256,
          (uniformWords.length * 4 + 15) & ~15,
        );
        if (_uniformBuffer == ffi.nullptr ||
            _uniformBufferCapacity < uniformBytes) {
          if (_uniformBuffer != ffi.nullptr) {
            bindings.bufferDestroy(_uniformBuffer);
            bindings.bufferRelease(_uniformBuffer);
          }
          if (_uniformHostMemory != ffi.nullptr) {
            calloc.free(_uniformHostMemory);
          }
          _uniformBufferCapacity = math.max(uniformBytes, 256);
          _uniformBuffer = bindings.createBuffer(
            device,
            size: _uniformBufferCapacity,
            usage: WGPUBufferUsage.uniform | WGPUBufferUsage.copyDst,
            label: 'shared_uniform_buffer',
          );
          _uniformHostMemory = calloc<ffi.Uint32>(_uniformBufferCapacity ~/ 4);
        }
        final uniformDwords = uniformBytes ~/ 4;
        for (var wordIndex = 0; wordIndex < uniformDwords; wordIndex++) {
          _uniformHostMemory[wordIndex] = wordIndex < uniformWords.length
              ? uniformWords[wordIndex]
              : 0;
        }
        bindings.queueWriteBuffer(
          queue,
          _uniformBuffer,
          data: _uniformHostMemory.cast<ffi.Void>(),
          size: uniformBytes,
        );
        entries.add(
          WgpuBindGroupEntryData(
            binding: buffers.length,
            buffer: _uniformBuffer,
            size: uniformBytes,
          ),
        );
      }

      final bindGroup = bindings.createBindGroup(
        device,
        layout: bindGroupLayout,
        entries: entries,
      );

      final pass = bindings.commandEncoderBeginComputePass(encoder);
      bindings.computePassSetPipeline(pass, pipeline);
      bindings.computePassSetBindGroup(pass, 0, bindGroup);
      bindings.computePassDispatchWorkgroups(
        pass,
        workgroupsX,
        workgroupsY,
        workgroupsZ,
      );
      bindings.computePassEnd(pass);
      final commandBuffer = bindings.commandEncoderFinish(encoder);

      bindings.queueSubmit(queue, [commandBuffer]);
      bindings.devicePoll(device, wait: true);

      bindings.commandBufferRelease(commandBuffer);
      bindings.computePassEncoderRelease(pass);
      bindings.bindGroupRelease(bindGroup);
    } finally {
      bindings.commandEncoderRelease(encoder);
      for (final clonedHandle in clonedBuffers) {
        bindings.bufferDestroy(clonedHandle);
        bindings.bufferRelease(clonedHandle);
      }
    }
  }

  /// Disposes of all allocated GPU resources, cached pipelines, and device contexts.
  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;

    for (final bindGroupLayout in _bindGroupLayouts.values) {
      if (bindGroupLayout != ffi.nullptr) {
        bindings.bindGroupLayoutRelease(bindGroupLayout);
      }
    }
    _bindGroupLayouts.clear();
    _pipelineHasUniform.clear();

    for (final pipeline in _pipelines.values) {
      bindings.computePipelineRelease(pipeline);
    }
    _pipelines.clear();

    for (final module in _shaderModules.values) {
      bindings.shaderModuleRelease(module);
    }
    _shaderModules.clear();

    if (_uniformBuffer != ffi.nullptr) {
      bindings.bufferDestroy(_uniformBuffer);
      bindings.bufferRelease(_uniformBuffer);
      _uniformBuffer = ffi.nullptr;
      _uniformBufferCapacity = 0;
    }
    if (_uniformHostMemory != ffi.nullptr) {
      calloc.free(_uniformHostMemory);
      _uniformHostMemory = ffi.nullptr;
    }

    if (_stagingBuffer != ffi.nullptr) {
      bindings.bufferDestroy(_stagingBuffer);
      bindings.bufferRelease(_stagingBuffer);
      _stagingBuffer = ffi.nullptr;
      _stagingBufferCapacity = 0;
    }

    for (final handleAddress in _allocatedBufferSizes.keys.toList()) {
      final handle = ffi.Pointer<ffi.Void>.fromAddress(handleAddress);
      if (handle != ffi.nullptr) {
        bindings.bufferDestroy(handle);
        bindings.bufferRelease(handle);
      }
    }
    _allocatedBufferSizes.clear();

    if (queue != ffi.nullptr) {
      bindings.queueRelease(queue);
    }
    if (device != ffi.nullptr) {
      bindings.deviceDestroy(device);
      bindings.deviceRelease(device);
    }
    if (adapter != ffi.nullptr) {
      bindings.adapterRelease(adapter);
    }
    if (instance != ffi.nullptr) {
      bindings.instanceRelease(instance);
    }
  }
}

/// Synchronously creates the default [GpuBackend] for native platforms.
GpuBackend createDefaultGpuBackend() => WgpuNativeBackend.createSync();

/// Creates a [GpuDevice] backed by a native WebGPU (`libwgpu_native`) compute engine.
Future<GpuDevice> createWebGpuDevice({
  String name = 'Native WebGPU Device',
  bool enableMemoryPool = true,
}) async {
  final backend = await WgpuNativeBackend.create();
  return GpuDevice.create(
    name: name,
    type: GpuDeviceType.webgpu,
    backend: backend,
    enableMemoryPool: enableMemoryPool,
  );
}
