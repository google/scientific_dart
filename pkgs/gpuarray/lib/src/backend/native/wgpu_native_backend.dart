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

/// Record of a dispatched compute shader pass in mock or tracking mode.
final class MockDispatchRecord {
  final WgslShaderModule shaderModule;
  final List<GpuBuffer> buffers;
  final List<int>? uniforms;
  final int workgroupsX;
  final int workgroupsY;
  final int workgroupsZ;
  final DateTime timestamp;

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

/// Internal tracking representation for GPU and host memory allocations.
final class _WgpuBufferAllocation {
  final ffi.Pointer<ffi.Uint8> hostPointer;
  final ffi.Pointer<ffi.Void> gpuBuffer;
  final int sizeInBytes;
  final int usage;
  bool isGpuDirty;
  bool isHostDirty;

  _WgpuBufferAllocation({
    required this.hostPointer,
    required this.gpuBuffer,
    required this.sizeInBytes,
    required this.usage,
    this.isGpuDirty = false,
    this.isHostDirty = false,
  });
}

/// Native WebGPU hardware driver backend backed by `libwgpu_native` (Vulkan, Metal, DirectX 12).
final class WgpuNativeBackend extends GpuBackend {
  final WgpuNativeLib? lib;
  final ffi.Pointer<ffi.Void> instance;
  final ffi.Pointer<ffi.Void> adapter;
  final ffi.Pointer<ffi.Void> device;
  final ffi.Pointer<ffi.Void> queue;
  final bool isMock;

  final Map<int, _WgpuBufferAllocation> _allocations = {};
  final Map<String, ffi.Pointer<ffi.Void>> _shaderModules = {};
  final Map<String, ffi.Pointer<ffi.Void>> _pipelines = {};
  final List<MockDispatchRecord> _dispatches = [];

  bool _isDisposed = false;

  WgpuNativeBackend._({
    required this.lib,
    required this.instance,
    required this.adapter,
    required this.device,
    required this.queue,
    this.isMock = false,
  });

  /// Creates a mock/simulation backend for headless testing environments.
  factory WgpuNativeBackend({bool isMock = true, bool isSimulated = false}) =>
      WgpuNativeBackend.mock(isMock: isMock || isSimulated);

  /// Creates a mock/simulation backend for headless testing environments.
  factory WgpuNativeBackend.mock({bool isMock = true}) {
    return WgpuNativeBackend._(
      lib: null,
      instance: ffi.nullptr,
      adapter: ffi.nullptr,
      device: ffi.nullptr,
      queue: ffi.nullptr,
      isMock: isMock,
    );
  }

  /// Asynchronously creates and initializes a native WebGPU device backend.
  static Future<WgpuNativeBackend> create({
    String? libPath,
    WgpuNativeLib? lib,
    bool useMockIfUnavailable = true,
    bool fallbackToSimulation = true,
  }) async {
    final allowFallback = useMockIfUnavailable && fallbackToSimulation;
    var nativeLib = lib;
    nativeLib ??= WgpuNativeLib.tryLoad(customPath: libPath);

    if (nativeLib == null || !nativeLib.isAvailable) {
      if (allowFallback) {
        return WgpuNativeBackend.mock();
      }
      throw GpuDeviceException(
        'WebGPU native dynamic library (libwgpu_native) could not be loaded and fallback mode is disabled.',
      );
    }

    try {
      final instance = nativeLib.createInstance();
      final adapter = await nativeLib.requestAdapter(instance);
      final device = await nativeLib.requestDevice(
        instance,
        adapter,
        label: 'ScientificDart_WGPUDevice',
      );
      final queue = nativeLib.deviceGetQueue(device);

      return WgpuNativeBackend._(
        lib: nativeLib,
        instance: instance,
        adapter: adapter,
        device: device,
        queue: queue,
        isMock: false,
      );
    } catch (e) {
      if (allowFallback) {
        return WgpuNativeBackend.mock();
      }
      rethrow;
    }
  }

  @override
  GpuDeviceType get deviceType => GpuDeviceType.webgpu;

  /// Whether this driver backend has been disposed.
  bool get isDisposed => _isDisposed;

  /// Whether this backend is running in simulation/mock mode.
  @override
  bool get isSimulated => isMock;

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

  /// Total count of active low-level buffer allocations.
  int get activeAllocationCount => _allocations.length;

  @override
  ffi.Pointer<ffi.Uint8> allocateBuffer(int sizeInBytes) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot allocate buffer on disposed WgpuNativeBackend.',
      );
    }
    if (sizeInBytes <= 0) return ffi.nullptr;

    final hostPtr = calloc<ffi.Uint8>(sizeInBytes);
    ffi.Pointer<ffi.Void> gpuBuffer = ffi.nullptr;

    if (!isMock && device != ffi.nullptr && lib != null) {
      final alignedSize = math.max(16, (sizeInBytes + 3) & ~3);
      gpuBuffer = lib!.createBuffer(
        device,
        size: alignedSize,
        usage:
            WGPUBufferUsage.storage |
            WGPUBufferUsage.copySrc |
            WGPUBufferUsage.copyDst |
            WGPUBufferUsage.uniform,
        label: 'GpuBuffer_${hostPtr.address}',
      );
    }

    _allocations[hostPtr.address] = _WgpuBufferAllocation(
      hostPointer: hostPtr,
      gpuBuffer: gpuBuffer,
      sizeInBytes: sizeInBytes,
      usage:
          WGPUBufferUsage.storage |
          WGPUBufferUsage.copySrc |
          WGPUBufferUsage.copyDst |
          WGPUBufferUsage.uniform,
      isGpuDirty: false,
      isHostDirty: false,
    );

    return hostPtr;
  }

  @override
  void freeBuffer(ffi.Pointer<ffi.Uint8> pointer, int sizeInBytes) {
    if (pointer == ffi.nullptr) return;

    if (_allocations.remove(pointer.address) case final alloc?) {
      if (!isMock && alloc.gpuBuffer != ffi.nullptr && lib != null) {
        lib!.bufferDestroy(alloc.gpuBuffer);
        lib!.bufferRelease(alloc.gpuBuffer);
      }
      calloc.free(pointer);
    }
  }

  @override
  void ensureHostSynced(GpuBuffer buffer) {
    if (isMock ||
        device == ffi.nullptr ||
        queue == ffi.nullptr ||
        lib == null) {
      return;
    }
    if (_allocations[buffer.address.address] case final alloc?) {
      if (alloc.gpuBuffer != ffi.nullptr && alloc.isGpuDirty) {
        _syncGpuBufferToHost(alloc, offset: 0, bytes: alloc.sizeInBytes);
      }
    }
  }

  @override
  void markHostModified(GpuBuffer buffer) {
    if (_allocations[buffer.address.address] case final alloc?) {
      alloc.isHostDirty = true;
    }
  }

  @override
  void ensureGpuSynced(GpuBuffer buffer) {
    if (isMock || queue == ffi.nullptr || lib == null) return;
    if (_allocations[buffer.address.address] case final alloc?) {
      if (alloc.gpuBuffer != ffi.nullptr && alloc.isHostDirty) {
        final alignedBytes = math.max(16, (alloc.sizeInBytes + 3) & ~3);
        if (alignedBytes == alloc.sizeInBytes) {
          lib!.queueWriteBuffer(
            queue,
            alloc.gpuBuffer,
            bufferOffset: 0,
            data: alloc.hostPointer.cast<ffi.Void>(),
            size: alloc.sizeInBytes,
          );
        } else {
          using((arena) {
            final staging = arena<ffi.Uint8>(alignedBytes);
            staging
                .asTypedList(alloc.sizeInBytes)
                .setAll(0, alloc.hostPointer.asTypedList(alloc.sizeInBytes));
            lib!.queueWriteBuffer(
              queue,
              alloc.gpuBuffer,
              bufferOffset: 0,
              data: staging.cast<ffi.Void>(),
              size: alignedBytes,
            );
          });
        }
        alloc.isHostDirty = false;
      }
    }
  }

  @override
  void copyHostToBuffer(
    ffi.Pointer<ffi.Uint8> src,
    GpuBuffer dst,
    int bytes, {
    int offset = 0,
  }) {
    super.copyHostToBuffer(src, dst, bytes, offset: offset);

    if (!isMock && queue != ffi.nullptr && lib != null) {
      if (_allocations[dst.address.address] case final alloc?) {
        if (alloc.gpuBuffer != ffi.nullptr) {
          if (bytes % 4 == 0 && offset % 4 == 0) {
            lib!.queueWriteBuffer(
              queue,
              alloc.gpuBuffer,
              bufferOffset: offset,
              data: src.cast<ffi.Void>(),
              size: bytes,
            );
            alloc.isHostDirty = false;
          } else {
            alloc.isHostDirty = true;
            ensureGpuSynced(dst);
          }
          alloc.isGpuDirty = false;
        }
      }
    }
  }

  @override
  void copyBufferToHost(
    GpuBuffer src,
    ffi.Pointer<ffi.Uint8> dst,
    int bytes, {
    int offset = 0,
  }) {
    ensureHostSynced(src);
    super.copyBufferToHost(src, dst, bytes, offset: offset);
  }

  @override
  void copyBufferToBuffer(
    GpuBuffer src,
    GpuBuffer dst,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    ensureHostSynced(src);
    super.copyBufferToBuffer(
      src,
      dst,
      bytes,
      srcOffset: srcOffset,
      dstOffset: dstOffset,
    );

    if (bytes == 0) return;

    if (!isMock &&
        device != ffi.nullptr &&
        queue != ffi.nullptr &&
        lib != null) {
      ensureGpuSynced(src);
      final srcAlloc = _allocations[src.address.address];
      final dstAlloc = _allocations[dst.address.address];
      if (srcAlloc != null &&
          dstAlloc != null &&
          srcAlloc.gpuBuffer != ffi.nullptr &&
          dstAlloc.gpuBuffer != ffi.nullptr) {
        final alignedBytes = math.max(16, (bytes + 3) & ~3);
        if (srcOffset % 4 == 0 &&
            dstOffset % 4 == 0 &&
            srcOffset + alignedBytes <=
                math.max(16, (srcAlloc.sizeInBytes + 3) & ~3) &&
            dstOffset + alignedBytes <=
                math.max(16, (dstAlloc.sizeInBytes + 3) & ~3)) {
          final encoder = lib!.createCommandEncoder(
            device,
            label: 'copyBufferToBuffer_encoder',
          );
          lib!.commandEncoderCopyBufferToBuffer(
            encoder,
            srcAlloc.gpuBuffer,
            srcOffset,
            dstAlloc.gpuBuffer,
            dstOffset,
            alignedBytes,
          );
          final commandBuffer = lib!.commandEncoderFinish(encoder);
          lib!.queueSubmit(queue, [commandBuffer]);
          lib!.commandBufferRelease(commandBuffer);
          lib!.commandEncoderRelease(encoder);
          dstAlloc.isGpuDirty = true;
          dstAlloc.isHostDirty = false;
        } else {
          dstAlloc.isHostDirty = true;
          ensureGpuSynced(dst);
        }
      }
    }
  }

  void _syncGpuBufferToHost(
    _WgpuBufferAllocation alloc, {
    int offset = 0,
    int bytes = 0,
  }) {
    if (isMock ||
        lib == null ||
        device == ffi.nullptr ||
        queue == ffi.nullptr) {
      return;
    }
    final fullBytes = alloc.sizeInBytes;
    if (fullBytes <= 0) return;
    final syncBytes = math.max(16, (fullBytes + 3) & ~3);

    final stagingBuffer = lib!.createBuffer(
      device,
      size: syncBytes,
      usage: WGPUBufferUsage.mapRead | WGPUBufferUsage.copyDst,
      label: 'staging_readback',
    );

    try {
      final encoder = lib!.createCommandEncoder(
        device,
        label: 'staging_copy_encoder',
      );
      lib!.commandEncoderCopyBufferToBuffer(
        encoder,
        alloc.gpuBuffer,
        0,
        stagingBuffer,
        0,
        syncBytes,
      );
      final commandBuffer = lib!.commandEncoderFinish(encoder);
      lib!.queueSubmit(queue, [commandBuffer]);
      lib!.commandBufferRelease(commandBuffer);
      lib!.commandEncoderRelease(encoder);

      lib!.bufferMapSync(
        instance,
        stagingBuffer,
        device: device,
        mode: WGPUMapMode.read,
        offset: 0,
        size: syncBytes,
      );
      lib!.devicePoll(device, wait: true);

      final mappedPtr = lib!.bufferGetMappedRange(
        stagingBuffer,
        offset: 0,
        size: syncBytes,
      );
      if (mappedPtr != ffi.nullptr) {
        final srcBytes = mappedPtr.cast<ffi.Uint8>().asTypedList(fullBytes);
        final dstBytes = alloc.hostPointer.asTypedList(fullBytes);
        dstBytes.setAll(0, srcBytes);
        lib!.bufferUnmap(stagingBuffer);
      }
      alloc.isGpuDirty = false;
    } finally {
      lib!.bufferDestroy(stagingBuffer);
      lib!.bufferRelease(stagingBuffer);
    }
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
      throw GpuMemoryException(
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

    if (isMock ||
        lib == null ||
        device == ffi.nullptr ||
        queue == ffi.nullptr) {
      final cpuKernel = shaderModule.metadata['cpu_kernel'];
      if (cpuKernel is Function) {
        cpuKernel(buffers, uniforms, workgroupsX, workgroupsY, workgroupsZ);
      }
      return;
    }

    for (final buffer in buffers) {
      ensureGpuSynced(buffer);
    }

    // 1. Retrieve or compile WGPUShaderModule
    final module = _shaderModules[shaderModule.code] ??= lib!
        .createShaderModule(
          device,
          shaderModule.code,
          label: shaderModule.name,
        );

    // 2. Retrieve or create WGPUComputePipeline
    final pipelineKey =
        '${shaderModule.name}_${shaderModule.entryPoint}_${shaderModule.code.hashCode}';
    final pipeline = _pipelines[pipelineKey] ??= lib!.createComputePipeline(
      device,
      shaderModule: module,
      entryPoint: shaderModule.entryPoint,
      label: '${shaderModule.name}_pipeline',
    );

    // 3. Get bind group layout
    final bgLayout = lib!.pipelineGetBindGroupLayout(pipeline, 0);

    // 4. Create bind group entries
    final entries = <WgpuBindGroupEntryData>[];
    for (var i = 0; i < buffers.length; i++) {
      final buffer = buffers[i];
      if (_allocations[buffer.address.address] case final alloc?) {
        if (alloc.gpuBuffer != ffi.nullptr) {
          final alignedSize = math.max(16, (buffer.sizeInBytes + 3) & ~3);
          entries.add(
            WgpuBindGroupEntryData(
              binding: i,
              buffer: alloc.gpuBuffer,
              size: alignedSize,
            ),
          );
        }
      }
    }

    ffi.Pointer<ffi.Void> uniformGpuBuffer = ffi.nullptr;
    if (uniforms != null && uniforms.isNotEmpty) {
      final uniformBytes = math.max(16, ((uniforms.length * 4 + 15) & ~15));
      uniformGpuBuffer = lib!.createBuffer(
        device,
        size: uniformBytes,
        usage: WGPUBufferUsage.uniform | WGPUBufferUsage.copyDst,
        label: '${shaderModule.name}_uniforms',
      );
      using((arena) {
        final uniformDwords = uniformBytes ~/ 4;
        final uniformMem = arena<ffi.Uint32>(uniformDwords);
        for (var u = 0; u < uniformDwords; u++) {
          uniformMem[u] = u < uniforms.length ? uniforms[u] : 0;
        }
        lib!.queueWriteBuffer(
          queue,
          uniformGpuBuffer,
          data: uniformMem.cast<ffi.Void>(),
          size: uniformBytes,
        );
      });
      entries.add(
        WgpuBindGroupEntryData(
          binding: buffers.length,
          buffer: uniformGpuBuffer,
          size: uniformBytes,
        ),
      );
    }

    // 5. Create bind group
    final bindGroup = lib!.createBindGroup(
      device,
      layout: bgLayout,
      entries: entries,
      label: '${shaderModule.name}_bindGroup',
    );

    // 6. Encode and submit compute pass
    final encoder = lib!.createCommandEncoder(
      device,
      label: '${shaderModule.name}_encoder',
    );
    final pass = lib!.commandEncoderBeginComputePass(
      encoder,
      label: '${shaderModule.name}_pass',
    );
    lib!.computePassSetPipeline(pass, pipeline);
    lib!.computePassSetBindGroup(pass, 0, bindGroup);
    lib!.computePassDispatchWorkgroups(
      pass,
      workgroupsX,
      workgroupsY,
      workgroupsZ,
    );
    lib!.computePassEnd(pass);
    final commandBuffer = lib!.commandEncoderFinish(
      encoder,
      label: '${shaderModule.name}_cmdbuf',
    );

    lib!.queueSubmit(queue, [commandBuffer]);

    // Mark buffers as dirty on GPU
    for (final buffer in buffers) {
      final alloc = _allocations[buffer.address.address];
      if (alloc != null) {
        alloc.isGpuDirty = true;
      }
    }

    // 7. Clean up temporary objects
    lib!.commandBufferRelease(commandBuffer);
    lib!.commandEncoderRelease(encoder);
    lib!.computePassEncoderRelease(pass);
    lib!.bindGroupRelease(bindGroup);
    if (bgLayout != ffi.nullptr) {
      lib!.bindGroupLayoutRelease(bgLayout);
    }
    if (uniformGpuBuffer != ffi.nullptr) {
      lib!.devicePoll(device, wait: true);
      lib!.bufferDestroy(uniformGpuBuffer);
      lib!.bufferRelease(uniformGpuBuffer);
    }
  }

  /// Disposes of all allocated GPU resources, cached pipelines, and device contexts.
  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;

    if (lib != null) {
      for (final p in _pipelines.values) {
        lib!.computePipelineRelease(p);
      }
      _pipelines.clear();

      for (final s in _shaderModules.values) {
        lib!.shaderModuleRelease(s);
      }
      _shaderModules.clear();

      for (final alloc in _allocations.values) {
        if (alloc.gpuBuffer != ffi.nullptr) {
          lib!.bufferDestroy(alloc.gpuBuffer);
          lib!.bufferRelease(alloc.gpuBuffer);
        }
      }
      _allocations.clear();

      if (queue != ffi.nullptr) {
        lib!.queueRelease(queue);
      }
      if (device != ffi.nullptr) {
        lib!.deviceDestroy(device);
        lib!.deviceRelease(device);
      }
      if (adapter != ffi.nullptr) {
        lib!.adapterRelease(adapter);
      }
      if (instance != ffi.nullptr) {
        lib!.instanceRelease(instance);
      }
    } else {
      _allocations.clear();
    }
  }
}

/// Creates a [GpuDevice] backed by a native WebGPU (`libwgpu_native`) compute engine.
Future<GpuDevice> createWebGpuDevice({
  String name = 'Native WebGPU Device',
  String? libPath,
  WgpuNativeLib? lib,
  bool enableMemoryPool = true,
  bool fallbackToSimulation = true,
}) async {
  final backend = await WgpuNativeBackend.create(
    libPath: libPath,
    lib: lib,
    useMockIfUnavailable: fallbackToSimulation,
  );
  return GpuDevice.create(
    name: name,
    type: GpuDeviceType.webgpu,
    backend: backend,
    enableMemoryPool: enableMemoryPool,
  );
}
