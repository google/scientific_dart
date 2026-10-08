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
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:web/web.dart' as web;

import '../../buffer.dart';
import '../../device.dart';
import '../../exceptions.dart';
import '../backend.dart';
import '../wgsl/wgsl_types.dart';

/// Standard WebGPU buffer usage flag constants.
extension type const GPUBufferUsageConstants._(int value) implements int {
  static const int mapRead = 0x0001;
  static const int mapWrite = 0x0002;
  static const int copySrc = 0x0004;
  static const int copyDst = 0x0008;
  static const int index = 0x0010;
  static const int vertex = 0x0020;
  static const int uniform = 0x0040;
  static const int storage = 0x0080;
  static const int indirect = 0x0100;
  static const int queryResolve = 0x0200;
}

/// Standard WebGPU buffer mapping mode constants.
extension type const GPUMapModeConstants._(int value) implements int {
  static const int read = 0x0001;
  static const int write = 0x0002;
}

/// Standard WebGPU shader stage visibility constants.
extension type const GPUShaderStageConstants._(int value) implements int {
  static const int vertex = 0x0001;
  static const int fragment = 0x0002;
  static const int compute = 0x0004;
}

// ---------------------------------------------------------------------------
// WebGPU JS Interop Extension Types (W3C WebGPU Specification)
// ---------------------------------------------------------------------------

@JS()
extension type GPU(JSObject _) implements JSObject {
  external JSPromise<GPUAdapter?> requestAdapter([
    GPURequestAdapterOptions? options,
  ]);
  external String getPreferredCanvasFormat();
}

@JS()
extension type GPUAdapter(JSObject _) implements JSObject {
  external JSPromise<GPUDevice> requestDevice([
    GPUDeviceDescriptor? descriptor,
  ]);
  external GPUAdapterInfo get info;
  external GPUSupportedLimits get limits;
  external GPUSupportedFeatures get features;
}

@JS()
extension type GPUAdapterInfo(JSObject _) implements JSObject {
  external String get vendor;
  external String get architecture;
  external String get device;
  external String get description;
}

@JS()
extension type GPUSupportedLimits(JSObject _) implements JSObject {
  external int get maxComputeWorkgroupSizeX;
  external int get maxComputeWorkgroupSizeY;
  external int get maxComputeWorkgroupSizeZ;
  external int get maxComputeInvocationsPerWorkgroup;
  external int get maxComputeWorkgroupsPerDimension;
  external int get maxStorageBufferBindingSize;
  external int get maxBufferSize;
  external int get maxUniformBufferBindingSize;
}

@JS()
extension type GPUSupportedFeatures(JSObject _) implements JSObject {
  external bool has(String name);
}

@JS()
extension type GPUDevice(JSObject _) implements JSObject {
  external GPUQueue get queue;
  external GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
  external GPUShaderModule createShaderModule(
    GPUShaderModuleDescriptor descriptor,
  );
  external GPUComputePipeline createComputePipeline(
    GPUComputePipelineDescriptor descriptor,
  );
  external GPUBindGroupLayout createBindGroupLayout(
    GPUBindGroupLayoutDescriptor descriptor,
  );
  external GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor);
  external GPUCommandEncoder createCommandEncoder([
    GPUCommandEncoderDescriptor? descriptor,
  ]);
  external void destroy();
}

@JS()
extension type GPUQueue(JSObject _) implements JSObject {
  external void writeBuffer(
    GPUBuffer buffer,
    int bufferOffset,
    JSAny data, [
    int dataOffset,
    int size,
  ]);
  external void submit(JSArray<GPUCommandBuffer> commandBuffers);
  external JSPromise<JSAny?> onSubmittedWorkDone();
}

@JS()
extension type GPUBuffer(JSObject _) implements JSObject {
  external int get size;
  external int get usage;
  external int get mapState;
  external JSPromise<JSAny?> mapAsync(int mode, [int offset, int size]);
  external JSArrayBuffer getMappedRange([int offset, int size]);
  external void unmap();
  external void destroy();
}

@JS()
extension type GPUShaderModule(JSObject _) implements JSObject {
  external JSPromise<GPUCompilationInfo> getCompilationInfo();
}

@JS()
extension type GPUCompilationInfo(JSObject _) implements JSObject {
  external JSArray<GPUCompilationMessage> get messages;
}

@JS()
extension type GPUCompilationMessage(JSObject _) implements JSObject {
  external String get message;
  external String get type;
  @JS('lineNum')
  external int get lineNumber;
  @JS('linePos')
  external int get linePosition;
}

@JS()
extension type GPUComputePipeline(JSObject _) implements JSObject {
  external GPUBindGroupLayout getBindGroupLayout(int index);
}

@JS()
extension type GPUBindGroupLayout(JSObject _) implements JSObject {}

@JS()
extension type GPUBindGroup(JSObject _) implements JSObject {}

@JS()
extension type GPUCommandEncoder(JSObject _) implements JSObject {
  external GPUComputePassEncoder beginComputePass([
    GPUComputePassDescriptor? descriptor,
  ]);
  external void copyBufferToBuffer(
    GPUBuffer source,
    int sourceOffset,
    GPUBuffer destination,
    int destinationOffset,
    int size,
  );
  external GPUCommandBuffer finish([GPUCommandBufferDescriptor? descriptor]);
}

@JS()
extension type GPUComputePassEncoder(JSObject _) implements JSObject {
  external void setPipeline(GPUComputePipeline pipeline);
  external void setBindGroup(
    int index,
    GPUBindGroup? bindGroup, [
    JSArray<JSNumber>? dynamicOffsets,
  ]);
  external void dispatchWorkgroups(
    int workgroupCountX, [
    int workgroupCountY,
    int workgroupCountZ,
  ]);
  external void end();
}

@JS()
extension type GPUCommandBuffer(JSObject _) implements JSObject {}

// ---------------------------------------------------------------------------
// WebGPU JS Interop Descriptors
// ---------------------------------------------------------------------------

extension type GPURequestAdapterOptions._(JSObject _) implements JSObject {
  external factory GPURequestAdapterOptions({
    String? powerPreference,
    bool? forceFallbackAdapter,
  });
}

extension type GPUDeviceDescriptor._(JSObject _) implements JSObject {
  external factory GPUDeviceDescriptor({String? label});
}

extension type GPUBufferDescriptor._(JSObject _) implements JSObject {
  external factory GPUBufferDescriptor({
    String? label,
    required int size,
    required int usage,
    bool? mappedAtCreation,
  });
}

extension type GPUShaderModuleDescriptor._(JSObject _) implements JSObject {
  external factory GPUShaderModuleDescriptor({
    String? label,
    required String code,
  });
}

extension type GPUComputePipelineDescriptor._(JSObject _) implements JSObject {
  external factory GPUComputePipelineDescriptor({
    String? label,
    required JSAny layout,
    required GPUProgrammableStage compute,
  });
}

extension type GPUProgrammableStage._(JSObject _) implements JSObject {
  external factory GPUProgrammableStage({
    required GPUShaderModule module,
    String? entryPoint,
  });
}

extension type GPUBindGroupLayoutDescriptor._(JSObject _) implements JSObject {
  external factory GPUBindGroupLayoutDescriptor({
    String? label,
    required JSArray<GPUBindGroupLayoutEntry> entries,
  });
}

extension type GPUBindGroupLayoutEntry._(JSObject _) implements JSObject {
  external factory GPUBindGroupLayoutEntry({
    required int binding,
    required int visibility,
    GPUBufferBindingLayout? buffer,
  });
}

extension type GPUBufferBindingLayout._(JSObject _) implements JSObject {
  external factory GPUBufferBindingLayout({
    String? type,
    bool? hasDynamicOffset,
    int? minBindingSize,
  });
}

extension type GPUBindGroupDescriptor._(JSObject _) implements JSObject {
  external factory GPUBindGroupDescriptor({
    String? label,
    required GPUBindGroupLayout layout,
    required JSArray<GPUBindGroupEntry> entries,
  });
}

extension type GPUBindGroupEntry._(JSObject _) implements JSObject {
  external factory GPUBindGroupEntry({
    required int binding,
    required JSAny resource,
  });
}

extension type GPUBufferBinding._(JSObject _) implements JSObject {
  external factory GPUBufferBinding({
    required GPUBuffer buffer,
    int? offset,
    int? size,
  });
}

extension type GPUCommandEncoderDescriptor._(JSObject _) implements JSObject {
  external factory GPUCommandEncoderDescriptor({String? label});
}

extension type GPUComputePassDescriptor._(JSObject _) implements JSObject {
  external factory GPUComputePassDescriptor({String? label});
}

extension type GPUCommandBufferDescriptor._(JSObject _) implements JSObject {
  external factory GPUCommandBufferDescriptor({String? label});
}

/// Navigator helper extension to access the browser WebGPU instance.
extension NavigatorWebGpu on web.Navigator {
  /// The browser's [GPU] entry point, or `null` if WebGPU is unavailable.
  GPU? get gpu {
    try {
      final jsObj = this as JSObject;
      if (jsObj.hasProperty("gpu".toJS).toDart) {
        final prop = jsObj.getProperty("gpu".toJS);
        if (prop.isDefinedAndNotNull) {
          return prop as GPU;
        }
      }
    } catch (_) {}
    return null;
  }
}

// ---------------------------------------------------------------------------
// Browser WebGPU Hardware Compute Backend Driver
// ---------------------------------------------------------------------------

/// Hardware compute driver executing WGSL compute shaders directly on the browser WebGPU engine.
final class BrowserWebGpuBackend extends GpuBackend {
  /// The acquired WebGPU physical GPU adapter, if available.
  final GPUAdapter? adapter;

  /// The active WebGPU logical compute device, if available.
  final GPUDevice? device;

  int _nextBufferHandleId = 1;
  final Map<int, GPUBuffer> _deviceBuffers = {};
  final Map<int, int> _bufferSizes = {};
  final Map<String, GPUComputePipeline> _pipelineCache = {};
  final List<String> _dispatchLog = [];
  bool _isDisposed = false;

  /// Creates a [BrowserWebGpuBackend] wrapping [adapter] and [device].
  BrowserWebGpuBackend({this.adapter, this.device});

  @override
  GpuDeviceType get deviceType => GpuDeviceType.webgpu;

  /// Whether this backend driver has been disposed.
  bool get isDisposed => _isDisposed;

  /// Chronological log of all compute shader dispatches recorded by this backend.
  List<String> get dispatchLog => List.unmodifiable(_dispatchLog);

  /// Number of compiled pipelines currently cached in memory.
  int get pipelineCacheSize => _pipelineCache.length;

  /// Clears cached compute pipelines.
  void clearPipelineCache() => _pipelineCache.clear();

  /// Creates and initializes a [BrowserWebGpuBackend] using `window.navigator.gpu`.
  ///
  /// Throws a [GpuDeviceException] if WebGPU is not supported or device creation fails.
  static Future<BrowserWebGpuBackend> create({
    String? label,
    bool highPerformance = true,
  }) async {
    try {
      final navigator = web.window.navigator;
      final gpu = navigator.gpu;
      if (gpu == null) {
        throw const GpuDeviceException(
          'WebGPU is not supported in this browser environment (navigator.gpu is null).',
        );
      }

      final options = GPURequestAdapterOptions(
        powerPreference: highPerformance ? 'high-performance' : 'low-power',
      );
      final adapter = await gpu.requestAdapter(options).toDart;
      if (adapter == null) {
        throw const GpuDeviceException(
          'Failed to acquire a WebGPU hardware adapter.',
        );
      }

      final descriptor = GPUDeviceDescriptor(label: label);
      final device = await adapter.requestDevice(descriptor).toDart;
      return BrowserWebGpuBackend(adapter: adapter, device: device);
    } catch (error) {
      if (error is GpuException) rethrow;
      throw GpuDeviceException('WebGPU initialization failed: $error');
    }
  }

  @override
  ffi.Pointer<ffi.Void> allocateBuffer(
    int sizeInBytes, {
    GpuBufferUsage usage = GpuBufferUsage.defaultCompute,
  }) {
    RangeError.checkNotNegative(sizeInBytes, 'sizeInBytes');
    if (sizeInBytes == 0) return ffi.nullptr;
    final activeDevice = device;
    if (activeDevice == null) {
      throw const GpuDeviceException(
        'Cannot allocate buffer on uninitialized BrowserWebGpuBackend.',
      );
    }

    final alignedSize = math.max(16, (sizeInBytes + 3) & ~3);
    final gpuBuffer = activeDevice.createBuffer(
      GPUBufferDescriptor(
        size: alignedSize,
        usage:
            GPUBufferUsageConstants.storage |
            GPUBufferUsageConstants.copySrc |
            GPUBufferUsageConstants.copyDst |
            GPUBufferUsageConstants.uniform |
            usage.value,
      ),
    );
    final handleId = _nextBufferHandleId++;
    _deviceBuffers[handleId] = gpuBuffer;
    _bufferSizes[handleId] = alignedSize;
    return ffi.Pointer<ffi.Void>.fromAddress(handleId);
  }

  @override
  void freeBuffer(ffi.Pointer<ffi.Void> handle, int sizeInBytes) {
    if (handle == ffi.nullptr) return;
    final gpuBuffer = _deviceBuffers.remove(handle.address);
    if (gpuBuffer != null) {
      try {
        gpuBuffer.destroy();
      } catch (_) {}
    }
    _bufferSizes.remove(handle.address);
  }

  @override
  void copyHostToBuffer(
    ffi.Pointer<ffi.Uint8> src,
    GpuBuffer dst,
    int bytes, {
    int offset = 0,
  }) {
    super.copyHostToBuffer(src, dst, bytes, offset: offset);
    if (device != null && bytes > 0) {
      final gpuBuffer = _deviceBuffers[dst.nativeHandle.address];
      if (gpuBuffer != null) {
        final sourceBytes = src.asTypedList(bytes);
        final jsArray = sourceBytes.toJS;
        device!.queue.writeBuffer(gpuBuffer, offset, jsArray, 0, bytes);
      }
    }
  }

  /// Copies memory from GPU device buffer [src] into host pointer [dst]
  /// via a staging buffer and WebGPU `mapAsync(GPUMapMode.READ)`.
  Future<void> copyBufferToHostAsync(
    GpuBuffer src,
    ffi.Pointer<ffi.Uint8> dst,
    int bytes, {
    int offset = 0,
  }) async {
    if (device == null || bytes <= 0) {
      copyBufferToHost(src, dst, bytes, offset: offset);
      return;
    }
    final sourceGpu = _deviceBuffers[src.nativeHandle.address];
    if (sourceGpu == null) {
      copyBufferToHost(src, dst, bytes, offset: offset);
      return;
    }

    final alignedBytes = math.max(16, (bytes + 3) & ~3);
    final stagingBuffer = device!.createBuffer(
      GPUBufferDescriptor(
        size: alignedBytes,
        usage:
            GPUBufferUsageConstants.mapRead | GPUBufferUsageConstants.copyDst,
      ),
    );

    final encoder = device!.createCommandEncoder();
    encoder.copyBufferToBuffer(
      sourceGpu,
      offset,
      stagingBuffer,
      0,
      alignedBytes,
    );
    final commandBuffer = encoder.finish();
    device!.queue.submit([commandBuffer].toJS);

    await stagingBuffer
        .mapAsync(GPUMapModeConstants.read, 0, alignedBytes)
        .toDart;
    final arrayBuffer = stagingBuffer.getMappedRange(0, alignedBytes);
    final dartBytes = arrayBuffer.toDart.asUint8List();
    final destinationBytes = dst.asTypedList(bytes);
    destinationBytes.setRange(0, bytes, dartBytes);

    stagingBuffer.unmap();
    stagingBuffer.destroy();
  }

  @override
  void copyBufferToBuffer(
    GpuBuffer src,
    GpuBuffer dst,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    super.copyBufferToBuffer(
      src,
      dst,
      bytes,
      srcOffset: srcOffset,
      dstOffset: dstOffset,
    );
    if (device != null && bytes > 0) {
      final sourceGpu = _deviceBuffers[src.nativeHandle.address];
      final destinationGpu = _deviceBuffers[dst.nativeHandle.address];
      if (sourceGpu != null && destinationGpu != null) {
        final alignedBytes = math.max(16, (bytes + 3) & ~3);
        final encoder = device!.createCommandEncoder();
        encoder.copyBufferToBuffer(
          sourceGpu,
          srcOffset,
          destinationGpu,
          dstOffset,
          alignedBytes,
        );
        final commandBuffer = encoder.finish();
        device!.queue.submit([commandBuffer].toJS);
      }
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
        'Cannot dispatch compute pipeline on disposed BrowserWebGpuBackend.',
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

    _dispatchLog.add(
      '${shaderModule.name}($workgroupsX, $workgroupsY, $workgroupsZ)',
    );

    if (device == null) {
      throw const GpuDeviceException(
        'Cannot dispatch compute pipeline on uninitialized BrowserWebGpuBackend.',
      );
    }

    final pipeline = _getOrCreatePipeline(shaderModule);

    GPUBuffer? uniformBuffer;
    final entries = <GPUBindGroupEntry>[];

    if (uniforms != null && uniforms.isNotEmpty) {
      final u32List = Uint32List.fromList(uniforms);
      final uniformSize = math.max(16, (u32List.lengthInBytes + 15) & ~15);
      uniformBuffer = device!.createBuffer(
        GPUBufferDescriptor(
          size: uniformSize,
          usage:
              GPUBufferUsageConstants.uniform | GPUBufferUsageConstants.copyDst,
        ),
      );
      device!.queue.writeBuffer(uniformBuffer, 0, u32List.buffer.toJS);
    }

    var bufferIndex = 0;
    for (final binding in shaderModule.bindings) {
      if (binding.isUniform) {
        if (uniformBuffer != null) {
          entries.add(
            GPUBindGroupEntry(
              binding: binding.binding,
              resource: GPUBufferBinding(buffer: uniformBuffer),
            ),
          );
        }
      } else {
        if (bufferIndex < buffers.length) {
          final gpuBuffer =
              _deviceBuffers[buffers[bufferIndex].nativeHandle.address];
          if (gpuBuffer != null) {
            entries.add(
              GPUBindGroupEntry(
                binding: binding.binding,
                resource: GPUBufferBinding(buffer: gpuBuffer),
              ),
            );
          }
          bufferIndex++;
        }
      }
    }

    final bindGroup = device!.createBindGroup(
      GPUBindGroupDescriptor(
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.toJS,
      ),
    );

    final encoder = device!.createCommandEncoder();
    final pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();

    final commandBuffer = encoder.finish();
    device!.queue.submit([commandBuffer].toJS);
  }

  /// Compiles or retrieves a cached [GPUComputePipeline] for [shaderModule].
  GPUComputePipeline _getOrCreatePipeline(WgslShaderModule shaderModule) {
    final key =
        '${shaderModule.name}_${shaderModule.entryPoint}_${shaderModule.code}';
    if (_pipelineCache[key] case final cached?) return cached;

    final module = device!.createShaderModule(
      GPUShaderModuleDescriptor(
        label: shaderModule.name,
        code: shaderModule.code,
      ),
    );
    final pipeline = device!.createComputePipeline(
      GPUComputePipelineDescriptor(
        label: '${shaderModule.name}_pipeline',
        layout: 'auto'.toJS,
        compute: GPUProgrammableStage(
          module: module,
          entryPoint: shaderModule.entryPoint,
        ),
      ),
    );
    _pipelineCache[key] = pipeline;
    return pipeline;
  }

  /// Releases all allocated WebGPU device buffers, cached pipelines, and destroys the device context.
  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;

    for (final gpuBuffer in _deviceBuffers.values) {
      try {
        gpuBuffer.destroy();
      } catch (_) {}
    }
    _deviceBuffers.clear();
    _bufferSizes.clear();
    _pipelineCache.clear();

    try {
      device?.destroy();
    } catch (_) {}
  }
}

/// Synchronously creates a default [GpuBackend] placeholder on web platforms.
GpuBackend createDefaultGpuBackend() => BrowserWebGpuBackend();

/// Creates a [GpuDevice] backed by a browser WebGPU compute engine.
Future<GpuDevice> createWebGpuDevice({
  String name = 'Browser WebGPU Device',
  bool enableMemoryPool = true,
}) async {
  final backend = await BrowserWebGpuBackend.create();
  return GpuDevice.create(
    name: name,
    type: GpuDeviceType.webgpu,
    backend: backend,
    enableMemoryPool: enableMemoryPool,
  );
}
