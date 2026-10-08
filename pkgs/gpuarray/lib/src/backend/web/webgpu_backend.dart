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

GPU? _browserWebGpu() {
  try {
    final navigator = globalContext['navigator'];
    if (navigator != null && navigator.isA<JSObject>()) {
      final navigatorObject = navigator as JSObject;
      if (navigatorObject.hasProperty('gpu'.toJS).toDart) {
        final prop = navigatorObject.getProperty('gpu'.toJS);
        if (prop.isDefinedAndNotNull) {
          return prop as GPU;
        }
      }
    }
  } catch (_) {}
  return null;
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
  final Map<int, Uint8List> _hostBuffers = {};
  final Map<int, int> _bufferSizes = {};
  final Set<int> _dirtyGpuBuffers = <int>{};
  final List<GPUBuffer> _pendingTemporaryBuffers = [];
  final Map<String, GPUComputePipeline> _pipelineCache = {};
  final Map<String, bool> _pipelineHasUniform = {};
  final List<String> _dispatchLog = [];
  bool _isDisposed = false;

  /// Creates a [BrowserWebGpuBackend] wrapping [adapter] and [device].
  BrowserWebGpuBackend({this.adapter, this.device});

  @override
  GpuDeviceType get deviceType => GpuDeviceType.webgpu;

  @override
  bool get isInitialized => device != null;

  /// Whether this backend driver has been disposed.
  bool get isDisposed => _isDisposed;

  /// Chronological log of all compute shader dispatches recorded by this backend.
  List<String> get dispatchLog => List.unmodifiable(_dispatchLog);

  /// Number of compiled pipelines currently cached in memory.
  int get pipelineCacheSize => _pipelineCache.length;

  /// Clears cached compute pipelines.
  void clearPipelineCache() {
    _pipelineCache.clear();
    _pipelineHasUniform.clear();
  }

  /// Creates and initializes a [BrowserWebGpuBackend] using `navigator.gpu`.
  ///
  /// Throws a [GpuDeviceException] if WebGPU is not supported or device creation fails.
  static Future<BrowserWebGpuBackend> create({
    String? label,
    bool highPerformance = true,
  }) async {
    try {
      final gpu = _browserWebGpu();
      if (gpu == null) {
        throw const GpuDeviceException(
          'WebGPU is not supported in this browser environment (navigator.gpu is null).',
        );
      }

      final options = GPURequestAdapterOptions(
        powerPreference: highPerformance ? 'high-performance' : 'low-power',
      );
      var adapter = await gpu.requestAdapter(options).toDart;
      adapter ??= await gpu
          .requestAdapter(GPURequestAdapterOptions(forceFallbackAdapter: true))
          .toDart;
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
    _hostBuffers[handleId] = Uint8List(alignedSize);
    _bufferSizes[handleId] = alignedSize;
    return ffi.Pointer<ffi.Void>.fromAddress(handleId);
  }

  @override
  void freeBuffer(ffi.Pointer<ffi.Void> handle, int sizeInBytes) {
    if (handle == ffi.nullptr) return;
    final handleId = handle.address;
    final gpuBuffer = _deviceBuffers.remove(handleId);
    if (gpuBuffer != null) {
      try {
        gpuBuffer.destroy();
      } catch (_) {}
    }
    _hostBuffers.remove(handleId);
    _bufferSizes.remove(handleId);
    _dirtyGpuBuffers.remove(handleId);
  }

  @override
  void clearBuffer(GpuBuffer buffer, {int offset = 0, int? bytes}) {
    super.clearBuffer(buffer, offset: offset, bytes: bytes);
    final resolvedBytes = bytes ?? (buffer.sizeInBytes - offset);
    if (resolvedBytes <= 0) return;

    final handleId = buffer.nativeHandle.address;
    final hostBuffer = _hostBuffers[handleId];
    if (hostBuffer != null) {
      hostBuffer.fillRange(offset, offset + resolvedBytes, 0);
    }
    if (device != null) {
      final gpuBuffer = _deviceBuffers[handleId];
      if (gpuBuffer != null) {
        final alignedBytes = (resolvedBytes + 3) & ~3;
        final zeroes = Uint8List(alignedBytes);
        device!.queue.writeBuffer(
          gpuBuffer,
          offset,
          zeroes.toJS,
          0,
          alignedBytes,
        );
        if (offset == 0 && resolvedBytes >= buffer.sizeInBytes) {
          _dirtyGpuBuffers.remove(handleId);
        }
      }
    }
  }

  static const bool _isWasmRuntime = bool.fromEnvironment(
    'dart.tool.dart2wasm',
  );

  static void _copyPointerToBytes(
    ffi.Pointer<ffi.Uint8> src,
    Uint8List dst,
    int dstOffset,
    int bytes,
  ) {
    if (!_isWasmRuntime) {
      dst.setRange(dstOffset, dstOffset + bytes, src.asTypedList(bytes));
      return;
    }
    for (var i = 0; i < bytes; i++) {
      dst[dstOffset + i] = src[i];
    }
  }

  static void _copyBytesToPointer(
    Uint8List src,
    int srcOffset,
    ffi.Pointer<ffi.Uint8> dst,
    int bytes,
  ) {
    if (!_isWasmRuntime) {
      dst.asTypedList(bytes).setRange(0, bytes, src, srcOffset);
      return;
    }
    for (var i = 0; i < bytes; i++) {
      dst[i] = src[srcOffset + i];
    }
  }

  static void _zeroPointer(ffi.Pointer<ffi.Uint8> dst, int bytes) {
    if (!_isWasmRuntime) {
      dst.asTypedList(bytes).fillRange(0, bytes, 0);
      return;
    }
    for (var i = 0; i < bytes; i++) {
      dst[i] = 0;
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
    if (bytes <= 0) return;

    final handleId = dst.nativeHandle.address;
    final hostBuffer = _hostBuffers[handleId];
    if (hostBuffer != null) {
      _copyPointerToBytes(src, hostBuffer, offset, bytes);
    }
    if (device != null) {
      final gpuBuffer = _deviceBuffers[handleId];
      if (gpuBuffer != null) {
        final alignedOffset = offset & ~3;
        final alignedEnd = ((offset + bytes) + 3) & ~3;
        final alignedLength = alignedEnd - alignedOffset;
        if (hostBuffer != null && alignedEnd <= hostBuffer.length) {
          final slice = Uint8List.sublistView(
            hostBuffer,
            alignedOffset,
            alignedEnd,
          );
          device!.queue.writeBuffer(
            gpuBuffer,
            alignedOffset,
            slice.toJS,
            0,
            alignedLength,
          );
        }
        if (offset == 0 && bytes >= dst.sizeInBytes) {
          _dirtyGpuBuffers.remove(handleId);
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
    super.copyBufferToHost(src, dst, bytes, offset: offset);
    if (bytes <= 0) return;

    final hostBuffer = _hostBuffers[src.nativeHandle.address];
    if (hostBuffer != null) {
      _copyBytesToPointer(hostBuffer, offset, dst, bytes);
    } else {
      _zeroPointer(dst, bytes);
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
    final handleId = src.nativeHandle.address;
    final sourceGpu = _deviceBuffers[handleId];
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
    _copyBytesToPointer(dartBytes, 0, dst, bytes);

    final hostBuffer = _hostBuffers[handleId];
    if (hostBuffer != null && offset + bytes <= hostBuffer.length) {
      hostBuffer.setRange(offset, offset + bytes, dartBytes);
      if (offset == 0 && bytes >= src.sizeInBytes) {
        _dirtyGpuBuffers.remove(handleId);
      }
    }

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
    if (bytes <= 0) return;

    final sourceId = src.nativeHandle.address;
    final destinationId = dst.nativeHandle.address;
    final sourceHost = _hostBuffers[sourceId];
    final destinationHost = _hostBuffers[destinationId];
    if (sourceHost != null && destinationHost != null) {
      destinationHost.setRange(
        dstOffset,
        dstOffset + bytes,
        sourceHost,
        srcOffset,
      );
    }

    if (device != null) {
      final sourceGpu = _deviceBuffers[sourceId];
      final destinationGpu = _deviceBuffers[destinationId];
      if (sourceGpu != null && destinationGpu != null) {
        final alignedBytes = (bytes + 3) & ~3;
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
        if (_dirtyGpuBuffers.contains(sourceId)) {
          _dirtyGpuBuffers.add(destinationId);
        }
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

    final activeDevice = device;
    if (activeDevice == null) {
      throw const GpuDeviceException(
        'Cannot dispatch compute pipeline on uninitialized BrowserWebGpuBackend.',
      );
    }

    final pipelineKey =
        '${shaderModule.name}_${shaderModule.entryPoint}_${shaderModule.code}';
    final pipeline = _getOrCreatePipeline(shaderModule, pipelineKey);
    final encoder = activeDevice.createCommandEncoder();
    final entries = <GPUBindGroupEntry>[];

    for (var i = 0; i < buffers.length; i++) {
      final buffer = buffers[i];
      final handle = buffer.nativeHandle;
      if (handle == ffi.nullptr) {
        final emptyBuffer = activeDevice.createBuffer(
          GPUBufferDescriptor(
            size: 16,
            usage:
                GPUBufferUsageConstants.storage |
                GPUBufferUsageConstants.copySrc |
                GPUBufferUsageConstants.copyDst |
                GPUBufferUsageConstants.uniform,
          ),
        );
        _pendingTemporaryBuffers.add(emptyBuffer);
        entries.add(
          GPUBindGroupEntry(
            binding: i,
            resource: GPUBufferBinding(buffer: emptyBuffer, size: 16),
          ),
        );
        continue;
      }

      final handleId = handle.address;
      final gpuBuffer = _deviceBuffers[handleId];
      if (gpuBuffer == null) continue;

      final alignedSize =
          _bufferSizes[handleId] ??
          math.max(16, (buffer.allocatedBytes + 3) & ~3);

      var isAliasedLater = false;
      for (var j = i + 1; j < buffers.length; j++) {
        if (buffers[j].nativeHandle == handle) {
          isAliasedLater = true;
          break;
        }
      }

      if (isAliasedLater) {
        final clonedBuffer = activeDevice.createBuffer(
          GPUBufferDescriptor(
            size: alignedSize,
            usage:
                GPUBufferUsageConstants.storage |
                GPUBufferUsageConstants.copySrc |
                GPUBufferUsageConstants.copyDst |
                GPUBufferUsageConstants.uniform,
          ),
        );
        _pendingTemporaryBuffers.add(clonedBuffer);
        encoder.copyBufferToBuffer(gpuBuffer, 0, clonedBuffer, 0, alignedSize);
        entries.add(
          GPUBindGroupEntry(
            binding: i,
            resource: GPUBufferBinding(buffer: clonedBuffer, size: alignedSize),
          ),
        );
      } else {
        entries.add(
          GPUBindGroupEntry(
            binding: i,
            resource: GPUBufferBinding(buffer: gpuBuffer, size: alignedSize),
          ),
        );
      }
      _dirtyGpuBuffers.add(handleId);
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
      final uniformBytes = math.max(256, (uniformWords.length * 4 + 15) & ~15);
      final uniformDwords = uniformBytes ~/ 4;
      final uniformData = Uint32List(uniformDwords);
      for (var wordIndex = 0; wordIndex < uniformWords.length; wordIndex++) {
        uniformData[wordIndex] = uniformWords[wordIndex];
      }
      final uniformBuffer = activeDevice.createBuffer(
        GPUBufferDescriptor(
          size: uniformBytes,
          usage:
              GPUBufferUsageConstants.uniform | GPUBufferUsageConstants.copyDst,
        ),
      );
      _pendingTemporaryBuffers.add(uniformBuffer);
      activeDevice.queue.writeBuffer(
        uniformBuffer,
        0,
        Uint8List.sublistView(uniformData).toJS,
        0,
        uniformBytes,
      );
      entries.add(
        GPUBindGroupEntry(
          binding: uniformBindingIndex,
          resource: GPUBufferBinding(buffer: uniformBuffer, size: uniformBytes),
        ),
      );
    }

    final bindGroup = activeDevice.createBindGroup(
      GPUBindGroupDescriptor(
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.toJS,
      ),
    );

    final pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();

    final commandBuffer = encoder.finish();
    activeDevice.queue.submit([commandBuffer].toJS);
  }

  /// Compiles or retrieves a cached [GPUComputePipeline] for [shaderModule].
  GPUComputePipeline _getOrCreatePipeline(
    WgslShaderModule shaderModule,
    String key,
  ) {
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

  @override
  Future<void> synchronize() async {
    if (_isDisposed || device == null) return;
    if (_dirtyGpuBuffers.isEmpty) {
      for (final temporaryBuffer in _pendingTemporaryBuffers) {
        try {
          temporaryBuffer.destroy();
        } catch (_) {}
      }
      _pendingTemporaryBuffers.clear();
      return;
    }

    final activeDevice = device!;
    final dirtyIds = <int>[];
    final stagingBuffers = <GPUBuffer>[];
    final stagingSizes = <int>[];

    final encoder = activeDevice.createCommandEncoder();
    for (final handleId in _dirtyGpuBuffers) {
      final sourceGpu = _deviceBuffers[handleId];
      final bufferSize = _bufferSizes[handleId];
      if (sourceGpu == null || bufferSize == null || bufferSize <= 0) {
        continue;
      }
      final stagingBuffer = activeDevice.createBuffer(
        GPUBufferDescriptor(
          size: bufferSize,
          usage:
              GPUBufferUsageConstants.mapRead | GPUBufferUsageConstants.copyDst,
        ),
      );
      encoder.copyBufferToBuffer(sourceGpu, 0, stagingBuffer, 0, bufferSize);
      dirtyIds.add(handleId);
      stagingBuffers.add(stagingBuffer);
      stagingSizes.add(bufferSize);
    }
    _dirtyGpuBuffers.clear();

    if (stagingBuffers.isNotEmpty) {
      final commandBuffer = encoder.finish();
      activeDevice.queue.submit([commandBuffer].toJS);

      for (var i = 0; i < stagingBuffers.length; i++) {
        final handleId = dirtyIds[i];
        final stagingBuffer = stagingBuffers[i];
        final bufferSize = stagingSizes[i];
        try {
          await stagingBuffer
              .mapAsync(GPUMapModeConstants.read, 0, bufferSize)
              .toDart;
          final arrayBuffer = stagingBuffer.getMappedRange(0, bufferSize);
          final mappedBytes = arrayBuffer.toDart.asUint8List();
          final hostBuffer = _hostBuffers[handleId];
          if (hostBuffer != null) {
            final copyBytes = math.min(hostBuffer.length, mappedBytes.length);
            hostBuffer.setRange(0, copyBytes, mappedBytes);
          }
          stagingBuffer.unmap();
        } finally {
          try {
            stagingBuffer.destroy();
          } catch (_) {}
        }
      }
    }

    for (final temporaryBuffer in _pendingTemporaryBuffers) {
      try {
        temporaryBuffer.destroy();
      } catch (_) {}
    }
    _pendingTemporaryBuffers.clear();
  }

  /// Releases all allocated WebGPU device buffers, cached pipelines, and destroys the device context.
  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;

    for (final temporaryBuffer in _pendingTemporaryBuffers) {
      try {
        temporaryBuffer.destroy();
      } catch (_) {}
    }
    _pendingTemporaryBuffers.clear();

    for (final gpuBuffer in _deviceBuffers.values) {
      try {
        gpuBuffer.destroy();
      } catch (_) {}
    }
    _deviceBuffers.clear();
    _hostBuffers.clear();
    _bufferSizes.clear();
    _dirtyGpuBuffers.clear();
    _pipelineCache.clear();
    _pipelineHasUniform.clear();

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
