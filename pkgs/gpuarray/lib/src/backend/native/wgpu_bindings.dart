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

// ignore_for_file: constant_identifier_names

/// Native C-FFI bindings for `libwgpu_native` resolved via Dart Native Assets.
@ffi.DefaultAsset('package:gpuarray/wgpu_native')
library;

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import '../../exceptions.dart';

// =============================================================================
// WebGPU Standard Usage and Enumeration Constants
// =============================================================================

/// Bitflag constants specifying allowed operations on a `WGPUBuffer`.
extension type const WGPUBufferUsage._(int value) implements int {
  /// No usage flags set.
  static const int none = 0x00000000;

  /// Allows mapping the buffer for CPU reading (`wgpuBufferMapAsync`).
  static const int mapRead = 0x00000001;

  /// Allows mapping the buffer for CPU writing (`wgpuBufferMapAsync`).
  static const int mapWrite = 0x00000002;

  /// Allows using the buffer as the source of a copy operation.
  static const int copySrc = 0x00000004;

  /// Allows using the buffer as the destination of a copy or write operation.
  static const int copyDst = 0x00000008;

  /// Allows binding the buffer as an index buffer.
  static const int index = 0x00000010;

  /// Allows binding the buffer as a vertex buffer.
  static const int vertex = 0x00000020;

  /// Allows binding the buffer as a uniform buffer (`var<uniform>`).
  static const int uniform = 0x00000040;

  /// Allows binding the buffer as a storage buffer (`var<storage>`).
  static const int storage = 0x00000080;

  /// Allows using the buffer for indirect dispatch parameters.
  static const int indirect = 0x00000100;

  /// Allows using the buffer as a query resolve destination.
  static const int queryResolve = 0x00000200;
}

/// C-style alias for [WGPUBufferUsage.none].
const int WGPUBufferUsage_None = WGPUBufferUsage.none;

/// C-style alias for [WGPUBufferUsage.mapRead].
const int WGPUBufferUsage_MapRead = WGPUBufferUsage.mapRead;

/// C-style alias for [WGPUBufferUsage.mapWrite].
const int WGPUBufferUsage_MapWrite = WGPUBufferUsage.mapWrite;

/// C-style alias for [WGPUBufferUsage.copySrc].
const int WGPUBufferUsage_CopySrc = WGPUBufferUsage.copySrc;

/// C-style alias for [WGPUBufferUsage.copyDst].
const int WGPUBufferUsage_CopyDst = WGPUBufferUsage.copyDst;

/// C-style alias for [WGPUBufferUsage.index].
const int WGPUBufferUsage_Index = WGPUBufferUsage.index;

/// C-style alias for [WGPUBufferUsage.vertex].
const int WGPUBufferUsage_Vertex = WGPUBufferUsage.vertex;

/// C-style alias for [WGPUBufferUsage.uniform].
const int WGPUBufferUsage_Uniform = WGPUBufferUsage.uniform;

/// C-style alias for [WGPUBufferUsage.storage].
const int WGPUBufferUsage_Storage = WGPUBufferUsage.storage;

/// C-style alias for [WGPUBufferUsage.indirect].
const int WGPUBufferUsage_Indirect = WGPUBufferUsage.indirect;

/// C-style alias for [WGPUBufferUsage.queryResolve].
const int WGPUBufferUsage_QueryResolve = WGPUBufferUsage.queryResolve;

/// Bitflag constants specifying CPU access mode when mapping a `WGPUBuffer`.
extension type const WGPUMapMode._(int value) implements int {
  /// No mapping mode specified.
  static const int none = 0x00000000;

  /// Maps the buffer range for host read access.
  static const int read = 0x00000001;

  /// Maps the buffer range for host write access.
  static const int write = 0x00000002;
}

/// C-style alias for [WGPUMapMode.none].
const int WGPUMapMode_None = WGPUMapMode.none;

/// C-style alias for [WGPUMapMode.read].
const int WGPUMapMode_Read = WGPUMapMode.read;

/// C-style alias for [WGPUMapMode.write].
const int WGPUMapMode_Write = WGPUMapMode.write;

/// Structure type discriminators for `WGPUChainedStruct` extension chains.
extension type const WGPUSType._(int value) implements int {
  /// Uninitialized or invalid chained structure type.
  static const int invalid = 0x00000000;

  /// Metal layer surface descriptor chain type.
  static const int surfaceDescriptorFromMetalLayer = 0x00000001;

  /// Windows HWND surface descriptor chain type.
  static const int surfaceDescriptorFromWindowsHWND = 0x00000002;

  /// X11 Xlib window surface descriptor chain type.
  static const int surfaceDescriptorFromXlibWindow = 0x00000003;

  /// HTML canvas selector surface descriptor chain type.
  static const int surfaceDescriptorFromCanvasHTMLSelector = 0x00000004;

  /// WGSL shader source descriptor chain type in `webgpu.h` v29+.
  static const int shaderSourceWGSL = 0x00000002;

  /// Legacy WGSL shader module descriptor chain type.
  static const int shaderModuleWGSLDescriptor = 0x00000006;
}

/// C-style alias for [WGPUSType.invalid].
const int WGPUSType_Invalid = WGPUSType.invalid;

/// C-style alias for [WGPUSType.shaderSourceWGSL].
const int WGPUSType_ShaderSourceWGSL = WGPUSType.shaderSourceWGSL;

/// C-style alias for [WGPUSType.shaderModuleWGSLDescriptor].
const int WGPUSType_ShaderModuleWGSLDescriptor =
    WGPUSType.shaderModuleWGSLDescriptor;

/// Execution modes for asynchronous WebGPU C callbacks.
extension type const WGPUCallbackMode._(int value) implements int {
  /// Fires callbacks only inside `wgpuInstanceWaitAny`.
  static const int waitAnyOnly = 0x00000001;

  /// Allows callbacks to fire during `wgpuInstanceProcessEvents`.
  static const int allowProcessEvents = 0x00000002;

  /// Allows callbacks to fire spontaneously on completion or during polling.
  static const int allowSpontaneous = 0x00000004;
}

/// Adapter power preference hints for `wgpuInstanceRequestAdapter`.
extension type const WGPUPowerPreference._(int value) implements int {
  /// No preference specified.
  static const int undefined = 0x00000000;

  /// Prefers integrated or energy-efficient adapters.
  static const int lowPower = 0x00000001;

  /// Prefers discrete high-throughput adapters.
  static const int highPerformance = 0x00000002;
}

/// C-style alias for [WGPUPowerPreference.undefined].
const int WGPUPowerPreference_Undefined = WGPUPowerPreference.undefined;

/// C-style alias for [WGPUPowerPreference.lowPower].
const int WGPUPowerPreference_LowPower = WGPUPowerPreference.lowPower;

/// C-style alias for [WGPUPowerPreference.highPerformance].
const int WGPUPowerPreference_HighPerformance =
    WGPUPowerPreference.highPerformance;

/// Graphics/compute driver API identifiers in `webgpu.h`.
extension type const WGPUBackendType._(int value) implements int {
  /// Unspecified backend type.
  static const int undefined = 0x00000000;

  /// Null no-op backend.
  static const int nullBackend = 0x00000001;

  /// Browser WebGPU backend.
  static const int webGpu = 0x00000002;

  /// Direct3D 11 backend.
  static const int d3d11 = 0x00000003;

  /// Direct3D 12 backend.
  static const int d3d12 = 0x00000004;

  /// Apple Metal backend.
  static const int metal = 0x00000005;

  /// Khronos Vulkan backend.
  static const int vulkan = 0x00000006;

  /// Desktop OpenGL backend.
  static const int openGl = 0x00000007;

  /// OpenGL ES backend.
  static const int openGlEs = 0x00000008;
}

// =============================================================================
// WebGPU C-FFI Native Struct Definitions
// =============================================================================

/// Sized UTF-8 string slice used by `webgpu.h` descriptors and callbacks.
final class WGPUStringView extends ffi.Struct {
  /// Pointer to the UTF-8 byte sequence.
  external ffi.Pointer<Utf8> data;

  /// Byte length of the UTF-8 string slice.
  @ffi.UintPtr()
  external int length;
}

/// Header node for linked extension descriptor chains in `webgpu.h`.
final class WGPUChainedStruct extends ffi.Struct {
  /// Pointer to the next chained extension struct, or `nullptr`.
  external ffi.Pointer<WGPUChainedStruct> next;

  /// Structure type discriminator from [WGPUSType].
  @ffi.Uint32()
  external int sType;
}

/// Descriptor passed to [wgpuCreateInstance].
final class WGPUInstanceDescriptor extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
}

/// Callback configuration passed to [wgpuAdapterRequestDevice].
final class WGPURequestDeviceCallbackInfo extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<ffi.Void> nextInChain;

  /// Callback invocation mode from [WGPUCallbackMode].
  @ffi.Uint32()
  external int mode;

  /// Native callback function invoked when device creation completes.
  external ffi.Pointer<
    ffi.NativeFunction<
      ffi.Void Function(
        ffi.Uint32,
        ffi.Pointer<ffi.Void>,
        WGPUStringView,
        ffi.Pointer<ffi.Void>,
        ffi.Pointer<ffi.Void>,
      )
    >
  >
  callback;

  /// First opaque user data pointer forwarded to [callback].
  external ffi.Pointer<ffi.Void> userdata1;

  /// Second opaque user data pointer forwarded to [callback].
  external ffi.Pointer<ffi.Void> userdata2;
}

/// Callback configuration passed to [wgpuBufferMapAsync].
final class WGPUBufferMapCallbackInfo extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<ffi.Void> nextInChain;

  /// Callback invocation mode from [WGPUCallbackMode].
  @ffi.Uint32()
  external int mode;

  /// Native callback function invoked when buffer mapping completes.
  external ffi.Pointer<
    ffi.NativeFunction<
      ffi.Void Function(
        ffi.Uint32,
        WGPUStringView,
        ffi.Pointer<ffi.Void>,
        ffi.Pointer<ffi.Void>,
      )
    >
  >
  callback;

  /// First opaque user data pointer forwarded to [callback].
  external ffi.Pointer<ffi.Void> userdata1;

  /// Second opaque user data pointer forwarded to [callback].
  external ffi.Pointer<ffi.Void> userdata2;
}

/// Descriptor specifying size and usage flags for [wgpuDeviceCreateBuffer].
final class WGPUBufferDescriptor extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  /// Debug label for the buffer.
  external WGPUStringView label;

  /// Bitwise combination of [WGPUBufferUsage] flags.
  @ffi.Uint64()
  external int usage;

  /// Buffer allocation size in bytes.
  @ffi.Uint64()
  external int size;

  /// Non-zero if the buffer should be mapped at creation time.
  @ffi.Uint32()
  external int mappedAtCreation;
}

/// Chained descriptor supplying WGSL source code to [wgpuDeviceCreateShaderModule].
final class WGPUShaderSourceWGSL extends ffi.Struct {
  /// Base chained struct with `sType` set to [WGPUSType.shaderSourceWGSL].
  external WGPUChainedStruct chain;

  /// UTF-8 WGSL source code view.
  external WGPUStringView code;
}

/// Descriptor passed to [wgpuDeviceCreateShaderModule].
final class WGPUShaderModuleDescriptor extends ffi.Struct {
  /// Pointer to the chained shader source descriptor (such as [WGPUShaderSourceWGSL]).
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  /// Debug label for the shader module.
  external WGPUStringView label;
}

/// Descriptor passed to [wgpuDeviceCreateComputePipeline].
final class WGPUComputePipelineDescriptor extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  /// Debug label for the compute pipeline.
  external WGPUStringView label;

  /// Pipeline layout handle, or `nullptr` for automatic layout derivation.
  external ffi.Pointer<ffi.Void> layout;

  /// Chained extension struct for the programmable compute stage.
  external ffi.Pointer<WGPUChainedStruct> computeNextInChain;

  /// Compiled `WGPUShaderModule` handle.
  external ffi.Pointer<ffi.Void> computeModule;

  /// Name of the compute entry-point function in the shader module.
  external WGPUStringView computeEntryPoint;

  /// Number of pipeline-overridable constant entries.
  @ffi.UintPtr()
  external int computeConstantCount;

  /// Pointer to pipeline-overridable constant entries.
  external ffi.Pointer<ffi.Void> computeConstants;
}

/// Single resource binding entry within a [WGPUBindGroupDescriptor].
final class WGPUBindGroupEntry extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  /// Binding index matching `@binding(n)` in WGSL.
  @ffi.Uint32()
  external int binding;

  /// `WGPUBuffer` handle to bind, or `nullptr` if not a buffer binding.
  external ffi.Pointer<ffi.Void> buffer;

  /// Byte offset within [buffer].
  @ffi.Uint64()
  external int offset;

  /// Byte length of the bound range within [buffer].
  @ffi.Uint64()
  external int size;

  /// `WGPUSampler` handle to bind, or `nullptr`.
  external ffi.Pointer<ffi.Void> sampler;

  /// `WGPUTextureView` handle to bind, or `nullptr`.
  external ffi.Pointer<ffi.Void> textureView;
}

/// Descriptor passed to [wgpuDeviceCreateBindGroup].
final class WGPUBindGroupDescriptor extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  /// Debug label for the bind group.
  external WGPUStringView label;

  /// `WGPUBindGroupLayout` handle defining the bind group schema.
  external ffi.Pointer<ffi.Void> layout;

  /// Number of entries in [entries].
  @ffi.UintPtr()
  external int entryCount;

  /// Pointer to an array of [entryCount] bind group entries.
  external ffi.Pointer<WGPUBindGroupEntry> entries;
}

/// Descriptor passed to [wgpuDeviceCreateCommandEncoder].
final class WGPUCommandEncoderDescriptor extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  /// Debug label for the command encoder.
  external WGPUStringView label;
}

/// Descriptor passed to [wgpuCommandEncoderBeginComputePass].
final class WGPUComputePassDescriptor extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  /// Debug label for the compute pass.
  external WGPUStringView label;

  /// Number of timestamp write entries.
  @ffi.UintPtr()
  external int timestampWritesCount;

  /// Pointer to timestamp write configurations, or `nullptr`.
  external ffi.Pointer<ffi.Void> timestampWrites;
}

/// Descriptor passed to [wgpuCommandEncoderFinish].
final class WGPUCommandBufferDescriptor extends ffi.Struct {
  /// Pointer to optional chained extension descriptors.
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  /// Debug label for the finished command buffer.
  external WGPUStringView label;
}

/// Dart configuration value describing a buffer binding in a bind group.
final class WgpuBindGroupEntryData {
  /// Binding slot index matching `@binding(n)` in the WGSL shader.
  final int binding;

  /// Native `WGPUBuffer` handle.
  final ffi.Pointer<ffi.Void> buffer;

  /// Byte offset within [buffer].
  final int offset;

  /// Byte length to bind from [buffer].
  final int size;

  /// Creates a [WgpuBindGroupEntryData] for a buffer binding slot.
  const WgpuBindGroupEntryData({
    required this.binding,
    required this.buffer,
    this.offset = 0,
    this.size = 0,
  });
}

extension type const _WgpuStaticState._(Object? _) {
  static ffi.Pointer<ffi.Void> lastAcquiredDevice = ffi.nullptr;
  static bool mapDone = false;
}

void _onGlobalDeviceRequested(
  int status,
  ffi.Pointer<ffi.Void> device,
  WGPUStringView message,
  ffi.Pointer<ffi.Void> userdata1,
  ffi.Pointer<ffi.Void> userdata2,
) {
  _WgpuStaticState.lastAcquiredDevice = device;
}

void _onGlobalBufferMapped(
  int status,
  WGPUStringView message,
  ffi.Pointer<ffi.Void> userdata1,
  ffi.Pointer<ffi.Void> userdata2,
) {
  _WgpuStaticState.mapDone = true;
}

// =============================================================================
// Top-Level @ffi.Native C Function Bindings (libwgpu_native)
// =============================================================================

/// Creates a new `WGPUInstance` handle from an optional [descriptor].
@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>)>()
external ffi.Pointer<ffi.Void> wgpuCreateInstance(
  ffi.Pointer<ffi.Void> descriptor,
);

/// Enumerates physical or software `WGPUAdapter` handles available on [instance].
@ffi.Native<
  ffi.UintPtr Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Pointer<ffi.Void>>,
  )
>()
external int wgpuInstanceEnumerateAdapters(
  ffi.Pointer<ffi.Void> instance,
  ffi.Pointer<ffi.Void> options,
  ffi.Pointer<ffi.Pointer<ffi.Void>> adapters,
);

/// Requests a logical `WGPUDevice` from [adapter] using [callbackInfo].
@ffi.Native<
  ffi.Uint64 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    WGPURequestDeviceCallbackInfo,
  )
>()
external int wgpuAdapterRequestDevice(
  ffi.Pointer<ffi.Void> adapter,
  ffi.Pointer<ffi.Void> descriptor,
  WGPURequestDeviceCallbackInfo callbackInfo,
);

/// Processes pending asynchronous callbacks on [instance].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuInstanceProcessEvents(ffi.Pointer<ffi.Void> instance);

/// Retrieves the default `WGPUQueue` associated with [device].
@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>)>()
external ffi.Pointer<ffi.Void> wgpuDeviceGetQueue(ffi.Pointer<ffi.Void> device);

/// Allocates a new `WGPUBuffer` on [device] matching [descriptor].
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUBufferDescriptor>,
  )
>()
external ffi.Pointer<ffi.Void> wgpuDeviceCreateBuffer(
  ffi.Pointer<ffi.Void> device,
  ffi.Pointer<WGPUBufferDescriptor> descriptor,
);

/// Schedules a write of [size] bytes from host [data] into [buffer] at [bufferOffset].
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Uint64,
    ffi.Pointer<ffi.Void>,
    ffi.UintPtr,
  )
>()
external void wgpuQueueWriteBuffer(
  ffi.Pointer<ffi.Void> queue,
  ffi.Pointer<ffi.Void> buffer,
  int bufferOffset,
  ffi.Pointer<ffi.Void> data,
  int size,
);

/// Asynchronously maps [size] bytes of [buffer] starting at [offset] for host access.
@ffi.Native<
  ffi.Uint64 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Uint32,
    ffi.UintPtr,
    ffi.UintPtr,
    WGPUBufferMapCallbackInfo,
  )
>()
external int wgpuBufferMapAsync(
  ffi.Pointer<ffi.Void> buffer,
  int mode,
  int offset,
  int size,
  WGPUBufferMapCallbackInfo callbackInfo,
);

/// Obtains a host pointer to a mapped subrange of [buffer].
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.UintPtr,
    ffi.UintPtr,
  )
>()
external ffi.Pointer<ffi.Void> wgpuBufferGetMappedRange(
  ffi.Pointer<ffi.Void> buffer,
  int offset,
  int size,
);

/// Unmaps a previously mapped [buffer] and flushes host modifications.
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuBufferUnmap(ffi.Pointer<ffi.Void> buffer);

/// Compiles a `WGPUShaderModule` on [device] from [descriptor].
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUShaderModuleDescriptor>,
  )
>()
external ffi.Pointer<ffi.Void> wgpuDeviceCreateShaderModule(
  ffi.Pointer<ffi.Void> device,
  ffi.Pointer<WGPUShaderModuleDescriptor> descriptor,
);

/// Compiles a `WGPUComputePipeline` on [device] from [descriptor].
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUComputePipelineDescriptor>,
  )
>()
external ffi.Pointer<ffi.Void> wgpuDeviceCreateComputePipeline(
  ffi.Pointer<ffi.Void> device,
  ffi.Pointer<WGPUComputePipelineDescriptor> descriptor,
);

/// Retrieves the `WGPUBindGroupLayout` for [groupIndex] of [computePipeline].
@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Uint32)>()
external ffi.Pointer<ffi.Void> wgpuComputePipelineGetBindGroupLayout(
  ffi.Pointer<ffi.Void> computePipeline,
  int groupIndex,
);

/// Creates a `WGPUBindGroup` on [device] from [descriptor].
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUBindGroupDescriptor>,
  )
>()
external ffi.Pointer<ffi.Void> wgpuDeviceCreateBindGroup(
  ffi.Pointer<ffi.Void> device,
  ffi.Pointer<WGPUBindGroupDescriptor> descriptor,
);

/// Creates a `WGPUCommandEncoder` on [device] from [descriptor].
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)
>()
external ffi.Pointer<ffi.Void> wgpuDeviceCreateCommandEncoder(
  ffi.Pointer<ffi.Void> device,
  ffi.Pointer<ffi.Void> descriptor,
);

/// Begins recording a `WGPUComputePassEncoder` on [commandEncoder].
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)
>()
external ffi.Pointer<ffi.Void> wgpuCommandEncoderBeginComputePass(
  ffi.Pointer<ffi.Void> commandEncoder,
  ffi.Pointer<ffi.Void> descriptor,
);

/// Binds [pipeline] to [computePassEncoder].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)>()
external void wgpuComputePassEncoderSetPipeline(
  ffi.Pointer<ffi.Void> computePassEncoder,
  ffi.Pointer<ffi.Void> pipeline,
);

/// Binds [group] at [groupIndex] on [computePassEncoder].
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Uint32,
    ffi.Pointer<ffi.Void>,
    ffi.UintPtr,
    ffi.Pointer<ffi.Uint32>,
  )
>()
external void wgpuComputePassEncoderSetBindGroup(
  ffi.Pointer<ffi.Void> computePassEncoder,
  int groupIndex,
  ffi.Pointer<ffi.Void> group,
  int dynamicOffsetCount,
  ffi.Pointer<ffi.Uint32> dynamicOffsets,
);

/// Dispatches compute workgroups of grid dimensions ([workgroupCountX], [workgroupCountY], [workgroupCountZ]).
@ffi.Native<
  ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Uint32, ffi.Uint32, ffi.Uint32)
>()
external void wgpuComputePassEncoderDispatchWorkgroups(
  ffi.Pointer<ffi.Void> computePassEncoder,
  int workgroupCountX,
  int workgroupCountY,
  int workgroupCountZ,
);

/// Ends the active compute pass on [computePassEncoder].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuComputePassEncoderEnd(
  ffi.Pointer<ffi.Void> computePassEncoder,
);

/// Encodes a GPU buffer-to-buffer copy of [size] bytes from [source] to [destination].
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Uint64,
    ffi.Pointer<ffi.Void>,
    ffi.Uint64,
    ffi.Uint64,
  )
>()
external void wgpuCommandEncoderCopyBufferToBuffer(
  ffi.Pointer<ffi.Void> commandEncoder,
  ffi.Pointer<ffi.Void> source,
  int sourceOffset,
  ffi.Pointer<ffi.Void> destination,
  int destinationOffset,
  int size,
);

/// Encodes a GPU zero-fill of [size] bytes on [buffer] starting at [offset].
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Uint64,
    ffi.Uint64,
  )
>()
external void wgpuCommandEncoderClearBuffer(
  ffi.Pointer<ffi.Void> commandEncoder,
  ffi.Pointer<ffi.Void> buffer,
  int offset,
  int size,
);

/// Finishes recording commands on [commandEncoder] and produces a `WGPUCommandBuffer`.
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)
>()
external ffi.Pointer<ffi.Void> wgpuCommandEncoderFinish(
  ffi.Pointer<ffi.Void> commandEncoder,
  ffi.Pointer<ffi.Void> descriptor,
);

/// Submits [commandCount] encoded [commands] to [queue] for execution.
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.UintPtr,
    ffi.Pointer<ffi.Pointer<ffi.Void>>,
  )
>()
external void wgpuQueueSubmit(
  ffi.Pointer<ffi.Void> queue,
  int commandCount,
  ffi.Pointer<ffi.Pointer<ffi.Void>> commands,
);

/// Polls [device] to advance work and optionally waits for queue completion when [wait] is non-zero.
@ffi.Native<
  ffi.Uint32 Function(ffi.Pointer<ffi.Void>, ffi.Uint32, ffi.Pointer<ffi.Void>)
>()
external int wgpuDevicePoll(
  ffi.Pointer<ffi.Void> device,
  int wait,
  ffi.Pointer<ffi.Void> wrappedSubmissionIndex,
);

/// Immediately destroys the underlying GPU memory backing [buffer].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuBufferDestroy(ffi.Pointer<ffi.Void> buffer);

/// Destroys the logical [device] and releases its hardware queue resources.
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuDeviceDestroy(ffi.Pointer<ffi.Void> device);

/// Releases a reference to [instance].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuInstanceRelease(ffi.Pointer<ffi.Void> instance);

/// Releases a reference to [adapter].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuAdapterRelease(ffi.Pointer<ffi.Void> adapter);

/// Releases a reference to [device].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuDeviceRelease(ffi.Pointer<ffi.Void> device);

/// Releases a reference to [queue].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuQueueRelease(ffi.Pointer<ffi.Void> queue);

/// Releases a reference to [buffer].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuBufferRelease(ffi.Pointer<ffi.Void> buffer);

/// Releases a reference to [shaderModule].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuShaderModuleRelease(ffi.Pointer<ffi.Void> shaderModule);

/// Releases a reference to [computePipeline].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuComputePipelineRelease(ffi.Pointer<ffi.Void> computePipeline);

/// Releases a reference to [bindGroup].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuBindGroupRelease(ffi.Pointer<ffi.Void> bindGroup);

/// Releases a reference to [bindGroupLayout].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuBindGroupLayoutRelease(ffi.Pointer<ffi.Void> bindGroupLayout);

/// Releases a reference to [commandEncoder].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuCommandEncoderRelease(ffi.Pointer<ffi.Void> commandEncoder);

/// Releases a reference to [computePassEncoder].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuComputePassEncoderRelease(
  ffi.Pointer<ffi.Void> computePassEncoder,
);

/// Releases a reference to [commandBuffer].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void wgpuCommandBufferRelease(ffi.Pointer<ffi.Void> commandBuffer);

/// High-level FFI wrapper invoking `@ffi.Native`-bound `libwgpu_native` functions.
final class WgpuNativeBindings {
  /// Whether native WebGPU bindings are linked via Native Assets.
  final bool isAvailable;

  /// Creates a [WgpuNativeBindings] facade over `@ffi.Native` WebGPU symbols.
  const WgpuNativeBindings({this.isAvailable = true});

  // ===========================================================================
  // High-Level FFI Invocation Methods
  // ===========================================================================

  /// Allocates a new `WGPUInstance` handle.
  ffi.Pointer<ffi.Void> createInstance() {
    return wgpuCreateInstance(ffi.nullptr);
  }

  /// Synchronously enumerates and selects the first available `WGPUAdapter` on [instance].
  ///
  /// Throws a [GpuDeviceException] if no WebGPU adapters are available.
  ffi.Pointer<ffi.Void> requestAdapterSync(
    ffi.Pointer<ffi.Void> instance, {
    int powerPreference = WGPUPowerPreference.highPerformance,
    int backendType = WGPUBackendType.undefined,
  }) {
    return using((arena) {
      final count = wgpuInstanceEnumerateAdapters(
        instance,
        ffi.nullptr,
        ffi.nullptr,
      );
      if (count <= 0) {
        throw const GpuDeviceException('No WebGPU adapters found.');
      }
      final adapters = arena<ffi.Pointer<ffi.Void>>(count);
      wgpuInstanceEnumerateAdapters(instance, ffi.nullptr, adapters);
      return adapters[0];
    });
  }

  /// Enumerates and selects the first available `WGPUAdapter` on [instance].
  ///
  /// Throws a [GpuDeviceException] if no WebGPU adapters are available.
  Future<ffi.Pointer<ffi.Void>> requestAdapter(
    ffi.Pointer<ffi.Void> instance, {
    int powerPreference = WGPUPowerPreference.highPerformance,
    int backendType = WGPUBackendType.undefined,
  }) async {
    return requestAdapterSync(
      instance,
      powerPreference: powerPreference,
      backendType: backendType,
    );
  }

  /// Synchronously requests a logical `WGPUDevice` from [adapter] on [instance].
  ///
  /// Throws a [GpuDeviceException] if device acquisition fails.
  ffi.Pointer<ffi.Void> requestDeviceSync(
    ffi.Pointer<ffi.Void> instance,
    ffi.Pointer<ffi.Void> adapter, {
    String? label,
  }) {
    _WgpuStaticState.lastAcquiredDevice = ffi.nullptr;

    return using((arena) {
      final callbackPointer =
          ffi.Pointer.fromFunction<
            ffi.Void Function(
              ffi.Uint32,
              ffi.Pointer<ffi.Void>,
              WGPUStringView,
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
            )
          >(_onGlobalDeviceRequested);

      final callbackInfo = arena<WGPURequestDeviceCallbackInfo>();
      callbackInfo.ref.mode = WGPUCallbackMode.allowSpontaneous;
      callbackInfo.ref.callback = callbackPointer;

      wgpuAdapterRequestDevice(adapter, ffi.nullptr, callbackInfo.ref);

      for (var i = 0; i < 100; i++) {
        wgpuInstanceProcessEvents(instance);
        if (_WgpuStaticState.lastAcquiredDevice != ffi.nullptr) break;
      }

      if (_WgpuStaticState.lastAcquiredDevice == ffi.nullptr) {
        throw const GpuDeviceException('Failed to acquire WebGPU Device.');
      }
      return _WgpuStaticState.lastAcquiredDevice;
    });
  }

  /// Requests a logical `WGPUDevice` from [adapter] on [instance].
  ///
  /// Throws a [GpuDeviceException] if device acquisition fails.
  Future<ffi.Pointer<ffi.Void>> requestDevice(
    ffi.Pointer<ffi.Void> instance,
    ffi.Pointer<ffi.Void> adapter, {
    String? label,
  }) async {
    return requestDeviceSync(instance, adapter, label: label);
  }

  /// Retrieves the default `WGPUQueue` for [device].
  ///
  /// Throws a [GpuDeviceException] if the queue handle is null.
  ffi.Pointer<ffi.Void> deviceGetQueue(ffi.Pointer<ffi.Void> device) {
    final queue = wgpuDeviceGetQueue(device);
    if (queue == ffi.nullptr) {
      throw const GpuDeviceException('Failed to retrieve device queue.');
    }
    return queue;
  }

  /// Allocates a `WGPUBuffer` of [size] bytes with [usage] flags on [device].
  ///
  /// Throws a [GpuMemoryException] if the allocation fails.
  ffi.Pointer<ffi.Void> createBuffer(
    ffi.Pointer<ffi.Void> device, {
    required int size,
    required int usage,
    bool mappedAtCreation = false,
    String? label,
  }) {
    return using((arena) {
      final descriptor = arena<WGPUBufferDescriptor>();
      descriptor.ref.nextInChain = ffi.nullptr;
      descriptor.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      descriptor.ref.label.length = label?.length ?? 0;
      descriptor.ref.usage = usage;
      descriptor.ref.size = size;
      descriptor.ref.mappedAtCreation = mappedAtCreation ? 1 : 0;

      final allocatedBuffer = wgpuDeviceCreateBuffer(device, descriptor);
      if (allocatedBuffer == ffi.nullptr) {
        throw GpuMemoryException(
          'Failed to allocate GPU buffer of size $size bytes.',
        );
      }
      return allocatedBuffer;
    });
  }

  /// Enqueues a write of [size] bytes from host [data] into [buffer] at [bufferOffset].
  void queueWriteBuffer(
    ffi.Pointer<ffi.Void> queue,
    ffi.Pointer<ffi.Void> buffer, {
    int bufferOffset = 0,
    required ffi.Pointer<ffi.Void> data,
    required int size,
  }) {
    if (size == 0) return;
    wgpuQueueWriteBuffer(queue, buffer, bufferOffset, data, size);
  }

  /// Synchronously maps [size] bytes of [buffer] starting at [offset] for host access.
  void bufferMapSync(
    ffi.Pointer<ffi.Void> instance,
    ffi.Pointer<ffi.Void> buffer, {
    ffi.Pointer<ffi.Void>? device,
    int mode = WGPUMapMode.read,
    int offset = 0,
    required int size,
  }) {
    _WgpuStaticState.mapDone = false;

    using((arena) {
      final callbackPointer =
          ffi.Pointer.fromFunction<
            ffi.Void Function(
              ffi.Uint32,
              WGPUStringView,
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
            )
          >(_onGlobalBufferMapped);

      final callbackInfo = arena<WGPUBufferMapCallbackInfo>();
      callbackInfo.ref.mode = WGPUCallbackMode.allowSpontaneous;
      callbackInfo.ref.callback = callbackPointer;

      wgpuBufferMapAsync(buffer, mode, offset, size, callbackInfo.ref);

      for (var i = 0; i < 1000; i++) {
        if (device != null && device != ffi.nullptr) {
          devicePoll(device, wait: true);
        }
        wgpuInstanceProcessEvents(instance);
        if (_WgpuStaticState.mapDone) break;
      }
    });
  }

  /// Maps [size] bytes of [buffer] starting at [offset] for host access.
  Future<int> bufferMapAsync(
    ffi.Pointer<ffi.Void> instance,
    ffi.Pointer<ffi.Void> buffer, {
    ffi.Pointer<ffi.Void>? device,
    int mode = WGPUMapMode.read,
    int offset = 0,
    required int size,
  }) async {
    bufferMapSync(
      instance,
      buffer,
      device: device,
      mode: mode,
      offset: offset,
      size: size,
    );
    return 1;
  }

  /// Obtains the host pointer for a mapped range of [buffer] of [size] bytes at [offset].
  ffi.Pointer<ffi.Void> bufferGetMappedRange(
    ffi.Pointer<ffi.Void> buffer, {
    int offset = 0,
    required int size,
  }) {
    return wgpuBufferGetMappedRange(buffer, offset, size);
  }

  /// Unmaps [buffer] after host access is complete.
  void bufferUnmap(ffi.Pointer<ffi.Void> buffer) {
    wgpuBufferUnmap(buffer);
  }

  /// Compiles [wgslSource] into a `WGPUShaderModule` on [device].
  ///
  /// Throws a [GpuComputeException] if compilation fails.
  ffi.Pointer<ffi.Void> createShaderModule(
    ffi.Pointer<ffi.Void> device,
    String wgslSource, {
    String? label,
  }) {
    return using((arena) {
      final wgslChain = arena<WGPUShaderSourceWGSL>();
      wgslChain.ref.chain.next = ffi.nullptr;
      wgslChain.ref.chain.sType = WGPUSType.shaderSourceWGSL;
      wgslChain.ref.code.data = wgslSource.toNativeUtf8(allocator: arena);
      wgslChain.ref.code.length = wgslSource.length;

      final descriptor = arena<WGPUShaderModuleDescriptor>();
      descriptor.ref.nextInChain = wgslChain.cast<WGPUChainedStruct>();
      descriptor.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      descriptor.ref.label.length = label?.length ?? 0;

      final module = wgpuDeviceCreateShaderModule(device, descriptor);
      if (module == ffi.nullptr) {
        throw const GpuComputeException(
          'Failed to compile WGSL shader module.',
        );
      }
      return module;
    });
  }

  /// Creates a `WGPUComputePipeline` on [device] for [shaderModule] and [entryPoint].
  ///
  /// Throws a [GpuComputeException] if pipeline creation fails.
  ffi.Pointer<ffi.Void> createComputePipeline(
    ffi.Pointer<ffi.Void> device, {
    required ffi.Pointer<ffi.Void> shaderModule,
    String entryPoint = 'main',
    ffi.Pointer<ffi.Void>? layout,
    String? label,
  }) {
    return using((arena) {
      final descriptor = arena<WGPUComputePipelineDescriptor>();
      descriptor.ref.nextInChain = ffi.nullptr;
      descriptor.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      descriptor.ref.label.length = label?.length ?? 0;
      descriptor.ref.layout = layout ?? ffi.nullptr;
      descriptor.ref.computeNextInChain = ffi.nullptr;
      descriptor.ref.computeModule = shaderModule;
      descriptor.ref.computeEntryPoint.data = entryPoint.toNativeUtf8(
        allocator: arena,
      );
      descriptor.ref.computeEntryPoint.length = entryPoint.length;
      descriptor.ref.computeConstantCount = 0;
      descriptor.ref.computeConstants = ffi.nullptr;

      final pipeline = wgpuDeviceCreateComputePipeline(device, descriptor);
      if (pipeline == ffi.nullptr) {
        throw const GpuComputeException('Failed to create compute pipeline.');
      }
      return pipeline;
    });
  }

  /// Retrieves the `WGPUBindGroupLayout` at [groupIndex] from [pipeline].
  ffi.Pointer<ffi.Void> pipelineGetBindGroupLayout(
    ffi.Pointer<ffi.Void> pipeline,
    int groupIndex,
  ) {
    return wgpuComputePipelineGetBindGroupLayout(pipeline, groupIndex);
  }

  /// Creates a `WGPUBindGroup` on [device] binding [entries] to [layout].
  ///
  /// Throws a [GpuComputeException] if bind group creation fails.
  ffi.Pointer<ffi.Void> createBindGroup(
    ffi.Pointer<ffi.Void> device, {
    required ffi.Pointer<ffi.Void> layout,
    required List<WgpuBindGroupEntryData> entries,
    String? label,
  }) {
    return using((arena) {
      final entriesPtr = arena<WGPUBindGroupEntry>(entries.length);
      for (var i = 0; i < entries.length; i++) {
        final entry = entries[i];
        final entryPtr = entriesPtr + i;
        entryPtr.ref.nextInChain = ffi.nullptr;
        entryPtr.ref.binding = entry.binding;
        entryPtr.ref.buffer = entry.buffer;
        entryPtr.ref.offset = entry.offset;
        entryPtr.ref.size = entry.size;
        entryPtr.ref.sampler = ffi.nullptr;
        entryPtr.ref.textureView = ffi.nullptr;
      }

      final descriptor = arena<WGPUBindGroupDescriptor>();
      descriptor.ref.nextInChain = ffi.nullptr;
      descriptor.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      descriptor.ref.label.length = label?.length ?? 0;
      descriptor.ref.layout = layout;
      descriptor.ref.entryCount = entries.length;
      descriptor.ref.entries = entriesPtr;

      final bindGroup = wgpuDeviceCreateBindGroup(device, descriptor);
      if (bindGroup == ffi.nullptr) {
        throw const GpuComputeException('Failed to create bind group.');
      }
      return bindGroup;
    });
  }

  /// Creates a `WGPUCommandEncoder` on [device].
  ///
  /// Throws a [GpuComputeException] if command encoder creation fails.
  ffi.Pointer<ffi.Void> createCommandEncoder(
    ffi.Pointer<ffi.Void> device, {
    String? label,
  }) {
    return using((arena) {
      final descriptor = arena<WGPUCommandEncoderDescriptor>();
      descriptor.ref.nextInChain = ffi.nullptr;
      descriptor.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      descriptor.ref.label.length = label?.length ?? 0;

      final encoder = wgpuDeviceCreateCommandEncoder(device, descriptor.cast());
      if (encoder == ffi.nullptr) {
        throw const GpuComputeException('Failed to create command encoder.');
      }
      return encoder;
    });
  }

  /// Begins a `WGPUComputePassEncoder` on [encoder].
  ///
  /// Throws a [GpuComputeException] if the compute pass cannot be started.
  ffi.Pointer<ffi.Void> commandEncoderBeginComputePass(
    ffi.Pointer<ffi.Void> encoder, {
    String? label,
  }) {
    return using((arena) {
      final descriptor = arena<WGPUComputePassDescriptor>();
      descriptor.ref.nextInChain = ffi.nullptr;
      descriptor.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      descriptor.ref.label.length = label?.length ?? 0;
      descriptor.ref.timestampWritesCount = 0;
      descriptor.ref.timestampWrites = ffi.nullptr;

      final pass = wgpuCommandEncoderBeginComputePass(
        encoder,
        descriptor.cast(),
      );
      if (pass == ffi.nullptr) {
        throw const GpuComputeException('Failed to begin compute pass.');
      }
      return pass;
    });
  }

  /// Binds [pipeline] to the active compute [pass].
  void computePassSetPipeline(
    ffi.Pointer<ffi.Void> pass,
    ffi.Pointer<ffi.Void> pipeline,
  ) {
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
  }

  /// Binds [bindGroup] at [groupIndex] on the active compute [pass].
  void computePassSetBindGroup(
    ffi.Pointer<ffi.Void> pass,
    int groupIndex,
    ffi.Pointer<ffi.Void> bindGroup,
  ) {
    wgpuComputePassEncoderSetBindGroup(
      pass,
      groupIndex,
      bindGroup,
      0,
      ffi.nullptr,
    );
  }

  /// Dispatches compute workgroups of dimensions ([workgroupsX], [workgroupsY], [workgroupsZ]) on [pass].
  void computePassDispatchWorkgroups(
    ffi.Pointer<ffi.Void> pass,
    int workgroupsX,
    int workgroupsY,
    int workgroupsZ,
  ) {
    wgpuComputePassEncoderDispatchWorkgroups(
      pass,
      workgroupsX,
      workgroupsY,
      workgroupsZ,
    );
  }

  /// Ends the active compute [pass].
  void computePassEnd(ffi.Pointer<ffi.Void> pass) {
    wgpuComputePassEncoderEnd(pass);
  }

  /// Encodes a GPU copy of [size] bytes from [source] at [sourceOffset] to [destination] at [destinationOffset].
  void commandEncoderCopyBufferToBuffer(
    ffi.Pointer<ffi.Void> encoder,
    ffi.Pointer<ffi.Void> source,
    int sourceOffset,
    ffi.Pointer<ffi.Void> destination,
    int destinationOffset,
    int size,
  ) {
    wgpuCommandEncoderCopyBufferToBuffer(
      encoder,
      source,
      sourceOffset,
      destination,
      destinationOffset,
      size,
    );
  }

  /// Encodes a GPU zero-fill of [size] bytes on [buffer] starting at [offset].
  void commandEncoderClearBuffer(
    ffi.Pointer<ffi.Void> encoder,
    ffi.Pointer<ffi.Void> buffer,
    int offset,
    int size,
  ) {
    wgpuCommandEncoderClearBuffer(encoder, buffer, offset, size);
  }

  /// Finishes recording commands on [encoder] and produces a `WGPUCommandBuffer`.
  ///
  /// Throws a [GpuComputeException] if finishing the encoder fails.
  ffi.Pointer<ffi.Void> commandEncoderFinish(
    ffi.Pointer<ffi.Void> encoder, {
    String? label,
  }) {
    return using((arena) {
      final descriptor = arena<WGPUCommandBufferDescriptor>();
      descriptor.ref.nextInChain = ffi.nullptr;
      descriptor.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      descriptor.ref.label.length = label?.length ?? 0;

      final commandBuffer = wgpuCommandEncoderFinish(
        encoder,
        descriptor.cast(),
      );
      if (commandBuffer == ffi.nullptr) {
        throw const GpuComputeException('Failed to finish command encoder.');
      }
      return commandBuffer;
    });
  }

  /// Submits [commandBuffers] to [queue] for execution.
  void queueSubmit(
    ffi.Pointer<ffi.Void> queue,
    List<ffi.Pointer<ffi.Void>> commandBuffers,
  ) {
    if (commandBuffers.isEmpty) return;
    using((arena) {
      final array = arena<ffi.Pointer<ffi.Void>>(commandBuffers.length);
      for (var i = 0; i < commandBuffers.length; i++) {
        array[i] = commandBuffers[i];
      }
      wgpuQueueSubmit(queue, commandBuffers.length, array);
    });
  }

  /// Polls [device] to process in-flight GPU work, optionally blocking when [wait] is `true`.
  bool devicePoll(ffi.Pointer<ffi.Void> device, {bool wait = false}) {
    return wgpuDevicePoll(device, wait ? 1 : 0, ffi.nullptr) != 0;
  }

  /// Destroys the GPU allocation backing [buffer].
  void bufferDestroy(ffi.Pointer<ffi.Void> buffer) {
    if (buffer != ffi.nullptr) {
      wgpuBufferDestroy(buffer);
    }
  }

  /// Destroys the logical [device].
  void deviceDestroy(ffi.Pointer<ffi.Void> device) {
    if (device != ffi.nullptr) {
      wgpuDeviceDestroy(device);
    }
  }

  /// Releases a reference to [instance].
  void instanceRelease(ffi.Pointer<ffi.Void> instance) {
    if (instance != ffi.nullptr) {
      wgpuInstanceRelease(instance);
    }
  }

  /// Releases a reference to [adapter].
  void adapterRelease(ffi.Pointer<ffi.Void> adapter) {
    if (adapter != ffi.nullptr) {
      wgpuAdapterRelease(adapter);
    }
  }

  /// Releases a reference to [device].
  void deviceRelease(ffi.Pointer<ffi.Void> device) {
    if (device != ffi.nullptr) {
      wgpuDeviceRelease(device);
    }
  }

  /// Releases a reference to [queue].
  void queueRelease(ffi.Pointer<ffi.Void> queue) {
    if (queue != ffi.nullptr) {
      wgpuQueueRelease(queue);
    }
  }

  /// Releases a reference to [buffer].
  void bufferRelease(ffi.Pointer<ffi.Void> buffer) {
    if (buffer != ffi.nullptr) {
      wgpuBufferRelease(buffer);
    }
  }

  /// Releases a reference to [shaderModule].
  void shaderModuleRelease(ffi.Pointer<ffi.Void> shaderModule) {
    if (shaderModule != ffi.nullptr) {
      wgpuShaderModuleRelease(shaderModule);
    }
  }

  /// Releases a reference to [computePipeline].
  void computePipelineRelease(ffi.Pointer<ffi.Void> computePipeline) {
    if (computePipeline != ffi.nullptr) {
      wgpuComputePipelineRelease(computePipeline);
    }
  }

  /// Releases a reference to [bindGroup].
  void bindGroupRelease(ffi.Pointer<ffi.Void> bindGroup) {
    if (bindGroup != ffi.nullptr) {
      wgpuBindGroupRelease(bindGroup);
    }
  }

  /// Releases a reference to [bindGroupLayout].
  void bindGroupLayoutRelease(ffi.Pointer<ffi.Void> bindGroupLayout) {
    if (bindGroupLayout != ffi.nullptr) {
      wgpuBindGroupLayoutRelease(bindGroupLayout);
    }
  }

  /// Releases a reference to [commandEncoder].
  void commandEncoderRelease(ffi.Pointer<ffi.Void> commandEncoder) {
    if (commandEncoder != ffi.nullptr) {
      wgpuCommandEncoderRelease(commandEncoder);
    }
  }

  /// Releases a reference to [computePassEncoder].
  void computePassEncoderRelease(ffi.Pointer<ffi.Void> computePassEncoder) {
    if (computePassEncoder != ffi.nullptr) {
      wgpuComputePassEncoderRelease(computePassEncoder);
    }
  }

  /// Releases a reference to [commandBuffer].
  void commandBufferRelease(ffi.Pointer<ffi.Void> commandBuffer) {
    if (commandBuffer != ffi.nullptr) {
      wgpuCommandBufferRelease(commandBuffer);
    }
  }
}
