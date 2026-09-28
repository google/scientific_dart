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
import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';
import '../../exceptions.dart';

// =============================================================================
// WebGPU Standard Usage and Enumeration Constants
// =============================================================================

extension type const WGPUBufferUsage._(int value) implements int {
  static const int none = 0x00000000;
  static const int mapRead = 0x00000001;
  static const int mapWrite = 0x00000002;
  static const int copySrc = 0x00000004;
  static const int copyDst = 0x00000008;
  static const int index = 0x00000010;
  static const int vertex = 0x00000020;
  static const int uniform = 0x00000040;
  static const int storage = 0x00000080;
  static const int indirect = 0x00000100;
  static const int queryResolve = 0x00000200;
}

const int WGPUBufferUsage_None = WGPUBufferUsage.none;
const int WGPUBufferUsage_MapRead = WGPUBufferUsage.mapRead;
const int WGPUBufferUsage_MapWrite = WGPUBufferUsage.mapWrite;
const int WGPUBufferUsage_CopySrc = WGPUBufferUsage.copySrc;
const int WGPUBufferUsage_CopyDst = WGPUBufferUsage.copyDst;
const int WGPUBufferUsage_Index = WGPUBufferUsage.index;
const int WGPUBufferUsage_Vertex = WGPUBufferUsage.vertex;
const int WGPUBufferUsage_Uniform = WGPUBufferUsage.uniform;
const int WGPUBufferUsage_Storage = WGPUBufferUsage.storage;
const int WGPUBufferUsage_Indirect = WGPUBufferUsage.indirect;
const int WGPUBufferUsage_QueryResolve = WGPUBufferUsage.queryResolve;

extension type const WGPUMapMode._(int value) implements int {
  static const int none = 0x00000000;
  static const int read = 0x00000001;
  static const int write = 0x00000002;
}

const int WGPUMapMode_None = WGPUMapMode.none;
const int WGPUMapMode_Read = WGPUMapMode.read;
const int WGPUMapMode_Write = WGPUMapMode.write;

extension type const WGPUSType._(int value) implements int {
  static const int invalid = 0x00000000;
  static const int surfaceDescriptorFromMetalLayer = 0x00000001;
  static const int surfaceDescriptorFromWindowsHWND = 0x00000002;
  static const int surfaceDescriptorFromXlibWindow = 0x00000003;
  static const int surfaceDescriptorFromCanvasHTMLSelector = 0x00000004;
  static const int shaderSourceWGSL = 0x00000002;
  static const int shaderModuleWGSLDescriptor = 0x00000006;
}

const int WGPUSType_Invalid = WGPUSType.invalid;
const int WGPUSType_ShaderSourceWGSL = WGPUSType.shaderSourceWGSL;
const int WGPUSType_ShaderModuleWGSLDescriptor =
    WGPUSType.shaderModuleWGSLDescriptor;

extension type const WGPUCallbackMode._(int value) implements int {
  static const int waitAnyOnly = 0x00000001;
  static const int allowProcessEvents = 0x00000002;
  static const int allowSpontaneous = 0x00000004;
}

extension type const WGPUPowerPreference._(int value) implements int {
  static const int undefined = 0x00000000;
  static const int lowPower = 0x00000001;
  static const int highPerformance = 0x00000002;
}

const int WGPUPowerPreference_Undefined = WGPUPowerPreference.undefined;
const int WGPUPowerPreference_LowPower = WGPUPowerPreference.lowPower;
const int WGPUPowerPreference_HighPerformance =
    WGPUPowerPreference.highPerformance;

extension type const WGPUBackendType._(int value) implements int {
  static const int undefined = 0x00000000;
  static const int nullBackend = 0x00000001;
  static const int webGpu = 0x00000002;
  static const int d3d11 = 0x00000003;
  static const int d3d12 = 0x00000004;
  static const int metal = 0x00000005;
  static const int vulkan = 0x00000006;
  static const int openGl = 0x00000007;
  static const int openGlEs = 0x00000008;
}

// =============================================================================
// WebGPU C-FFI Native Struct Definitions
// =============================================================================

final class WGPUStringView extends ffi.Struct {
  external ffi.Pointer<Utf8> data;

  @ffi.UintPtr()
  external int length;
}

final class WGPUChainedStruct extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> next;

  @ffi.Uint32()
  external int sType;
}

final class WGPUInstanceDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
}

final class WGPURequestDeviceCallbackInfo extends ffi.Struct {
  external ffi.Pointer<ffi.Void> nextInChain;

  @ffi.Uint32()
  external int mode;

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

  external ffi.Pointer<ffi.Void> userdata1;
  external ffi.Pointer<ffi.Void> userdata2;
}

final class WGPUBufferMapCallbackInfo extends ffi.Struct {
  external ffi.Pointer<ffi.Void> nextInChain;

  @ffi.Uint32()
  external int mode;

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

  external ffi.Pointer<ffi.Void> userdata1;
  external ffi.Pointer<ffi.Void> userdata2;
}

final class WGPUBufferDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external WGPUStringView label;

  @ffi.Uint64()
  external int usage;

  @ffi.Uint64()
  external int size;

  @ffi.Uint32()
  external int mappedAtCreation;
}

final class WGPUShaderSourceWGSL extends ffi.Struct {
  external WGPUChainedStruct chain;
  external WGPUStringView code;
}

final class WGPUShaderModuleDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external WGPUStringView label;
}

final class WGPUComputePipelineDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external WGPUStringView label;
  external ffi.Pointer<ffi.Void> layout;
  external ffi.Pointer<WGPUChainedStruct> computeNextInChain;
  external ffi.Pointer<ffi.Void> computeModule;
  external WGPUStringView computeEntryPoint;
  @ffi.UintPtr()
  external int computeConstantCount;
  external ffi.Pointer<ffi.Void> computeConstants;
}

final class WGPUBindGroupEntry extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;

  @ffi.Uint32()
  external int binding;

  external ffi.Pointer<ffi.Void> buffer;

  @ffi.Uint64()
  external int offset;

  @ffi.Uint64()
  external int size;

  external ffi.Pointer<ffi.Void> sampler;
  external ffi.Pointer<ffi.Void> textureView;
}

final class WGPUBindGroupDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external WGPUStringView label;
  external ffi.Pointer<ffi.Void> layout;

  @ffi.UintPtr()
  external int entryCount;

  external ffi.Pointer<WGPUBindGroupEntry> entries;
}

final class WGPUCommandEncoderDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external WGPUStringView label;
}

final class WGPUComputePassDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external WGPUStringView label;

  @ffi.UintPtr()
  external int timestampWritesCount;

  external ffi.Pointer<ffi.Void> timestampWrites;
}

final class WGPUCommandBufferDescriptor extends ffi.Struct {
  external ffi.Pointer<WGPUChainedStruct> nextInChain;
  external WGPUStringView label;
}

final class WgpuBindGroupEntryData {
  final int binding;
  final ffi.Pointer<ffi.Void> buffer;
  final int offset;
  final int size;

  const WgpuBindGroupEntryData({
    required this.binding,
    required this.buffer,
    this.offset = 0,
    this.size = 0,
  });
}

// =============================================================================
// WebGPU Dynamic Native Library Loader & Driver Interface
// =============================================================================

extension type const _WgpuStaticState._(Object? _) {
  static ffi.Pointer<ffi.Void> lastAcquiredDevice = ffi.nullptr;
  static bool mapDone = false;
}

void _onGlobalDeviceRequested(
  int status,
  ffi.Pointer<ffi.Void> device,
  WGPUStringView message,
  ffi.Pointer<ffi.Void> u1,
  ffi.Pointer<ffi.Void> u2,
) {
  _WgpuStaticState.lastAcquiredDevice = device;
}

void _onGlobalBufferMapped(
  int status,
  WGPUStringView message,
  ffi.Pointer<ffi.Void> u1,
  ffi.Pointer<ffi.Void> u2,
) {
  _WgpuStaticState.mapDone = true;
}

final class WgpuNativeLib {
  final ffi.DynamicLibrary dylib;
  final String libraryPath;

  // C function pointers
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>)
  _wgpuCreateInstance;
  late final int Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Pointer<ffi.Void>>,
  )
  _wgpuInstanceEnumerateAdapters;
  late final int Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    WGPURequestDeviceCallbackInfo,
  )
  _wgpuAdapterRequestDevice;
  late final void Function(ffi.Pointer<ffi.Void>) _wgpuInstanceProcessEvents;
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>)
  _wgpuDeviceGetQueue;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUBufferDescriptor>,
  )
  _wgpuDeviceCreateBuffer;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Void>,
    int,
  )
  _wgpuQueueWriteBuffer;
  late final int Function(
    ffi.Pointer<ffi.Void>,
    int,
    int,
    int,
    WGPUBufferMapCallbackInfo,
  )?
  _wgpuBufferMapAsync;
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int, int)?
  _wgpuBufferGetMappedRange;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuBufferUnmap;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUShaderModuleDescriptor>,
  )
  _wgpuDeviceCreateShaderModule;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUComputePipelineDescriptor>,
  )
  _wgpuDeviceCreateComputePipeline;
  late final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int)?
  _wgpuComputePipelineGetBindGroupLayout;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<WGPUBindGroupDescriptor>,
  )
  _wgpuDeviceCreateBindGroup;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
  )
  _wgpuDeviceCreateCommandEncoder;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
  )
  _wgpuCommandEncoderBeginComputePass;
  late final void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)
  _wgpuComputePassEncoderSetPipeline;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Uint32>,
  )
  _wgpuComputePassEncoderSetBindGroup;
  late final void Function(ffi.Pointer<ffi.Void>, int, int, int)
  _wgpuComputePassEncoderDispatchWorkgroups;
  late final void Function(ffi.Pointer<ffi.Void>) _wgpuComputePassEncoderEnd;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Void>,
    int,
    int,
  )?
  _wgpuCommandEncoderCopyBufferToBuffer;
  late final ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
  )
  _wgpuCommandEncoderFinish;
  late final void Function(
    ffi.Pointer<ffi.Void>,
    int,
    ffi.Pointer<ffi.Pointer<ffi.Void>>,
  )
  _wgpuQueueSubmit;
  late final int Function(ffi.Pointer<ffi.Void>, int, ffi.Pointer<ffi.Void>)?
  _wgpuDevicePoll;
  late final void Function(ffi.Pointer<ffi.Void>) _wgpuBufferDestroy;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuDeviceDestroy;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuInstanceRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuAdapterRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuDeviceRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuQueueRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuBufferRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuShaderModuleRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuComputePipelineRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuBindGroupRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuBindGroupLayoutRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuCommandEncoderRelease;
  late final void Function(ffi.Pointer<ffi.Void>)?
  _wgpuComputePassEncoderRelease;
  late final void Function(ffi.Pointer<ffi.Void>)? _wgpuCommandBufferRelease;

  bool _isAvailable = false;
  bool get isAvailable => _isAvailable;

  WgpuNativeLib(this.dylib, {required this.libraryPath}) {
    try {
      _lookupSymbols();
      _isAvailable = true;
    } catch (_) {
      _isAvailable = false;
    }
  }

  static T? _tryLookup<T>(T Function() lookupFn) {
    try {
      return lookupFn();
    } catch (_) {
      return null;
    }
  }

  void _lookupSymbols() {
    _wgpuCreateInstance = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>)
          >
        >("wgpuCreateInstance")
        .asFunction();

    _wgpuInstanceEnumerateAdapters = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.UintPtr Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Pointer<ffi.Void>>,
            )
          >
        >("wgpuInstanceEnumerateAdapters")
        .asFunction();

    _wgpuAdapterRequestDevice = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Uint64 Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
              WGPURequestDeviceCallbackInfo,
            )
          >
        >("wgpuAdapterRequestDevice")
        .asFunction();

    _wgpuInstanceProcessEvents = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
          "wgpuInstanceProcessEvents",
        )
        .asFunction();

    _wgpuDeviceGetQueue = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>)
          >
        >("wgpuDeviceGetQueue")
        .asFunction();

    _wgpuDeviceCreateBuffer = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<WGPUBufferDescriptor>,
            )
          >
        >("wgpuDeviceCreateBuffer")
        .asFunction();

    _wgpuQueueWriteBuffer = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
              ffi.Uint64,
              ffi.Pointer<ffi.Void>,
              ffi.UintPtr,
            )
          >
        >("wgpuQueueWriteBuffer")
        .asFunction();

    _wgpuBufferMapAsync = _tryLookup(
      () => dylib
          .lookup<
            ffi.NativeFunction<
              ffi.Uint64 Function(
                ffi.Pointer<ffi.Void>,
                ffi.Uint32,
                ffi.UintPtr,
                ffi.UintPtr,
                WGPUBufferMapCallbackInfo,
              )
            >
          >("wgpuBufferMapAsync")
          .asFunction(),
    );

    _wgpuBufferGetMappedRange = _tryLookup(
      () => dylib
          .lookup<
            ffi.NativeFunction<
              ffi.Pointer<ffi.Void> Function(
                ffi.Pointer<ffi.Void>,
                ffi.UintPtr,
                ffi.UintPtr,
              )
            >
          >("wgpuBufferGetMappedRange")
          .asFunction(),
    );

    _wgpuBufferUnmap = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuBufferUnmap",
          )
          .asFunction(),
    );

    _wgpuDeviceCreateShaderModule = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<WGPUShaderModuleDescriptor>,
            )
          >
        >("wgpuDeviceCreateShaderModule")
        .asFunction();

    _wgpuDeviceCreateComputePipeline = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<WGPUComputePipelineDescriptor>,
            )
          >
        >("wgpuDeviceCreateComputePipeline")
        .asFunction();

    _wgpuComputePipelineGetBindGroupLayout = _tryLookup(
      () => dylib
          .lookup<
            ffi.NativeFunction<
              ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Uint32)
            >
          >("wgpuComputePipelineGetBindGroupLayout")
          .asFunction(),
    );

    _wgpuDeviceCreateBindGroup = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<WGPUBindGroupDescriptor>,
            )
          >
        >("wgpuDeviceCreateBindGroup")
        .asFunction();

    _wgpuDeviceCreateCommandEncoder = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
            )
          >
        >("wgpuDeviceCreateCommandEncoder")
        .asFunction();

    _wgpuCommandEncoderBeginComputePass = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
            )
          >
        >("wgpuCommandEncoderBeginComputePass")
        .asFunction();

    _wgpuComputePassEncoderSetPipeline = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)
          >
        >("wgpuComputePassEncoderSetPipeline")
        .asFunction();

    _wgpuComputePassEncoderSetBindGroup = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(
              ffi.Pointer<ffi.Void>,
              ffi.Uint32,
              ffi.Pointer<ffi.Void>,
              ffi.UintPtr,
              ffi.Pointer<ffi.Uint32>,
            )
          >
        >("wgpuComputePassEncoderSetBindGroup")
        .asFunction();

    _wgpuComputePassEncoderDispatchWorkgroups = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(
              ffi.Pointer<ffi.Void>,
              ffi.Uint32,
              ffi.Uint32,
              ffi.Uint32,
            )
          >
        >("wgpuComputePassEncoderDispatchWorkgroups")
        .asFunction();

    _wgpuComputePassEncoderEnd = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
          "wgpuComputePassEncoderEnd",
        )
        .asFunction();

    _wgpuCommandEncoderCopyBufferToBuffer = _tryLookup(
      () => dylib
          .lookup<
            ffi.NativeFunction<
              ffi.Void Function(
                ffi.Pointer<ffi.Void>,
                ffi.Pointer<ffi.Void>,
                ffi.Uint64,
                ffi.Pointer<ffi.Void>,
                ffi.Uint64,
                ffi.Uint64,
              )
            >
          >("wgpuCommandEncoderCopyBufferToBuffer")
          .asFunction(),
    );

    _wgpuCommandEncoderFinish = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Pointer<ffi.Void> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
            )
          >
        >("wgpuCommandEncoderFinish")
        .asFunction();

    _wgpuQueueSubmit = dylib
        .lookup<
          ffi.NativeFunction<
            ffi.Void Function(
              ffi.Pointer<ffi.Void>,
              ffi.UintPtr,
              ffi.Pointer<ffi.Pointer<ffi.Void>>,
            )
          >
        >("wgpuQueueSubmit")
        .asFunction();

    _wgpuDevicePoll = _tryLookup(
      () => dylib
          .lookup<
            ffi.NativeFunction<
              ffi.Uint32 Function(
                ffi.Pointer<ffi.Void>,
                ffi.Uint32,
                ffi.Pointer<ffi.Void>,
              )
            >
          >("wgpuDevicePoll")
          .asFunction(),
    );

    _wgpuBufferDestroy = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
          "wgpuBufferDestroy",
        )
        .asFunction();

    _wgpuDeviceDestroy = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuDeviceDestroy",
          )
          .asFunction(),
    );

    _wgpuInstanceRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuInstanceRelease",
          )
          .asFunction(),
    );

    _wgpuAdapterRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuAdapterRelease",
          )
          .asFunction(),
    );

    _wgpuDeviceRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuDeviceRelease",
          )
          .asFunction(),
    );

    _wgpuQueueRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuQueueRelease",
          )
          .asFunction(),
    );

    _wgpuBufferRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuBufferRelease",
          )
          .asFunction(),
    );

    _wgpuShaderModuleRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuShaderModuleRelease",
          )
          .asFunction(),
    );

    _wgpuComputePipelineRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuComputePipelineRelease",
          )
          .asFunction(),
    );

    _wgpuBindGroupRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuBindGroupRelease",
          )
          .asFunction(),
    );

    _wgpuBindGroupLayoutRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuBindGroupLayoutRelease",
          )
          .asFunction(),
    );

    _wgpuCommandEncoderRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuCommandEncoderRelease",
          )
          .asFunction(),
    );

    _wgpuComputePassEncoderRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuComputePassEncoderRelease",
          )
          .asFunction(),
    );

    _wgpuCommandBufferRelease = _tryLookup(
      () => dylib
          .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            "wgpuCommandBufferRelease",
          )
          .asFunction(),
    );
  }

  // ===========================================================================
  // High-Level FFI Invocation Methods
  // ===========================================================================

  ffi.Pointer<ffi.Void> createInstance() {
    return _wgpuCreateInstance(ffi.nullptr);
  }

  Future<ffi.Pointer<ffi.Void>> requestAdapter(
    ffi.Pointer<ffi.Void> instance, {
    int powerPreference = WGPUPowerPreference.highPerformance,
    int backendType = WGPUBackendType.undefined,
  }) async {
    return using((arena) {
      final count = _wgpuInstanceEnumerateAdapters(
        instance,
        ffi.nullptr,
        ffi.nullptr,
      );
      if (count <= 0) {
        throw GpuDeviceException("No WebGPU adapters found.");
      }
      final adapters = arena<ffi.Pointer<ffi.Void>>(count);
      _wgpuInstanceEnumerateAdapters(instance, ffi.nullptr, adapters);
      return adapters[0];
    });
  }

  Future<ffi.Pointer<ffi.Void>> requestDevice(
    ffi.Pointer<ffi.Void> instance,
    ffi.Pointer<ffi.Void> adapter, {
    String? label,
  }) async {
    _WgpuStaticState.lastAcquiredDevice = ffi.nullptr;

    return using((arena) {
      final cbPointer =
          ffi.Pointer.fromFunction<
            ffi.Void Function(
              ffi.Uint32,
              ffi.Pointer<ffi.Void>,
              WGPUStringView,
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
            )
          >(_onGlobalDeviceRequested);

      final cbInfo = arena<WGPURequestDeviceCallbackInfo>();
      cbInfo.ref.mode = WGPUCallbackMode.allowSpontaneous;
      cbInfo.ref.callback = cbPointer;

      _wgpuAdapterRequestDevice(adapter, ffi.nullptr, cbInfo.ref);

      for (var i = 0; i < 100; i++) {
        _wgpuInstanceProcessEvents(instance);
        if (_WgpuStaticState.lastAcquiredDevice != ffi.nullptr) break;
      }

      if (_WgpuStaticState.lastAcquiredDevice == ffi.nullptr) {
        throw GpuDeviceException("Failed to acquire WebGPU Device.");
      }
      return _WgpuStaticState.lastAcquiredDevice;
    });
  }

  ffi.Pointer<ffi.Void> deviceGetQueue(ffi.Pointer<ffi.Void> device) {
    final queue = _wgpuDeviceGetQueue(device);
    if (queue == ffi.nullptr) {
      throw GpuDeviceException("Failed to retrieve device queue.");
    }
    return queue;
  }

  ffi.Pointer<ffi.Void> createBuffer(
    ffi.Pointer<ffi.Void> device, {
    required int size,
    required int usage,
    bool mappedAtCreation = false,
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUBufferDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      desc.ref.label.length = label?.length ?? 0;
      desc.ref.usage = usage;
      desc.ref.size = size;
      desc.ref.mappedAtCreation = mappedAtCreation ? 1 : 0;

      final buf = _wgpuDeviceCreateBuffer(device, desc);
      if (buf == ffi.nullptr) {
        throw GpuMemoryException(
          "Failed to allocate GPU buffer of size $size bytes.",
        );
      }
      return buf;
    });
  }

  void queueWriteBuffer(
    ffi.Pointer<ffi.Void> queue,
    ffi.Pointer<ffi.Void> buffer, {
    int bufferOffset = 0,
    required ffi.Pointer<ffi.Void> data,
    required int size,
  }) {
    if (size == 0) return;
    _wgpuQueueWriteBuffer(queue, buffer, bufferOffset, data, size);
  }

  void bufferMapSync(
    ffi.Pointer<ffi.Void> instance,
    ffi.Pointer<ffi.Void> buffer, {
    ffi.Pointer<ffi.Void>? device,
    int mode = WGPUMapMode.read,
    int offset = 0,
    required int size,
  }) {
    final mapAsyncFn = _wgpuBufferMapAsync;
    if (mapAsyncFn == null) return;
    _WgpuStaticState.mapDone = false;

    using((arena) {
      final cbPointer =
          ffi.Pointer.fromFunction<
            ffi.Void Function(
              ffi.Uint32,
              WGPUStringView,
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
            )
          >(_onGlobalBufferMapped);

      final cbInfo = arena<WGPUBufferMapCallbackInfo>();
      cbInfo.ref.mode = WGPUCallbackMode.allowSpontaneous;
      cbInfo.ref.callback = cbPointer;

      mapAsyncFn(buffer, mode, offset, size, cbInfo.ref);

      for (var i = 0; i < 1000; i++) {
        if (device != null && device != ffi.nullptr) {
          devicePoll(device, wait: true);
        }
        _wgpuInstanceProcessEvents(instance);
        if (_WgpuStaticState.mapDone) break;
      }
    });
  }

  Future<int> bufferMapAsync(
    ffi.Pointer<ffi.Void> instance,
    ffi.Pointer<ffi.Void> buffer, {
    ffi.Pointer<ffi.Void>? device,
    int mode = WGPUMapMode.read,
    int offset = 0,
    required int size,
  }) async {
    final mapAsyncFn = _wgpuBufferMapAsync;
    if (mapAsyncFn == null) return 0;
    _WgpuStaticState.mapDone = false;

    return using((arena) {
      final cbPointer =
          ffi.Pointer.fromFunction<
            ffi.Void Function(
              ffi.Uint32,
              WGPUStringView,
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Void>,
            )
          >(_onGlobalBufferMapped);

      final cbInfo = arena<WGPUBufferMapCallbackInfo>();
      cbInfo.ref.mode = WGPUCallbackMode.allowSpontaneous;
      cbInfo.ref.callback = cbPointer;

      mapAsyncFn(buffer, mode, offset, size, cbInfo.ref);

      for (var i = 0; i < 1000; i++) {
        if (device != null && device != ffi.nullptr) {
          devicePoll(device, wait: true);
        }
        _wgpuInstanceProcessEvents(instance);
        if (_WgpuStaticState.mapDone) break;
      }
      return 1;
    });
  }

  ffi.Pointer<ffi.Void> bufferGetMappedRange(
    ffi.Pointer<ffi.Void> buffer, {
    int offset = 0,
    required int size,
  }) {
    final getMappedFn = _wgpuBufferGetMappedRange;
    if (getMappedFn == null) return ffi.nullptr;
    return getMappedFn(buffer, offset, size);
  }

  void bufferUnmap(ffi.Pointer<ffi.Void> buffer) {
    _wgpuBufferUnmap?.call(buffer);
  }

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

      final desc = arena<WGPUShaderModuleDescriptor>();
      desc.ref.nextInChain = wgslChain.cast<WGPUChainedStruct>();
      desc.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      desc.ref.label.length = label?.length ?? 0;

      final module = _wgpuDeviceCreateShaderModule(device, desc);
      if (module == ffi.nullptr) {
        throw GpuComputeException("Failed to compile WGSL shader module.");
      }
      return module;
    });
  }

  ffi.Pointer<ffi.Void> createComputePipeline(
    ffi.Pointer<ffi.Void> device, {
    required ffi.Pointer<ffi.Void> shaderModule,
    String entryPoint = "main",
    ffi.Pointer<ffi.Void>? layout,
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUComputePipelineDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      desc.ref.label.length = label?.length ?? 0;
      desc.ref.layout = layout ?? ffi.nullptr;
      desc.ref.computeNextInChain = ffi.nullptr;
      desc.ref.computeModule = shaderModule;
      desc.ref.computeEntryPoint.data = entryPoint.toNativeUtf8(
        allocator: arena,
      );
      desc.ref.computeEntryPoint.length = entryPoint.length;
      desc.ref.computeConstantCount = 0;
      desc.ref.computeConstants = ffi.nullptr;

      final pipeline = _wgpuDeviceCreateComputePipeline(device, desc);
      if (pipeline == ffi.nullptr) {
        throw GpuComputeException("Failed to create compute pipeline.");
      }
      return pipeline;
    });
  }

  ffi.Pointer<ffi.Void> pipelineGetBindGroupLayout(
    ffi.Pointer<ffi.Void> pipeline,
    int groupIndex,
  ) {
    final getLayoutFn = _wgpuComputePipelineGetBindGroupLayout;
    if (getLayoutFn == null) return ffi.nullptr;
    return getLayoutFn(pipeline, groupIndex);
  }

  ffi.Pointer<ffi.Void> createBindGroup(
    ffi.Pointer<ffi.Void> device, {
    required ffi.Pointer<ffi.Void> layout,
    required List<WgpuBindGroupEntryData> entries,
    String? label,
  }) {
    return using((arena) {
      final entriesPtr = arena<WGPUBindGroupEntry>(entries.length);
      for (var i = 0; i < entries.length; i++) {
        final e = entries[i];
        final entryPtr = entriesPtr + i;
        entryPtr.ref.nextInChain = ffi.nullptr;
        entryPtr.ref.binding = e.binding;
        entryPtr.ref.buffer = e.buffer;
        entryPtr.ref.offset = e.offset;
        entryPtr.ref.size = e.size;
        entryPtr.ref.sampler = ffi.nullptr;
        entryPtr.ref.textureView = ffi.nullptr;
      }

      final desc = arena<WGPUBindGroupDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      desc.ref.label.length = label?.length ?? 0;
      desc.ref.layout = layout;
      desc.ref.entryCount = entries.length;
      desc.ref.entries = entriesPtr;

      final bg = _wgpuDeviceCreateBindGroup(device, desc);
      if (bg == ffi.nullptr) {
        throw GpuComputeException("Failed to create bind group.");
      }
      return bg;
    });
  }

  ffi.Pointer<ffi.Void> createCommandEncoder(
    ffi.Pointer<ffi.Void> device, {
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUCommandEncoderDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      desc.ref.label.length = label?.length ?? 0;

      final encoder = _wgpuDeviceCreateCommandEncoder(device, desc.cast());
      if (encoder == ffi.nullptr) {
        throw GpuComputeException("Failed to create command encoder.");
      }
      return encoder;
    });
  }

  ffi.Pointer<ffi.Void> commandEncoderBeginComputePass(
    ffi.Pointer<ffi.Void> encoder, {
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUComputePassDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      desc.ref.label.length = label?.length ?? 0;
      desc.ref.timestampWritesCount = 0;
      desc.ref.timestampWrites = ffi.nullptr;

      final pass = _wgpuCommandEncoderBeginComputePass(encoder, desc.cast());
      if (pass == ffi.nullptr) {
        throw GpuComputeException("Failed to begin compute pass.");
      }
      return pass;
    });
  }

  void computePassSetPipeline(
    ffi.Pointer<ffi.Void> pass,
    ffi.Pointer<ffi.Void> pipeline,
  ) {
    _wgpuComputePassEncoderSetPipeline(pass, pipeline);
  }

  void computePassSetBindGroup(
    ffi.Pointer<ffi.Void> pass,
    int groupIndex,
    ffi.Pointer<ffi.Void> bindGroup,
  ) {
    _wgpuComputePassEncoderSetBindGroup(
      pass,
      groupIndex,
      bindGroup,
      0,
      ffi.nullptr,
    );
  }

  void computePassDispatchWorkgroups(
    ffi.Pointer<ffi.Void> pass,
    int workgroupsX,
    int workgroupsY,
    int workgroupsZ,
  ) {
    _wgpuComputePassEncoderDispatchWorkgroups(
      pass,
      workgroupsX,
      workgroupsY,
      workgroupsZ,
    );
  }

  void computePassEnd(ffi.Pointer<ffi.Void> pass) {
    _wgpuComputePassEncoderEnd(pass);
  }

  void commandEncoderCopyBufferToBuffer(
    ffi.Pointer<ffi.Void> encoder,
    ffi.Pointer<ffi.Void> source,
    int sourceOffset,
    ffi.Pointer<ffi.Void> destination,
    int destinationOffset,
    int size,
  ) {
    _wgpuCommandEncoderCopyBufferToBuffer?.call(
      encoder,
      source,
      sourceOffset,
      destination,
      destinationOffset,
      size,
    );
  }

  ffi.Pointer<ffi.Void> commandEncoderFinish(
    ffi.Pointer<ffi.Void> encoder, {
    String? label,
  }) {
    return using((arena) {
      final desc = arena<WGPUCommandBufferDescriptor>();
      desc.ref.nextInChain = ffi.nullptr;
      desc.ref.label.data = label != null
          ? label.toNativeUtf8(allocator: arena)
          : ffi.nullptr;
      desc.ref.label.length = label?.length ?? 0;

      final cmdBuf = _wgpuCommandEncoderFinish(encoder, desc.cast());
      if (cmdBuf == ffi.nullptr) {
        throw GpuComputeException("Failed to finish command encoder.");
      }
      return cmdBuf;
    });
  }

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
      _wgpuQueueSubmit(queue, commandBuffers.length, array);
    });
  }

  bool devicePoll(ffi.Pointer<ffi.Void> device, {bool wait = false}) {
    final pollFn = _wgpuDevicePoll;
    if (pollFn == null) return true;
    return pollFn(device, wait ? 1 : 0, ffi.nullptr) != 0;
  }

  void bufferDestroy(ffi.Pointer<ffi.Void> buffer) {
    if (buffer != ffi.nullptr) {
      _wgpuBufferDestroy(buffer);
    }
  }

  void deviceDestroy(ffi.Pointer<ffi.Void> device) {
    if (device != ffi.nullptr) {
      _wgpuDeviceDestroy?.call(device);
    }
  }

  void instanceRelease(ffi.Pointer<ffi.Void> instance) {
    if (instance != ffi.nullptr) {
      _wgpuInstanceRelease?.call(instance);
    }
  }

  void adapterRelease(ffi.Pointer<ffi.Void> adapter) {
    if (adapter != ffi.nullptr) {
      _wgpuAdapterRelease?.call(adapter);
    }
  }

  void deviceRelease(ffi.Pointer<ffi.Void> device) {
    if (device != ffi.nullptr) {
      _wgpuDeviceRelease?.call(device);
    }
  }

  void queueRelease(ffi.Pointer<ffi.Void> queue) {
    if (queue != ffi.nullptr) {
      _wgpuQueueRelease?.call(queue);
    }
  }

  void bufferRelease(ffi.Pointer<ffi.Void> buffer) {
    if (buffer != ffi.nullptr) {
      _wgpuBufferRelease?.call(buffer);
    }
  }

  void shaderModuleRelease(ffi.Pointer<ffi.Void> shaderModule) {
    if (shaderModule != ffi.nullptr) {
      _wgpuShaderModuleRelease?.call(shaderModule);
    }
  }

  void computePipelineRelease(ffi.Pointer<ffi.Void> computePipeline) {
    if (computePipeline != ffi.nullptr) {
      _wgpuComputePipelineRelease?.call(computePipeline);
    }
  }

  void bindGroupRelease(ffi.Pointer<ffi.Void> bindGroup) {
    if (bindGroup != ffi.nullptr) {
      _wgpuBindGroupRelease?.call(bindGroup);
    }
  }

  void bindGroupLayoutRelease(ffi.Pointer<ffi.Void> bindGroupLayout) {
    if (bindGroupLayout != ffi.nullptr) {
      _wgpuBindGroupLayoutRelease?.call(bindGroupLayout);
    }
  }

  void commandEncoderRelease(ffi.Pointer<ffi.Void> commandEncoder) {
    if (commandEncoder != ffi.nullptr) {
      _wgpuCommandEncoderRelease?.call(commandEncoder);
    }
  }

  void computePassEncoderRelease(ffi.Pointer<ffi.Void> computePassEncoder) {
    if (computePassEncoder != ffi.nullptr) {
      _wgpuComputePassEncoderRelease?.call(computePassEncoder);
    }
  }

  void commandBufferRelease(ffi.Pointer<ffi.Void> commandBuffer) {
    if (commandBuffer != ffi.nullptr) {
      _wgpuCommandBufferRelease?.call(commandBuffer);
    }
  }

  // ===========================================================================
  // Shared Dynamic Library Resolver
  // ===========================================================================

  static WgpuNativeLib? tryLoad({String? customPath}) {
    if (customPath != null && customPath.isNotEmpty) {
      try {
        final dylib = ffi.DynamicLibrary.open(customPath);
        return WgpuNativeLib(dylib, libraryPath: customPath);
      } catch (_) {
        return null;
      }
    }

    final candidatePaths = <String>[
      // 1. Try Native Assets bundled asset identifiers
      "package:gpuarray/wgpu_native",
      "wgpu_native",
      "libwgpu_native",
    ];

    final envPath = Platform.environment["WGPU_LIB_PATH"];
    if (envPath != null && envPath.isNotEmpty) {
      candidatePaths.add(envPath);
    }

    if (Platform.isLinux) {
      candidatePaths.addAll([
        "/tmp/wgpu_test/lib/libwgpu_native.so",
        "libwgpu_native.so",
        "libwgpu.so",
        "/usr/lib/libwgpu_native.so",
        "/usr/local/lib/libwgpu_native.so",
        "/usr/lib/x86_64-linux-gnu/libwgpu_native.so",
        "/usr/lib64/libwgpu_native.so",
      ]);
    } else if (Platform.isMacOS) {
      candidatePaths.addAll([
        "libwgpu_native.dylib",
        "libwgpu.dylib",
        "/usr/local/lib/libwgpu_native.dylib",
        "/opt/homebrew/lib/libwgpu_native.dylib",
      ]);
    } else if (Platform.isWindows) {
      candidatePaths.addAll(["wgpu_native.dll", "wgpu.dll"]);
    }

    for (final path in candidatePaths) {
      try {
        final dylib = ffi.DynamicLibrary.open(path);
        final lib = WgpuNativeLib(dylib, libraryPath: path);
        if (lib.isAvailable) return lib;
      } catch (_) {
        // Continue searching candidates
      }
    }

    // Try process lookup (embedded / statically linked on iOS or macOS)
    try {
      final processDylib = ffi.DynamicLibrary.process();
      final lib = WgpuNativeLib(processDylib, libraryPath: 'process');
      if (lib.isAvailable) return lib;
    } catch (_) {}

    return null;
  }

  static WgpuNativeLib load({String? customPath}) {
    final lib = tryLoad(customPath: customPath);
    if (lib == null) {
      throw GpuDeviceException(
        "Could not load native WebGPU dynamic library (libwgpu_native). "
        "Ensure libwgpu_native is installed or set the WGPU_LIB_PATH environment variable.",
      );
    }
    return lib;
  }
}
