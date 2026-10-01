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

import 'dart:typed_data';

import '../backend/wgsl/wgsl_types.dart';
import '../buffer.dart';
import '../device.dart';
import '../dtype.dart';
import '../gpu_array.dart';

/// Packs a 32-bit IEEE-754 float into its raw unsigned 32-bit integer bits.
int float32ToBits(double value) {
  final data = ByteData(4)..setFloat32(0, value, Endian.little);
  return data.getUint32(0, Endian.little);
}

/// Checks whether two shape lists [first] and [second] have identical dimensions.
bool areShapesIdentical(List<int> first, List<int> second) {
  if (identical(first, second)) return true;
  if (first.length != second.length) return false;
  for (var i = 0; i < first.length; i++) {
    if (first[i] != second[i]) return false;
  }
  return true;
}

/// WGSL helper functions for unpacking and packing IEEE-754 `f64` (`vec2<u32>`) values.
const String wgslF64ConversionHelpers = '''
fn f64_to_f32(word: vec2<u32>) -> f32 {
  let lo = word.x; let hi = word.y;
  let sign = hi & 0x80000000u; let exp64 = (hi >> 20u) & 0x7FFu; let mant_hi = hi & 0xFFFFFu;
  if (exp64 == 0u) { return bitcast<f32>(sign); }
  if (exp64 == 0x7FFu) {
    let is_nan = (mant_hi != 0u) || (lo != 0u);
    return bitcast<f32>(sign | 0x7F800000u | select(0u, 0x400000u, is_nan));
  }
  let exp32 = i32(exp64) - 896;
  if (exp32 >= 255) { return bitcast<f32>(sign | 0x7F800000u); }
  if (exp32 <= 0) {
    if (exp32 < -23) { return bitcast<f32>(sign); }
    let full_mant = 0x800000u | (mant_hi << 3u) | (lo >> 29u);
    return bitcast<f32>(sign | (full_mant >> u32(1 - exp32)));
  }
  let mant23 = (mant_hi << 3u) | (lo >> 29u);
  let round_bit = (lo >> 28u) & 1u; let sticky = lo & 0x0FFFFFFFu;
  var bits32 = sign | (u32(exp32) << 23u) | mant23;
  if (round_bit == 1u && (sticky != 0u || (mant23 & 1u) == 1u)) { bits32 = bits32 + 1u; }
  return bitcast<f32>(bits32);
}
fn f32_to_f64(value: f32) -> vec2<u32> {
  let bits = bitcast<u32>(value);
  let sign = bits & 0x80000000u; let exp32 = (bits >> 23u) & 0xFFu; let mant23 = bits & 0x7FFFFFu;
  if (exp32 == 0u) {
    if (mant23 == 0u) { return vec2<u32>(0u, sign); }
    let shift = 23u - firstLeadingBit(mant23);
    let norm_mant = (mant23 << shift) & 0x7FFFFFu;
    let exp64 = 1023u - 126u - shift;
    return vec2<u32>((norm_mant & 7u) << 29u, sign | (exp64 << 20u) | (norm_mant >> 3u));
  }
  if (exp32 == 0xFFu) {
    return vec2<u32>((mant23 & 7u) << 29u, sign | 0x7FF00000u | (mant23 >> 3u));
  }
  let exp64 = exp32 + 896u;
  return vec2<u32>((mant23 & 7u) << 29u, sign | (exp64 << 20u) | (mant23 >> 3u));
}
''';

/// Constructs a storage buffer [WgslBinding] for [dtype] at [binding].
WgslBinding storageBinding(
  int binding,
  String name,
  DType dtype,
  WgslBufferAccess access,
) {
  final customType = switch (dtype) {
    DType.float64 || DType.int64 || DType.uint64 => 'array<vec2<u32>>',
    DType.complex64 => 'array<vec2<f32>>',
    DType.complex128 => 'array<vec4<u32>>',
    DType.float32 => 'array<f32>',
    DType.int32 => 'array<i32>',
    DType.uint32 => 'array<u32>',
    DType.float16 ||
    DType.bfloat16 ||
    DType.int16 ||
    DType.uint16 ||
    DType.int8 ||
    DType.uint8 ||
    DType.boolean =>
      access == WgslBufferAccess.readWrite
          ? 'array<atomic<u32>>'
          : 'array<u32>',
  };
  return WgslBinding(
    group: 0,
    binding: binding,
    name: name,
    access: access,
    customTypeName: customType,
  );
}

/// Dispatches a 1D WGSL compute shader with [totalElements] threads on [device].
void dispatch1DKernel({
  required GpuDevice device,
  required String name,
  required String code,
  required List<WgslBinding> bindings,
  required List<GpuBuffer> buffers,
  required List<int> uniforms,
  required int totalElements,
}) {
  final module = WgslShaderModule(
    name: name,
    code: code,
    workgroupSize: WgslWorkgroupSize.linear1D,
    bindings: bindings,
  );
  final dispatch = module.calculateDispatch1D(totalElements);
  device.backend.dispatchComputePipeline(
    shaderModule: module,
    buffers: buffers,
    uniforms: uniforms,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
    workgroupsZ: dispatch.workgroupsZ,
  );
}

/// Generates a WGSL helper function [funcName] that reads an element from [bufferName] as `f32`.
String wgslLoadFloat(
  DType dtype,
  String bufferName,
  String funcName,
) => switch (dtype) {
  DType.float64 =>
    'fn $funcName(index: u32) -> f32 { return f64_to_f32($bufferName[index]); }',
  DType.float32 =>
    'fn $funcName(index: u32) -> f32 { return $bufferName[index]; }',
  DType.float16 =>
    'fn $funcName(index: u32) -> f32 { return unpack2x16float($bufferName[index >> 1u])[index & 1u]; }',
  DType.bfloat16 =>
    'fn $funcName(index: u32) -> f32 { let half = ($bufferName[index >> 1u] >> ((index & 1u) * 16u)) & 0xFFFFu; return bitcast<f32>(half << 16u); }',
  DType.int64 || DType.uint64 =>
    'fn $funcName(index: u32) -> f32 { return f32(bitcast<i32>($bufferName[index].x)); }',
  DType.int32 || DType.uint32 =>
    'fn $funcName(index: u32) -> f32 { return f32($bufferName[index]); }',
  DType.int16 =>
    'fn $funcName(index: u32) -> f32 { return f32(extractBits(i32($bufferName[index >> 1u]), (index & 1u) * 16u, 16u)); }',
  DType.uint16 =>
    'fn $funcName(index: u32) -> f32 { return f32(extractBits($bufferName[index >> 1u], (index & 1u) * 16u, 16u)); }',
  DType.int8 =>
    'fn $funcName(index: u32) -> f32 { return f32(extractBits(i32($bufferName[index >> 2u]), (index & 3u) * 8u, 8u)); }',
  DType.uint8 || DType.boolean =>
    'fn $funcName(index: u32) -> f32 { return f32(extractBits($bufferName[index >> 2u], (index & 3u) * 8u, 8u)); }',
  DType.complex64 =>
    'fn $funcName(index: u32) -> f32 { return $bufferName[index].x; }',
  DType.complex128 =>
    'fn $funcName(index: u32) -> f32 { return f64_to_f32($bufferName[index].xy); }',
};

/// Generates a WGSL helper function [funcName] that stores an `f32` value into [bufferName].
String wgslStoreFloat(
  DType dtype,
  String bufferName,
  String funcName,
) => switch (dtype) {
  DType.float64 =>
    'fn $funcName(index: u32, value: f32) { $bufferName[index] = f32_to_f64(value); }',
  DType.float32 =>
    'fn $funcName(index: u32, value: f32) { $bufferName[index] = value; }',
  DType.int32 =>
    'fn $funcName(index: u32, value: f32) { $bufferName[index] = i32(value); }',
  DType.uint32 =>
    'fn $funcName(index: u32, value: f32) { $bufferName[index] = u32(value); }',
  DType.int64 || DType.uint64 =>
    'fn $funcName(index: u32, value: f32) { let iv = i32(value); $bufferName[index] = vec2<u32>(bitcast<u32>(iv), select(0u, 0xFFFFFFFFu, iv < 0)); }',
  DType.complex64 =>
    'fn $funcName(index: u32, value: f32) { $bufferName[index] = vec2<f32>(value, 0.0); }',
  DType.complex128 =>
    'fn $funcName(index: u32, value: f32) { $bufferName[index] = vec4<u32>(f32_to_f64(value), vec2<u32>(0u, 0u)); }',
  DType.float16 ||
  DType.bfloat16 ||
  DType.int16 ||
  DType.uint16 ||
  DType.int8 ||
  DType.uint8 ||
  DType.boolean =>
    '''
fn $funcName(index: u32, value: f32) {
  let word_index = index >> 1u; let shift = (index & 1u) * 16u; let mask = ~(0xFFFFu << shift);
  let bits = (pack2x16float(vec2<f32>(value, 0.0)) & 0xFFFFu) << shift;
  var old_word = atomicLoad(&$bufferName[word_index]);
  loop {
    let res = atomicCompareExchangeWeak(&$bufferName[word_index], old_word, (old_word & mask) | bits);
    if (res.exchanged) { break; }
    old_word = res.old_value;
  }
}''',
};

/// Generates a WGSL helper function [funcName] that reads an index element from [bufferName] as `i32`.
String wgslLoadIndex(
  DType dtype,
  String bufferName,
  String funcName,
) => switch (dtype) {
  DType.int32 =>
    'fn $funcName(index: u32) -> i32 { return $bufferName[index]; }',
  DType.uint32 || DType.float32 =>
    'fn $funcName(index: u32) -> i32 { return i32($bufferName[index]); }',
  DType.int64 || DType.uint64 =>
    'fn $funcName(index: u32) -> i32 { return bitcast<i32>($bufferName[index].x); }',
  DType.float64 =>
    'fn $funcName(index: u32) -> i32 { return i32(f64_to_f32($bufferName[index])); }',
  DType.int16 =>
    'fn $funcName(index: u32) -> i32 { return extractBits(i32($bufferName[index >> 1u]), (index & 1u) * 16u, 16u); }',
  DType.uint16 =>
    'fn $funcName(index: u32) -> i32 { return i32(extractBits($bufferName[index >> 1u], (index & 1u) * 16u, 16u)); }',
  DType.int8 =>
    'fn $funcName(index: u32) -> i32 { return extractBits(i32($bufferName[index >> 2u]), (index & 3u) * 8u, 8u); }',
  DType.uint8 || DType.boolean =>
    'fn $funcName(index: u32) -> i32 { return i32(extractBits($bufferName[index >> 2u], (index & 3u) * 8u, 8u)); }',
  DType.float16 =>
    'fn $funcName(index: u32) -> i32 { return i32(unpack2x16float($bufferName[index >> 1u])[index & 1u]); }',
  DType.bfloat16 =>
    'fn $funcName(index: u32) -> i32 { let half = ($bufferName[index >> 1u] >> ((index & 1u) * 16u)) & 0xFFFFu; return i32(bitcast<f32>(half << 16u)); }',
  DType.complex64 =>
    'fn $funcName(index: u32) -> i32 { return i32($bufferName[index].x); }',
  DType.complex128 =>
    'fn $funcName(index: u32) -> i32 { return i32(f64_to_f32($bufferName[index].xy)); }',
};

/// Dispatches a GPU strided copy from [gradOutput] into [sliceView] for `SliceBackward`.
void dispatchSliceBackwardCopy(
  GpuArray<DTypeTag> gradOutput,
  GpuArray<DTypeTag> sliceView,
) {
  if (sliceView.size == 0) return;
  gradOutput.copy(out: sliceView);
}

/// Dispatches the `im2col` spatial patch extraction kernel on the GPU.
void dispatchIm2Col({
  required GpuArray<DTypeTag> input,
  required GpuArray<DTypeTag> columns,
  required int kernelSize,
  required int stride,
  required int padding,
  required int outHeight,
  required int outWidth,
}) {
  final totalElements = columns.size;
  if (totalElements == 0) return;

  final inChannels = input.shape[1];
  final inHeight = input.shape[2];
  final inWidth = input.shape[3];
  final patchSize = inChannels * kernelSize * kernelSize;
  final contiguousInput = input.isContiguous ? input : input.copy();

  try {
    final bindings = [
      storageBinding(
        0,
        'input_buf',
        contiguousInput.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(1, 'col_buf', columns.dtype, WgslBufferAccess.readWrite),
      const WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'Im2ColUniforms',
      ),
    ];
    final loadInputFn = wgslLoadFloat(
      contiguousInput.dtype,
      'input_buf',
      'load_input',
    );
    final storeColFn = wgslStoreFloat(columns.dtype, 'col_buf', 'store_col');
    final copyStmt =
        (contiguousInput.dtype == DType.float64 &&
            columns.dtype == DType.float64)
        ? 'col_buf[thread_index] = input_buf[input_offset];'
        : 'store_col(thread_index, load_input(input_offset));';

    final code =
        '''
$wgslF64ConversionHelpers
struct Im2ColUniforms {
  total_elements: u32, in_channels: u32, in_height: u32, in_width: u32,
  kernel_size: u32, stride: u32, padding: i32, out_height: u32,
  out_width: u32, patch_size: u32, input_offset: u32, pad0: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadInputFn
$storeColFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let patch_index = thread_index / uniforms.patch_size;
  let col_index = thread_index % uniforms.patch_size;
  let ow = patch_index % uniforms.out_width;
  let patch_oh_b = patch_index / uniforms.out_width;
  let oh = patch_oh_b % uniforms.out_height;
  let batch = patch_oh_b / uniforms.out_height;
  let kw = col_index % uniforms.kernel_size;
  let col_kh_c = col_index / uniforms.kernel_size;
  let kh = col_kh_c % uniforms.kernel_size;
  let channel = col_kh_c / uniforms.kernel_size;
  let ih = i32(oh * uniforms.stride + kh) - uniforms.padding;
  let iw = i32(ow * uniforms.stride + kw) - uniforms.padding;
  if (ih < 0 || ih >= i32(uniforms.in_height) || iw < 0 || iw >= i32(uniforms.in_width)) {
    store_col(thread_index, 0.0);
    return;
  }
  let input_offset = uniforms.input_offset +
      (((batch * uniforms.in_channels + channel) * uniforms.in_height + u32(ih)) * uniforms.in_width + u32(iw));
  $copyStmt
}
''';

    dispatch1DKernel(
      device: columns.device,
      name: 'autograd_im2col_${columns.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [contiguousInput.buffer, columns.buffer],
      uniforms: [
        totalElements,
        inChannels,
        inHeight,
        inWidth,
        kernelSize,
        stride,
        padding & 0xFFFFFFFF,
        outHeight,
        outWidth,
        patchSize,
        contiguousInput.offsetElements,
        0,
      ],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousInput, input)) {
      contiguousInput.dispose();
    }
  }
}

/// Dispatches the `col2im` spatial gradient accumulation kernel on the GPU.
void dispatchCol2Im({
  required GpuArray<DTypeTag> gradColumns,
  required GpuArray<DTypeTag> gradInput,
  required int kernelSize,
  required int stride,
  required int padding,
  required int outHeight,
  required int outWidth,
}) {
  final totalElements = gradInput.size;
  if (totalElements == 0) return;

  final inChannels = gradInput.shape[1];
  final inHeight = gradInput.shape[2];
  final inWidth = gradInput.shape[3];
  final patchSize = inChannels * kernelSize * kernelSize;
  final contiguousColumns = gradColumns.isContiguous
      ? gradColumns
      : gradColumns.copy();

  try {
    final bindings = [
      storageBinding(
        0,
        'col_buf',
        contiguousColumns.dtype,
        WgslBufferAccess.read,
      ),
      storageBinding(
        1,
        'grad_input_buf',
        gradInput.dtype,
        WgslBufferAccess.readWrite,
      ),
      const WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'Col2ImUniforms',
      ),
    ];
    final loadColFn = wgslLoadFloat(
      contiguousColumns.dtype,
      'col_buf',
      'load_col',
    );
    final storeGradFn = wgslStoreFloat(
      gradInput.dtype,
      'grad_input_buf',
      'store_grad',
    );

    final code =
        '''
$wgslF64ConversionHelpers
struct Col2ImUniforms {
  total_elements: u32, in_channels: u32, in_height: u32, in_width: u32,
  kernel_size: u32, stride: u32, padding: i32, out_height: u32,
  out_width: u32, patch_size: u32, col_offset: u32, pad0: u32,
}
${bindings.map((b) => b.toWgslDeclaration()).join('\n')}
$loadColFn
$storeGradFn
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_index = global_id.x + global_id.y * (num_wg.x * 256u);
  if (thread_index >= uniforms.total_elements) { return; }
  let iw = i32(thread_index % uniforms.in_width);
  let rem_ih = thread_index / uniforms.in_width;
  let ih = i32(rem_ih % uniforms.in_height);
  let rem_c = rem_ih / uniforms.in_height;
  let channel = rem_c % uniforms.in_channels;
  let batch = rem_c / uniforms.in_channels;
  var acc: f32 = 0.0;
  let stride_i = i32(uniforms.stride);
  let k_size = uniforms.kernel_size;
  for (var kh: u32 = 0u; kh < k_size; kh = kh + 1u) {
    let numer_h = ih + uniforms.padding - i32(kh);
    if (numer_h >= 0 && (numer_h % stride_i) == 0) {
      let oh = u32(numer_h / stride_i);
      if (oh < uniforms.out_height) {
        for (var kw: u32 = 0u; kw < k_size; kw = kw + 1u) {
          let numer_w = iw + uniforms.padding - i32(kw);
          if (numer_w >= 0 && (numer_w % stride_i) == 0) {
            let ow = u32(numer_w / stride_i);
            if (ow < uniforms.out_width) {
              let patch_row = (batch * uniforms.out_height + oh) * uniforms.out_width + ow;
              let patch_col = (channel * k_size + kh) * k_size + kw;
              acc = acc + load_col(uniforms.col_offset + patch_row * uniforms.patch_size + patch_col);
            }
          }
        }
      }
    }
  }
  store_grad(thread_index, acc);
}
''';

    dispatch1DKernel(
      device: gradInput.device,
      name: 'autograd_col2im_${gradInput.dtype.name}',
      code: code,
      bindings: bindings,
      buffers: [contiguousColumns.buffer, gradInput.buffer],
      uniforms: [
        totalElements,
        inChannels,
        inHeight,
        inWidth,
        kernelSize,
        stride,
        padding & 0xFFFFFFFF,
        outHeight,
        outWidth,
        patchSize,
        contiguousColumns.offsetElements,
        0,
      ],
      totalElements: totalElements,
    );
  } finally {
    if (!identical(contiguousColumns, gradColumns)) {
      contiguousColumns.dispose();
    }
  }
}
