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

import '../../buffer.dart';
import '../../device.dart';
import '../../dtype.dart';
import '../../gpu_array.dart';
import '../compute_engine.dart';
import 'kernel_fusion.dart';
import 'wgsl_types.dart';

/// Validation diagnostic results for WGSL compute shader syntax and binding layouts.
final class WgslValidationResult {
  /// Whether the shader passed all structural and binding checks.
  final bool isValid;

  /// Unmodifiable list of syntax or binding error messages.
  final List<String> errors;

  /// Unmodifiable list of non-fatal warning messages.
  final List<String> warnings;

  /// Creates a [WgslValidationResult] with unmodifiable [errors] and [warnings].
  WgslValidationResult({
    required this.isValid,
    List<String> errors = const [],
    List<String> warnings = const [],
  }) : errors = List<String>.unmodifiable(errors),
       warnings = List<String>.unmodifiable(warnings);

  @override
  String toString() {
    if (isValid) {
      return 'WgslValidationResult(VALID${warnings.isNotEmpty ? ", warnings: ${warnings.length}" : ""})';
    }
    return 'WgslValidationResult(INVALID, errors: $errors)';
  }
}

/// Validates a WGSL compute shader string against standard structure and WebGPU rules.
WgslValidationResult validateWgslShader(String code) {
  final errors = <String>[];
  final warnings = <String>[];

  // Strip block and line comments to avoid false positives
  final cleanCode = code
      .replaceAll(RegExp(r'/\*[\s\S]*?\*/'), '')
      .replaceAll(RegExp(r'//.*'), '');

  // 1. Bracket and parenthesis balance checks
  var braceCount = 0;
  var parenCount = 0;
  var bracketCount = 0;

  for (var i = 0; i < cleanCode.length; i++) {
    final char = cleanCode[i];
    if (char == '{') braceCount++;
    if (char == '}') braceCount--;
    if (char == '(') parenCount++;
    if (char == ')') parenCount--;
    if (char == '[') bracketCount++;
    if (char == ']') bracketCount--;

    if (braceCount < 0) {
      errors.add('Unmatched closing brace "}" at character $i');
      break;
    }
    if (parenCount < 0) {
      errors.add('Unmatched closing parenthesis ")" at character $i');
      break;
    }
    if (bracketCount < 0) {
      errors.add('Unmatched closing bracket "]" at character $i');
      break;
    }
  }

  if (braceCount > 0) {
    errors.add('Unclosed brace "{" (missing $braceCount "}")');
  }
  if (parenCount > 0) {
    errors.add('Unclosed parenthesis "(" (missing $parenCount ")")');
  }
  if (bracketCount > 0) {
    errors.add('Unclosed bracket "[" (missing $bracketCount "]")');
  }

  // 2. Entry point and compute stage annotations
  if (!cleanCode.contains('@compute')) {
    errors.add('Missing @compute shader stage attribute');
  }
  if (!cleanCode.contains('@workgroup_size')) {
    errors.add('Missing @workgroup_size attribute on compute shader');
  }
  if (!cleanCode.contains(RegExp(r'fn\s+\w+\s*\('))) {
    errors.add('Missing entry point function declaration ("fn <name>(...)")');
  }

  // 3. Binding uniqueness and validation
  final bindingRegex = RegExp(r'@group\((\d+)\)\s*@binding\((\d+)\)');
  final seenBindings = <String>{};
  for (final match in bindingRegex.allMatches(cleanCode)) {
    final group = match.group(1)!;
    final binding = match.group(2)!;
    final key = 'g$group:b$binding';
    if (!seenBindings.add(key)) {
      errors.add(
        'Duplicate resource binding detected: @group($group) @binding($binding)',
      );
    }
  }

  // 4. Storage buffer access qualifier validation
  final storageRegex = RegExp(r'var<storage,\s*(\w+)>');
  for (final match in storageRegex.allMatches(cleanCode)) {
    final access = match.group(1)!;
    if (access != 'read' && access != 'read_write') {
      errors.add(
        'Invalid storage buffer access qualifier "$access" (must be "read" or "read_write")',
      );
    }
  }

  // 5. Workgroup memory check
  if (cleanCode.contains('var<workgroup>')) {
    if (!cleanCode.contains('workgroupBarrier()') &&
        !cleanCode.contains('sdata[')) {
      warnings.add(
        'Shader declares workgroup memory but does not appear to synchronize or read from it',
      );
    }
  }

  // 6. Reserved keyword check (Naga WGSL reserved words)
  final reservedRegex = RegExp(
    r'\b(meta|pass|target|mod|ref|filter|set|final|match|override|handle|subpass)\b',
  );
  for (final match in reservedRegex.allMatches(cleanCode)) {
    errors.add('Reserved WGSL keyword "${match.group(1)}" used as identifier');
  }

  return WgslValidationResult(
    isValid: errors.isEmpty,
    errors: errors,
    warnings: warnings,
  );
}

/// Static validator for checking WGSL compute shader syntax, structure, and bindings.
extension type const WgslSyntaxValidator._(Object? _) {
  /// Validates a WGSL compute shader string against standard structure and WebGPU rules.
  static WgslValidationResult validate(String code) => validateWgslShader(code);
}

/// Executable fused WGSL compute kernel bound to a [FusedKernelDescriptor] and [WgslShaderModule].
final class CompiledWgslKernel {
  /// Underlying fused expression descriptor defining inputs, scalars, and layout.
  final FusedKernelDescriptor descriptor;

  /// Verified WGSL compute shader module and binding metadata.
  final WgslShaderModule shaderModule;

  /// Creates a [CompiledWgslKernel] wrapping [descriptor] and [shaderModule].
  const CompiledWgslKernel({
    required this.descriptor,
    required this.shaderModule,
  });

  /// Kernel identifier.
  String get name => descriptor.name;

  /// Generated WGSL compute shader source code.
  String get code => shaderModule.code;

  /// Ordered input array variable descriptors required by this kernel.
  List<VarExpr> get inputs => descriptor.inputs;

  /// Ordered runtime scalar uniform descriptors declared in this kernel.
  List<ScalarParamExpr> get scalarParams => descriptor.scalarParams;

  /// Dispatches this compiled WGSL kernel on the GPU over the provided [inputs] or [positionalInputs].
  ///
  /// Input arrays may be supplied either by variable name via [inputs] or in
  /// binding order via [positionalInputs]. Runtime scalar uniform parameters
  /// declared via [Expr.scalar] can be overridden in [scalars]. When [out] is
  /// provided, the result is written into [out] and returned.
  GpuArray<T> run<T extends DTypeTag>({
    Map<String, GpuArray<DTypeTag>> inputs = const {},
    List<GpuArray<DTypeTag>> positionalInputs = const [],
    Map<String, double>? scalars,
    List<int>? outputShape,
    List<int>? shape,
    DType<DTypeTag>? dtype,
    GpuDevice? device,
    GpuArray<T>? out,
  }) {
    if (out != null && out.isDisposed) {
      throw StateError('Cannot write into a disposed GpuArray out tensor.');
    }
    for (final entry in inputs.entries) {
      if (entry.value.isDisposed) {
        throw StateError(
          'Cannot execute kernel with disposed input tensor "${entry.key}".',
        );
      }
    }
    for (var i = 0; i < positionalInputs.length; i++) {
      if (positionalInputs[i].isDisposed) {
        throw StateError(
          'Cannot execute kernel with disposed positional input tensor at index $i.',
        );
      }
    }

    final requestedShape = outputShape ?? shape;
    if (out != null) {
      if (out.size > 1 && out.strides.contains(0)) {
        throw ArgumentError.value(
          out,
          'out',
          'Must be writeable and not a broadcasted view.',
        );
      }
      if (dtype != null && out.dtype != dtype) {
        throw ArgumentError.value(
          out.dtype,
          'out',
          'Must match requested dtype $dtype.',
        );
      }
      if (requestedShape != null &&
          !areShapesEqual(out.shape, requestedShape)) {
        throw ArgumentError.value(
          out.shape,
          'out',
          'Must match requested shape $requestedShape.',
        );
      }
    }

    final effectiveScalars = scalars ?? const <String, double>{};
    // Validate scalar parameter overrides.
    if (effectiveScalars.isNotEmpty) {
      final declaredScalars = {
        for (final param in descriptor.scalarParams) param.name,
      };
      for (final key in effectiveScalars.keys) {
        if (!declaredScalars.contains(key)) {
          throw ArgumentError.value(
            key,
            'scalars',
            'Must be a declared scalar parameter of the kernel (${declaredScalars.toList()}).',
          );
        }
      }
    }

    // Resolve input arrays in descriptor binding order.
    final resolvedInputs = <GpuArray<DTypeTag>>[];
    if (descriptor.inputs.isNotEmpty) {
      if (inputs.isNotEmpty && positionalInputs.isNotEmpty) {
        throw ArgumentError.value(
          positionalInputs,
          'positionalInputs',
          'Must not provide both named inputs and positionalInputs.',
        );
      }
      if (inputs.isNotEmpty) {
        for (final variable in descriptor.inputs) {
          final array = inputs[variable.name];
          if (array == null) {
            throw ArgumentError.value(
              inputs,
              'inputs',
              'Must provide a GpuArray for variable "${variable.name}".',
            );
          }
          resolvedInputs.add(array);
        }
      } else if (positionalInputs.isNotEmpty) {
        if (positionalInputs.length != descriptor.inputs.length) {
          throw ArgumentError.value(
            positionalInputs.length,
            'positionalInputs',
            'Must provide exactly ${descriptor.inputs.length} input arrays for ${descriptor.inputs.map((v) => v.name).toList()}.',
          );
        }
        resolvedInputs.addAll(positionalInputs);
      } else {
        throw ArgumentError.value(
          inputs,
          'inputs',
          'Must provide input arrays for kernel variables ${descriptor.inputs.map((v) => v.name).toList()}.',
        );
      }
    }

    final targetShape =
        out?.shape ??
        requestedShape ??
        (resolvedInputs.isNotEmpty ? resolvedInputs.first.shape : null);
    if (targetShape == null) {
      throw ArgumentError.value(
        requestedShape,
        'outputShape',
        'Must provide outputShape, shape, or out when the kernel has no input arrays.',
      );
    }
    if (out != null && !areShapesEqual(out.shape, targetShape)) {
      throw ArgumentError.value(
        out.shape,
        'out',
        'Must match target shape $targetShape.',
      );
    }

    final targetDevice =
        out?.device ??
        device ??
        (resolvedInputs.isNotEmpty
            ? resolvedInputs.first.device
            : GpuDevice.defaultDevice);
    final resolvedDType =
        (out?.dtype ??
                dtype ??
                _inferOutputDType<T>(
                  resolvedInputs.isNotEmpty ? resolvedInputs.first.dtype : null,
                ))
            as DType<T>;
    final totalElements = computeSize(targetShape);

    for (var i = 0; i < resolvedInputs.length; i++) {
      final input = resolvedInputs[i];
      if (descriptor.isStrided) {
        // Validate broadcast compatibility with targetShape.
        broadcastStrides(input.shape, input.strides, targetShape);
      } else if (input.size != totalElements) {
        throw ArgumentError.value(
          input.shape,
          'inputs',
          'Must have $totalElements elements matching target shape $targetShape.',
        );
      }
    }

    if (totalElements == 0) {
      if (out != null) return out;
      return GpuArray.empty(targetShape, resolvedDType, device: targetDevice);
    }

    final canWriteDirectToOut =
        out != null &&
        out.dtype == DType.float32 &&
        out.isContiguous &&
        out.offsetElements == 0 &&
        out.device == targetDevice;

    final f32Target = canWriteDirectToOut
        ? (out as GpuArray<Float32>)
        : GpuArray.empty(targetShape, DType.float32, device: targetDevice);

    final temporaryArrays = <GpuArray<DTypeTag>>[];
    try {
      final storageBuffers = <GpuBuffer>[];
      final stagedInputs = <GpuArray<Float32>>[];

      for (var i = 0; i < resolvedInputs.length; i++) {
        final input = resolvedInputs[i];
        final aliasesTarget = identical(input.buffer, f32Target.buffer);
        if (descriptor.isStrided) {
          if (input.dtype == DType.float32 &&
              input.device == targetDevice &&
              !aliasesTarget) {
            final typedInput = input as GpuArray<Float32>;
            stagedInputs.add(typedInput);
            storageBuffers.add(typedInput.buffer);
          } else {
            final converted = input.device == targetDevice
                ? (input.dtype == DType.float32
                      ? (input as GpuArray<Float32>).copy()
                      : input.astype(DType.float32))
                : input.toDevice(targetDevice).astype(DType.float32);
            temporaryArrays.add(converted);
            stagedInputs.add(converted);
            storageBuffers.add(converted.buffer);
          }
        } else {
          if (input.dtype == DType.float32 &&
              input.isContiguous &&
              input.offsetElements == 0 &&
              input.device == targetDevice &&
              !aliasesTarget) {
            final typedInput = input as GpuArray<Float32>;
            stagedInputs.add(typedInput);
            storageBuffers.add(typedInput.buffer);
          } else {
            final onDevice = input.device == targetDevice
                ? input
                : input.toDevice(targetDevice);
            if (!identical(onDevice, input)) {
              temporaryArrays.add(onDevice);
            }
            final contiguousF32 = onDevice.dtype == DType.float32
                ? (onDevice as GpuArray<Float32>).copy()
                : onDevice.astype(DType.float32);
            temporaryArrays.add(contiguousF32);
            stagedInputs.add(contiguousF32);
            storageBuffers.add(contiguousF32.buffer);
          }
        }
      }

      storageBuffers.add(f32Target.buffer);

      final List<int> uniformWords;
      if (!descriptor.isStrided) {
        final fieldCount = 1 + descriptor.scalarParams.length;
        final paddedWords = ((fieldCount + 3) ~/ 4) * 4;
        final byteData = ByteData(paddedWords * 4);
        byteData.setUint32(0, totalElements, Endian.little);
        for (var i = 0; i < descriptor.scalarParams.length; i++) {
          final param = descriptor.scalarParams[i];
          final scalarValue =
              effectiveScalars[param.name] ?? param.defaultValue;
          byteData.setFloat32((1 + i) * 4, scalarValue, Endian.little);
        }
        uniformWords = List<int>.generate(
          paddedWords,
          (index) => byteData.getUint32(index * 4, Endian.little),
        );
      } else {
        final byteData = ByteData(160);
        final rank = targetShape.length;
        byteData.setUint32(0, totalElements, Endian.little);
        byteData.setUint32(4, rank, Endian.little);
        byteData.setUint32(8, 0, Endian.little);
        byteData.setUint32(12, 0, Endian.little);
        for (var d = 0; d < 8; d++) {
          byteData.setUint32(
            16 + d * 4,
            d < rank ? targetShape[d] : 0,
            Endian.little,
          );
        }
        final stridesA = stagedInputs.isNotEmpty
            ? broadcastStrides(
                stagedInputs[0].shape,
                stagedInputs[0].strides,
                targetShape,
              )
            : List<int>.filled(rank, 0);
        final stridesB = stagedInputs.length > 1
            ? broadcastStrides(
                stagedInputs[1].shape,
                stagedInputs[1].strides,
                targetShape,
              )
            : stridesA;
        final stridesOut = f32Target.strides;
        for (var d = 0; d < 8; d++) {
          byteData.setInt32(
            48 + d * 4,
            d < rank ? stridesA[d] : 0,
            Endian.little,
          );
          byteData.setInt32(
            80 + d * 4,
            d < rank ? stridesB[d] : 0,
            Endian.little,
          );
          byteData.setInt32(
            112 + d * 4,
            d < rank ? stridesOut[d] : 0,
            Endian.little,
          );
        }
        byteData.setUint32(
          144,
          stagedInputs.isNotEmpty ? stagedInputs[0].offsetElements : 0,
          Endian.little,
        );
        byteData.setUint32(
          148,
          stagedInputs.length > 1
              ? stagedInputs[1].offsetElements
              : (stagedInputs.isNotEmpty ? stagedInputs[0].offsetElements : 0),
          Endian.little,
        );
        byteData.setUint32(152, f32Target.offsetElements, Endian.little);
        final firstScalar = descriptor.scalarParams.isNotEmpty
            ? (effectiveScalars[descriptor.scalarParams.first.name] ??
                  descriptor.scalarParams.first.defaultValue)
            : 0.0;
        byteData.setFloat32(156, firstScalar, Endian.little);
        uniformWords = List<int>.generate(
          40,
          (index) => byteData.getUint32(index * 4, Endian.little),
        );
      }

      final dispatch = shaderModule.calculateDispatch1D(totalElements);
      targetDevice.backend.dispatchComputePipeline(
        shaderModule: shaderModule,
        buffers: storageBuffers,
        uniforms: uniformWords,
        workgroupsX: dispatch.workgroupsX,
        workgroupsY: dispatch.workgroupsY,
        workgroupsZ: dispatch.workgroupsZ,
      );

      if (out != null) {
        if (!canWriteDirectToOut) {
          _copyResultIntoOut<T>(f32Target, out);
          f32Target.dispose();
        }
        return out;
      }

      if (resolvedDType == DType.float32) {
        return f32Target as GpuArray<T>;
      }
      final converted = f32Target.astype(resolvedDType);
      f32Target.dispose();
      return converted;
    } on Object {
      if (!canWriteDirectToOut) {
        f32Target.dispose();
      }
      rethrow;
    } finally {
      for (final temp in temporaryArrays) {
        temp.dispose();
      }
    }
  }

  /// Dispatches this compiled WGSL kernel on the GPU over the named [inputs] map.
  GpuArray<T> execute<T extends DTypeTag>(
    Map<String, GpuArray<DTypeTag>> inputs, {
    Map<String, double>? scalars,
    List<int>? outputShape,
    List<int>? shape,
    DType<DTypeTag>? dtype,
    GpuDevice? device,
    GpuArray<T>? out,
  }) => run<T>(
    inputs: inputs,
    scalars: scalars,
    outputShape: outputShape,
    shape: shape,
    dtype: dtype,
    device: device,
    out: out,
  );

  /// Dispatches this compiled WGSL kernel on the GPU over [inputs] in binding order.
  GpuArray<T> executePositional<T extends DTypeTag>(
    List<GpuArray<DTypeTag>> inputs, {
    Map<String, double>? scalars,
    List<int>? outputShape,
    List<int>? shape,
    DType<DTypeTag>? dtype,
    GpuDevice? device,
    GpuArray<T>? out,
  }) => run<T>(
    positionalInputs: inputs,
    scalars: scalars,
    outputShape: outputShape,
    shape: shape,
    dtype: dtype,
    device: device,
    out: out,
  );

  /// Callable shorthand for dispatching this compiled WGSL kernel on [inputs].
  GpuArray<T> call<T extends DTypeTag>(
    Map<String, GpuArray<DTypeTag>> inputs, {
    Map<String, double>? scalars,
    List<int>? outputShape,
    List<int>? shape,
    DType<DTypeTag>? dtype,
    GpuDevice? device,
    GpuArray<T>? out,
  }) => run<T>(
    inputs: inputs,
    scalars: scalars,
    outputShape: outputShape,
    shape: shape,
    dtype: dtype,
    device: device,
    out: out,
  );
}

DType<T> _inferOutputDType<T extends DTypeTag>(DType<DTypeTag>? firstInput) {
  if (T == Float32) return DType.float32 as DType<T>;
  if (T == Float64) return DType.float64 as DType<T>;
  if (T == Float16) return DType.float16 as DType<T>;
  if (T == BFloat16) return DType.bfloat16 as DType<T>;
  if (T == Int64) return DType.int64 as DType<T>;
  if (T == Int32) return DType.int32 as DType<T>;
  if (T == Int16) return DType.int16 as DType<T>;
  if (T == Int8) return DType.int8 as DType<T>;
  if (T == Uint64) return DType.uint64 as DType<T>;
  if (T == Uint32) return DType.uint32 as DType<T>;
  if (T == Uint16) return DType.uint16 as DType<T>;
  if (T == Uint8) return DType.uint8 as DType<T>;
  if (T == Boolean) return DType.boolean as DType<T>;
  if (firstInput != null) return firstInput as DType<T>;
  return DType.float32 as DType<T>;
}

void _copyResultIntoOut<T extends DTypeTag>(
  GpuArray<Float32> source,
  GpuArray<T> destination,
) {
  final typedSource = destination.dtype == DType.float32
      ? source
      : source.astype(destination.dtype);
  try {
    if (destination.isContiguous && destination.device == typedSource.device) {
      final byteCount = destination.size * destination.dtype.byteWidth;
      final dstByteOffset =
          destination.offsetElements * destination.dtype.byteWidth;
      final bytes = typedSource.buffer.readBytes(offset: 0, bytes: byteCount);
      destination.buffer.writeBytes(bytes, offset: dstByteOffset);
      return;
    }
    final shape = destination.shape;
    final strides = destination.strides;
    final rank = shape.length;
    final total = destination.size;
    for (var flatIndex = 0; flatIndex < total; flatIndex++) {
      var remaining = flatIndex;
      var dstOffset = destination.offsetElements;
      for (var d = rank - 1; d >= 0; d--) {
        final dimSize = shape[d];
        if (dimSize > 0) {
          final coord = remaining % dimSize;
          remaining ~/= dimSize;
          dstOffset += coord * strides[d];
        }
      }
      final val = readBufferAny(
        typedSource.buffer,
        typedSource.dtype,
        flatIndex,
        offsetElements: typedSource.offsetElements,
      );
      writeBufferAny(destination.buffer, destination.dtype, dstOffset, val);
    }
  } finally {
    if (!identical(typedSource, source)) {
      typedSource.dispose();
    }
  }
}

/// Dynamic JIT compiler for generating, validating, and caching fused WGSL compute shaders.
final class WgslJitCompiler {
  final Map<String, WgslShaderModule> _cache;

  /// Maximum number of compiled shader modules retained in the LRU cache.
  final int maxCacheSize;

  int _cacheHits = 0;
  int _cacheMisses = 0;

  /// Creates a [WgslJitCompiler] with an optional initial [cache] and [maxCacheSize].
  WgslJitCompiler({
    Map<String, WgslShaderModule>? cache,
    this.maxCacheSize = 512,
  }) : _cache = cache != null
           ? Map<String, WgslShaderModule>.of(cache)
           : <String, WgslShaderModule>{};

  /// Process-wide shared instance of the JIT compiler.
  static final WgslJitCompiler instance = WgslJitCompiler();

  /// Total number of cache hits.
  int get cacheHits => _cacheHits;

  /// Total number of cache misses.
  int get cacheMisses => _cacheMisses;

  /// Total number of currently cached shader modules.
  int get cachedCount => _cache.length;

  /// Clears the compilation cache and resets hit/miss counters.
  void clearCache() {
    _cache.clear();
    _cacheHits = 0;
    _cacheMisses = 0;
  }

  /// Whether a shader with [cacheKey] is currently cached.
  bool isCached(String cacheKey) => _cache.containsKey(cacheKey);

  /// Compiles an [Expr] tree into a [WgslShaderModule], utilizing the LRU cache when available.
  ///
  /// Throws a [FormatException] if [validate] is `true` and the generated WGSL fails validation.
  WgslShaderModule compile(
    Expr expression, {
    String? kernelName,
    bool strided = false,
    int workgroupSize = 256,
    WgslDType outputDType = WgslDType.float32,
    bool validate = true,
  }) {
    final name =
        kernelName ??
        'fused_kernel_${expression.variables.map((v) => v.name).join("_")}';
    final descriptor = FusedKernelDescriptor(
      name: name,
      expression: expression,
      outputDType: outputDType,
      isStrided: strided,
    );

    return compileDescriptor(
      descriptor,
      workgroupSize: workgroupSize,
      validate: validate,
    );
  }

  /// Compiles an [Expr] tree into an executable [CompiledWgslKernel].
  ///
  /// Throws a [FormatException] if [validate] is `true` and the generated WGSL fails validation.
  CompiledWgslKernel compileKernel(
    Expr expression, {
    String? kernelName,
    String? name,
    bool strided = false,
    int workgroupSize = 256,
    WgslDType outputDType = WgslDType.float32,
    bool validate = true,
  }) {
    final effectiveName =
        kernelName ??
        name ??
        'fused_kernel_${expression.variables.map((v) => v.name).join("_")}';
    final descriptor = FusedKernelDescriptor(
      name: effectiveName,
      expression: expression,
      outputDType: outputDType,
      isStrided: strided,
    );
    return compileKernelDescriptor(
      descriptor,
      workgroupSize: workgroupSize,
      validate: validate,
    );
  }

  /// Compiles a [FusedKernelDescriptor] into an executable [CompiledWgslKernel].
  ///
  /// Throws a [FormatException] if [validate] is `true` and the generated WGSL fails validation.
  CompiledWgslKernel compileKernelDescriptor(
    FusedKernelDescriptor descriptor, {
    int workgroupSize = 256,
    bool validate = true,
  }) {
    final module = compileDescriptor(
      descriptor,
      workgroupSize: workgroupSize,
      validate: validate,
    );
    return CompiledWgslKernel(descriptor: descriptor, shaderModule: module);
  }

  /// Compiles (or retrieves from cache) and immediately executes [expression] on the GPU.
  ///
  /// Throws a [FormatException] if [validate] is `true` and the generated WGSL fails validation.
  GpuArray<T> execute<T extends DTypeTag>(
    Expr expression,
    Map<String, GpuArray<DTypeTag>> inputs, {
    Map<String, double>? scalars,
    List<int>? outputShape,
    List<int>? shape,
    DType<DTypeTag>? dtype,
    GpuDevice? device,
    GpuArray<T>? out,
    String? kernelName,
    String? name,
    bool strided = false,
    int workgroupSize = 256,
    bool validate = true,
  }) {
    final kernel = compileKernel(
      expression,
      kernelName: kernelName ?? name,
      strided: strided,
      workgroupSize: workgroupSize,
      validate: validate,
    );
    return kernel.run<T>(
      inputs: inputs,
      scalars: scalars,
      outputShape: outputShape,
      shape: shape,
      dtype: dtype,
      device: device,
      out: out,
    );
  }

  /// Compiles a [FusedKernelDescriptor] into a verified [WgslShaderModule].
  ///
  /// Throws a [FormatException] if [validate] is `true` and the generated WGSL fails validation.
  WgslShaderModule compileDescriptor(
    FusedKernelDescriptor descriptor, {
    int workgroupSize = 256,
    bool validate = true,
  }) {
    final cacheKey = descriptor.generateCacheKey();

    if (_cache.remove(cacheKey) case final cached?) {
      _cacheHits++;
      _cache[cacheKey] = cached;
      return cached;
    }

    _cacheMisses++;
    final code = descriptor.generateWgslSource(workgroupSize: workgroupSize);

    if (validate) {
      final validation = validateWgslShader(code);
      if (!validation.isValid) {
        throw FormatException(
          'WGSL JIT Compilation Failed with syntax errors:\n${validation.errors.join("\n")}\n\nGenerated Code:\n$code',
        );
      }
    }

    final module = WgslShaderModule(
      name: descriptor.name,
      code: code,
      entryPoint: 'main',
      workgroupSize: WgslWorkgroupSize(workgroupSize, 1, 1),
      bindings: descriptor.createBindings(),
      metadata: {
        'cacheKey': cacheKey,
        'expression': descriptor.expression.toFingerprint(),
        'inputs': descriptor.inputs.map((i) => i.name).toList(),
        'isStrided': descriptor.isStrided,
      },
    );

    if (_cache.length >= maxCacheSize && _cache.isNotEmpty) {
      _cache.remove(_cache.keys.first);
    }
    _cache[cacheKey] = module;
    return module;
  }
}
