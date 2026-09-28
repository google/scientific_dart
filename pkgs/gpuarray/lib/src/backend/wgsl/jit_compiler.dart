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

import 'wgsl_types.dart';
import 'kernel_fusion.dart';

/// Validation diagnostic results for WGSL compute shader syntax and binding layouts.
final class WgslValidationResult {
  final bool isValid;
  final List<String> errors;
  final List<String> warnings;

  const WgslValidationResult({
    required this.isValid,
    this.errors = const [],
    this.warnings = const [],
  });

  @override
  String toString() {
    if (isValid) {
      return 'WgslValidationResult(VALID${warnings.isNotEmpty ? ", warnings: ${warnings.length}" : ""})';
    }
    return 'WgslValidationResult(INVALID, errors: $errors)';
  }
}

/// Static validator for checking WGSL compute shader syntax, structure, and bindings.
final class WgslSyntaxValidator {
  WgslSyntaxValidator._();

  /// Validates a WGSL compute shader string against standard structure and WebGPU rules.
  static WgslValidationResult validate(String code) {
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
      final ch = cleanCode[i];
      if (ch == '{') braceCount++;
      if (ch == '}') braceCount--;
      if (ch == '(') parenCount++;
      if (ch == ')') parenCount--;
      if (ch == '[') bracketCount++;
      if (ch == ']') bracketCount--;

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
      if (seenBindings.contains(key)) {
        errors.add(
          'Duplicate resource binding detected: @group($group) @binding($binding)',
        );
      }
      seenBindings.add(key);
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

    return WgslValidationResult(
      isValid: errors.isEmpty,
      errors: errors,
      warnings: warnings,
    );
  }
}

/// Dynamic JIT Compiler for generating and fusing WGSL compute shaders.
final class WgslJitCompiler {
  final Map<String, WgslShaderModule> _cache;
  final int maxCacheSize;
  int _cacheHits = 0;
  int _cacheMisses = 0;

  WgslJitCompiler({
    Map<String, WgslShaderModule>? cache,
    this.maxCacheSize = 512,
  }) : _cache = cache ?? <String, WgslShaderModule>{};

  /// Singleton instance of the JIT compiler.
  static final WgslJitCompiler instance = WgslJitCompiler();

  /// Total number of cache hits.
  int get cacheHits => _cacheHits;

  /// Total number of cache misses.
  int get cacheMisses => _cacheMisses;

  /// Total number of cached shader modules.
  int get cachedCount => _cache.length;

  /// Clears the compilation cache.
  void clearCache() {
    _cache.clear();
    _cacheHits = 0;
    _cacheMisses = 0;
  }

  /// Checks whether a shader with [cacheKey] is already cached.
  bool isCached(String cacheKey) => _cache.containsKey(cacheKey);

  /// Compiles an [Expr] tree into a [WgslShaderModule], utilizing cache when available.
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

  /// Compiles a [FusedKernelDescriptor] into a verified [WgslShaderModule].
  WgslShaderModule compileDescriptor(
    FusedKernelDescriptor descriptor, {
    int workgroupSize = 256,
    bool validate = true,
  }) {
    final cacheKey = descriptor.generateCacheKey();

    if (_cache.containsKey(cacheKey)) {
      _cacheHits++;
      final cached = _cache.remove(cacheKey)!;
      _cache[cacheKey] = cached;
      return cached;
    }

    _cacheMisses++;
    final code = descriptor.generateWgslSource(workgroupSize: workgroupSize);

    if (validate) {
      final validation = WgslSyntaxValidator.validate(code);
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
