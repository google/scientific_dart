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

import 'dart:convert';
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';

/// Build mode for resolving the native `ndarray` library in `hook/build.dart`.
enum BuildModeEnum {
  /// Fetch precompiled and provenance-attested binary from GitHub Releases.
  fetch,

  /// Use a locally existing binary specified via `localPath`.
  local,

  /// Compile the native library from C/C++ source on the host machine.
  source,
}

/// Configuration options for the `ndarray` native build hook.
final class BuildOptions {
  /// Selected build mode (`fetch`, `local`, or `source`).
  final BuildModeEnum buildMode;

  /// Whether [buildMode] was explicitly specified via user defines or environment variables.
  final bool isExplicit;

  /// Path to a prebuilt dynamic library when [buildMode] is [BuildModeEnum.local].
  final Uri? localPath;

  /// Path to a local package/source checkout when [buildMode] is [BuildModeEnum.source].
  final Uri? checkoutPath;

  /// Comma-separated C/C++ sanitizer list (for example `'address,undefined'`),
  /// or `null` when sanitizers are disabled.
  final String? sanitize;

  /// Whether native gcov/llvm-cov coverage instrumentation (`--coverage`) is enabled.
  final bool coverage;

  /// Custom x86-64 ISA flags for `source` builds (or `""` for baseline x86-64
  /// without AVX2/FMA/F16C), or `null` to use the default AVX2 flags.
  final String? x86Flags;

  /// Creates a [BuildOptions] configuration.
  const BuildOptions({
    required this.buildMode,
    this.isExplicit = false,
    this.localPath,
    this.checkoutPath,
    this.sanitize,
    this.coverage = false,
    this.x86Flags,
  });

  /// Normalized sanitizer list including `float-cast-overflow` whenever
  /// `undefined` is enabled.
  String? get effectiveSanitize {
    final raw = sanitize?.trim();
    if (raw == null || raw.isEmpty) return null;
    final parts = raw
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    if (parts.isEmpty) return null;
    if (parts.contains('undefined') && !parts.contains('float-cast-overflow')) {
      parts.add('float-cast-overflow');
    }
    return parts.join(',');
  }

  /// Compiler and linker flags for native sanitizers.
  List<String> get sanitizeFlags {
    final eff = effectiveSanitize;
    if (eff == null) return const <String>[];
    return <String>[
      '-fsanitize=$eff',
      '-fno-sanitize-recover=all',
      '-fno-omit-frame-pointer',
      '-g',
    ];
  }

  /// Compiler and linker flags for native code coverage.
  List<String> get coverageFlags =>
      coverage ? const <String>['--coverage', '-O1', '-g'] : const <String>[];

  /// Whether sanitizer or coverage instrumentation is enabled.
  bool get hasInstrumentation => effectiveSanitize != null || coverage;

  /// Resolves the x86-64 ISA compiler flags for the target toolchain.
  ///
  /// When [x86Flags] is `null`, defaults to AVX2+FMA+F16C. When [x86Flags] is
  /// `""` or `"none"`, returns an empty list (baseline x86-64).
  List<String> effectiveX86Flags({required bool isMSVC}) {
    if (x86Flags == null) {
      return isMSVC
          ? const <String>['/arch:AVX2']
          : const <String>['-mavx2', '-mfma', '-mf16c'];
    }
    final trimmed = x86Flags!.trim();
    if (trimmed.isEmpty || trimmed.toLowerCase() == 'none') {
      return const <String>[];
    }
    return trimmed.split(RegExp(r'\s+')).where((s) => s.isNotEmpty).toList();
  }

  /// Deterministic cache key suffix incorporating [sanitize], [coverage], and
  /// [x86Flags] so instrumented or custom-ISA builds never collide with cached
  /// default builds.
  String get cacheKey {
    final effSanitize = effectiveSanitize;
    if (effSanitize == null && !coverage && x86Flags == null) {
      return '';
    }
    final raw =
        'sanitize=${effSanitize ?? ''}|coverage=$coverage|x86Flags=${x86Flags ?? 'default'}';
    final digest = sha256.convert(utf8.encode(raw)).toString();
    return '-${digest.substring(0, 12)}';
  }

  /// Validates that target-specific options in this configuration can be
  /// applied to `targetOS`, `targetArch`, and `isMSVC`.
  ///
  /// Throws an [UnsupportedError] if an option cannot be applied on the target.
  void validateTarget({
    required OS targetOS,
    required Architecture targetArch,
    required bool isMSVC,
  }) {
    if (x86Flags != null && targetArch != Architecture.x64) {
      throw UnsupportedError(
        'x86Flags ("$x86Flags") cannot be applied when targeting '
        '${targetArch.name}; x86Flags is only supported on x64.',
      );
    }
    if (effectiveSanitize != null && isMSVC) {
      throw UnsupportedError(
        'sanitize ("$sanitize") is not supported with the MSVC toolchain on '
        'Windows; use GCC or Clang on Linux/macOS.',
      );
    }
    if (coverage && isMSVC) {
      throw UnsupportedError(
        'coverage is not supported with the MSVC toolchain on Windows; '
        'use GCC or Clang on Linux/macOS.',
      );
    }
  }

  /// Parses [BuildOptions] from `pubspec.yaml` `hooks.user_defines.ndarray`
  /// with optional environment variable overrides (`NDARRAY_BUILD_MODE`,
  /// `SCIENTIFIC_DART_BUILD_MODE`, `LOCAL_NDARRAY_BINARY`,
  /// `LOCAL_NDARRAY_CHECKOUT`, `NDARRAY_SANITIZE`, `NDARRAY_COVERAGE`,
  /// `NDARRAY_X86_FLAGS`).
  ///
  /// It is an error if `buildMode` is not one of `fetch`, `local`, or `source`,
  /// or if source-only options (`sanitize`, `coverage`, `x86Flags`,
  /// `checkoutPath`) or `localPath` are combined with an incompatible
  /// `buildMode`.
  factory BuildOptions.fromDefines(
    HookInputUserDefines defines, {
    Map<String, String>? environment,
  }) {
    final env = environment ?? Platform.environment;

    final rawSanitize = _parseStringDefine(
      defines['sanitize'],
      'sanitize',
      envFallback: env['NDARRAY_SANITIZE'],
    );
    final sanitize = (rawSanitize != null && rawSanitize.trim().isNotEmpty)
        ? rawSanitize.trim()
        : null;

    final coverage = _parseBoolDefine(
      defines['coverage'],
      'coverage',
      envFallback: env['NDARRAY_COVERAGE'],
    );

    String? x86Flags;
    final rawX86Define = defines['x86Flags'];
    if (rawX86Define != null) {
      if (rawX86Define is String) {
        x86Flags = rawX86Define;
      } else if (rawX86Define is List) {
        x86Flags = rawX86Define.map((e) => e.toString()).join(' ');
      } else {
        throw ArgumentError.value(
          rawX86Define,
          'x86Flags',
          'Must be a String or List of flag strings.',
        );
      }
    } else if (env.containsKey('NDARRAY_X86_FLAGS')) {
      x86Flags = env['NDARRAY_X86_FLAGS'];
    }

    final envMode =
        env['NDARRAY_BUILD_MODE'] ?? env['SCIENTIFIC_DART_BUILD_MODE'];
    final rawMode = envMode ?? defines['buildMode'];
    final isExplicit = rawMode != null;
    final hasSourceOnlyFlags = sanitize != null || coverage || x86Flags != null;

    final buildMode = switch (rawMode) {
      null => hasSourceOnlyFlags ? BuildModeEnum.source : BuildModeEnum.fetch,
      'fetch' => BuildModeEnum.fetch,
      'local' => BuildModeEnum.local,
      'source' || 'checkout' => BuildModeEnum.source,
      final other => throw ArgumentError.value(
        other,
        'buildMode',
        'Must be one of "fetch", "local", or "source" for package:ndarray.',
      ),
    };

    final envLocalPath = env['LOCAL_NDARRAY_BINARY'];
    final localPath = envLocalPath != null && envLocalPath.isNotEmpty
        ? Uri.file(envLocalPath)
        : defines.path('localPath');

    final envCheckoutPath = env['LOCAL_NDARRAY_CHECKOUT'];
    final checkoutPath = envCheckoutPath != null && envCheckoutPath.isNotEmpty
        ? Uri.directory(envCheckoutPath)
        : defines.path('checkoutPath');

    if (localPath != null && buildMode != BuildModeEnum.local) {
      throw ArgumentError.value(
        localPath.toString(),
        'localPath',
        'Must only be set when buildMode is "local" '
            '(got buildMode "${buildMode.name}").',
      );
    }
    if (checkoutPath != null && buildMode != BuildModeEnum.source) {
      throw ArgumentError.value(
        checkoutPath.toString(),
        'checkoutPath',
        'Must only be set when buildMode is "source" '
            '(got buildMode "${buildMode.name}").',
      );
    }
    if (buildMode != BuildModeEnum.source) {
      if (sanitize != null) {
        throw ArgumentError.value(
          sanitize,
          'sanitize',
          'Must only be used with buildMode "source" '
              '(cannot apply sanitizers in "${buildMode.name}" mode).',
        );
      }
      if (coverage) {
        throw ArgumentError.value(
          coverage,
          'coverage',
          'Must only be used with buildMode "source" '
              '(cannot collect native coverage in "${buildMode.name}" mode).',
        );
      }
      if (x86Flags != null) {
        throw ArgumentError.value(
          x86Flags,
          'x86Flags',
          'Must only be used with buildMode "source" '
              '(cannot customize x86 compiler flags in "${buildMode.name}" mode).',
        );
      }
    }

    return BuildOptions(
      buildMode: buildMode,
      isExplicit: isExplicit,
      localPath: localPath,
      checkoutPath: checkoutPath,
      sanitize: sanitize,
      coverage: coverage,
      x86Flags: x86Flags,
    );
  }

  static String? _parseStringDefine(
    Object? raw,
    String name, {
    String? envFallback,
  }) {
    if (raw != null) {
      if (raw is String) return raw;
      throw ArgumentError.value(raw, name, 'Must be a String.');
    }
    return envFallback;
  }

  static bool _parseBoolDefine(
    Object? raw,
    String name, {
    String? envFallback,
  }) {
    final value = raw ?? envFallback;
    if (value == null) return false;
    if (value is bool) return value;
    if (value is num) return value != 0;
    if (value is String) {
      final lower = value.trim().toLowerCase();
      if (lower.isEmpty || lower == 'false' || lower == '0') return false;
      if (lower == 'true' || lower == '1') return true;
    }
    throw ArgumentError.value(
      value,
      name,
      'Must be a boolean ("true", "false", "1", or "0").',
    );
  }

  /// Returns a formatted usage message for `pubspec.yaml` configuration.
  static String usageError(Object error) =>
      '''
Error: $error

Set the build mode for `ndarray` with either `fetch`, `local`, or `source` in your workspace `pubspec.yaml`:

* fetch: Download the precompiled binary from GitHub Releases (verified via SHA-256 & SLSA provenance).
```yaml
hooks:
  user_defines:
    ndarray:
      buildMode: fetch
```

* local: Use a locally existing binary (or set `LOCAL_NDARRAY_BINARY`).
```yaml
hooks:
  user_defines:
    ndarray:
      buildMode: local
      localPath: path/to/libndarray.so
```

* source: Compile a fresh library from C/C++ source on the host machine.
```yaml
hooks:
  user_defines:
    ndarray:
      buildMode: source
      # Optional source-mode flags:
      # sanitize: address,undefined
      # coverage: true
      # x86Flags: ""
```
''';

  @override
  String toString() =>
      'BuildOptions(buildMode: ${buildMode.name}, isExplicit: $isExplicit, '
      'localPath: $localPath, checkoutPath: $checkoutPath, '
      'sanitize: $sanitize, coverage: $coverage, x86Flags: $x86Flags)';
}
