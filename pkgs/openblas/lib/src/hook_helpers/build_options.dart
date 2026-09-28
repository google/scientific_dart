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

import 'dart:io';

import 'package:hooks/hooks.dart';

/// Build mode for resolving the native `openblas` libraries in `hook/build.dart`.
enum BuildModeEnum {
  /// Fetch precompiled and provenance-attested binaries from GitHub Releases.
  fetch,

  /// Use locally existing binaries specified via `localPath` (and optional `localExtensionsPath`).
  local,

  /// Compile OpenBLAS and/or custom extensions on the host machine.
  source,
}

/// Configuration options for the `openblas` native build hook.
final class BuildOptions {
  /// Selected build mode (`fetch`, `local`, or `source`).
  final BuildModeEnum buildMode;

  /// Whether [buildMode] was explicitly specified via user defines or environment variables.
  final bool isExplicit;

  /// Path to a directory containing `libopenblas` and `libopenblas_extensions`
  /// (or path to `libopenblas` directly when [localExtensionsPath] is also provided).
  final Uri? localPath;

  /// Optional path to `libopenblas_extensions` when [localPath] points to the
  /// `libopenblas` file rather than a directory.
  final Uri? localExtensionsPath;

  /// Path to a local package/source checkout when [buildMode] is [BuildModeEnum.source].
  final Uri? checkoutPath;

  /// Comma-separated C sanitizer list, or `null` when sanitizers are disabled.
  final String? sanitize;

  /// Whether native code coverage instrumentation is enabled.
  final bool coverage;

  /// Creates a [BuildOptions] configuration.
  const BuildOptions({
    required this.buildMode,
    this.isExplicit = false,
    this.localPath,
    this.localExtensionsPath,
    this.checkoutPath,
    this.sanitize,
    this.coverage = false,
  });

  /// Parses [BuildOptions] from `pubspec.yaml` `hooks.user_defines.openblas`
  /// with optional environment variable overrides (`OPENBLAS_BUILD_MODE`,
  /// `SCIENTIFIC_DART_BUILD_MODE`, `LOCAL_OPENBLAS_BINARY`,
  /// `LOCAL_OPENBLAS_EXTENSIONS_BINARY`, `LOCAL_OPENBLAS_CHECKOUT`).
  ///
  /// It is an error if `buildMode` is not one of `fetch`, `local`, or `source`,
  /// or if mode-specific options conflict with [buildMode].
  factory BuildOptions.fromDefines(
    HookInputUserDefines defines, {
    Map<String, String>? environment,
  }) {
    final env = environment ?? Platform.environment;

    final rawSanitize = _parseStringDefine(
      defines['sanitize'],
      'sanitize',
      envFallback: env['OPENBLAS_SANITIZE'],
    );
    final sanitize = (rawSanitize != null && rawSanitize.trim().isNotEmpty)
        ? rawSanitize.trim()
        : null;

    final coverage = _parseBoolDefine(
      defines['coverage'],
      'coverage',
      envFallback: env['OPENBLAS_COVERAGE'],
    );

    final envMode =
        env['OPENBLAS_BUILD_MODE'] ?? env['SCIENTIFIC_DART_BUILD_MODE'];
    final rawMode = envMode ?? defines['buildMode'];
    final isExplicit = rawMode != null;
    final hasSourceOnlyFlags = sanitize != null || coverage;

    final buildMode = switch (rawMode) {
      null => hasSourceOnlyFlags ? BuildModeEnum.source : BuildModeEnum.fetch,
      'fetch' => BuildModeEnum.fetch,
      'local' => BuildModeEnum.local,
      'source' || 'checkout' => BuildModeEnum.source,
      final other => throw ArgumentError.value(
        other,
        'buildMode',
        'Must be one of "fetch", "local", or "source" for package:openblas.',
      ),
    };

    final envLocalPath = env['LOCAL_OPENBLAS_BINARY'];
    Uri? localPath;
    if (envLocalPath != null && envLocalPath.isNotEmpty) {
      localPath = FileSystemEntity.isDirectorySync(envLocalPath)
          ? Uri.directory(envLocalPath)
          : Uri.file(envLocalPath);
    } else {
      localPath = defines.path('localPath');
    }

    final envLocalExtPath = env['LOCAL_OPENBLAS_EXTENSIONS_BINARY'];
    final localExtensionsPath =
        envLocalExtPath != null && envLocalExtPath.isNotEmpty
        ? Uri.file(envLocalExtPath)
        : defines.path('localExtensionsPath');

    final envCheckoutPath = env['LOCAL_OPENBLAS_CHECKOUT'];
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
    if (localExtensionsPath != null && buildMode != BuildModeEnum.local) {
      throw ArgumentError.value(
        localExtensionsPath.toString(),
        'localExtensionsPath',
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
    }

    return BuildOptions(
      buildMode: buildMode,
      isExplicit: isExplicit,
      localPath: localPath,
      localExtensionsPath: localExtensionsPath,
      checkoutPath: checkoutPath,
      sanitize: sanitize,
      coverage: coverage,
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

Set the build mode for `openblas` with either `fetch`, `local`, or `source` in your workspace `pubspec.yaml`:

* fetch: Download the precompiled binaries from GitHub Releases (verified via SHA-256 & SLSA provenance).
```yaml
hooks:
  user_defines:
    openblas:
      buildMode: fetch
```

* local: Use locally existing binaries from a directory or explicit file paths.
```yaml
hooks:
  user_defines:
    openblas:
      buildMode: local
      localPath: path/to/dir_or_libopenblas.so
      localExtensionsPath: path/to/libopenblas_extensions.so # optional if localPath is a directory
```

* source: Compile OpenBLAS / Accelerate wrappers and custom extensions on the host machine.
```yaml
hooks:
  user_defines:
    openblas:
      buildMode: source
```
''';

  @override
  String toString() =>
      'BuildOptions(buildMode: ${buildMode.name}, isExplicit: $isExplicit, '
      'localPath: $localPath, localExtensionsPath: $localExtensionsPath, '
      'checkoutPath: $checkoutPath, sanitize: $sanitize, coverage: $coverage)';
}
