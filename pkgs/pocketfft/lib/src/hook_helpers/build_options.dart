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

/// Build mode for resolving the native `pocketfft` library in `hook/build.dart`.
enum BuildModeEnum {
  /// Fetch precompiled and provenance-attested binary from GitHub Releases.
  fetch,

  /// Use a locally existing binary specified via `localPath`.
  local,

  /// Compile the native library from C source on the host machine.
  source,
}

/// Configuration options for the `pocketfft` native build hook.
final class BuildOptions {
  /// Selected build mode (`fetch`, `local`, or `source`).
  final BuildModeEnum buildMode;

  /// Whether [buildMode] was explicitly specified via user defines or environment variables.
  final bool isExplicit;

  /// Path to a prebuilt dynamic library when [buildMode] is [BuildModeEnum.local].
  final Uri? localPath;

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
    this.checkoutPath,
    this.sanitize,
    this.coverage = false,
  });

  /// Parses [BuildOptions] from `pubspec.yaml` `hooks.user_defines.pocketfft`
  /// with optional environment variable overrides (`POCKETFFT_BUILD_MODE`,
  /// `SCIENTIFIC_DART_BUILD_MODE`, `LOCAL_POCKETFFT_BINARY`,
  /// `LOCAL_POCKETFFT_CHECKOUT`).
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
      envFallback: env['POCKETFFT_SANITIZE'],
    );
    final sanitize = (rawSanitize != null && rawSanitize.trim().isNotEmpty)
        ? rawSanitize.trim()
        : null;

    final coverage = _parseBoolDefine(
      defines['coverage'],
      'coverage',
      envFallback: env['POCKETFFT_COVERAGE'],
    );

    final envMode =
        env['POCKETFFT_BUILD_MODE'] ?? env['SCIENTIFIC_DART_BUILD_MODE'];
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
        'Must be one of "fetch", "local", or "source" for package:pocketfft.',
      ),
    };

    final envLocalPath = env['LOCAL_POCKETFFT_BINARY'];
    final localPath = envLocalPath != null && envLocalPath.isNotEmpty
        ? Uri.file(envLocalPath)
        : defines.path('localPath');

    final envCheckoutPath = env['LOCAL_POCKETFFT_CHECKOUT'];
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
    }

    return BuildOptions(
      buildMode: buildMode,
      isExplicit: isExplicit,
      localPath: localPath,
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

Set the build mode for `pocketfft` with either `fetch`, `local`, or `source` in your workspace `pubspec.yaml`:

* fetch: Download the precompiled binary from GitHub Releases (verified via SHA-256 & SLSA provenance).
```yaml
hooks:
  user_defines:
    pocketfft:
      buildMode: fetch
```

* local: Use a locally existing binary (or set `LOCAL_POCKETFFT_BINARY`).
```yaml
hooks:
  user_defines:
    pocketfft:
      buildMode: local
      localPath: path/to/libpocketfft.so
```

* source: Compile a fresh library from C source on the host machine.
```yaml
hooks:
  user_defines:
    pocketfft:
      buildMode: source
```
''';

  @override
  String toString() =>
      'BuildOptions(buildMode: ${buildMode.name}, isExplicit: $isExplicit, '
      'localPath: $localPath, checkoutPath: $checkoutPath, '
      'sanitize: $sanitize, coverage: $coverage)';
}
