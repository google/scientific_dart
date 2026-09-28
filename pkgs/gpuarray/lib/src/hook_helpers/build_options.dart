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

/// Build mode for resolving the native `wgpu_native` library in `package:gpuarray`.
enum BuildModeEnum {
  /// Fetch precompiled `wgpu-native` binary release (verified via SHA-256).
  fetch,

  /// Use a locally existing `wgpu-native` binary specified via `localPath`.
  local,

  /// Build `wgpu-native` from a local Rust checkout specified via `checkoutPath`.
  source,
}

/// Configuration options for the `gpuarray` native build hook.
final class BuildOptions {
  /// Selected build mode (`fetch`, `local`, or `source`).
  final BuildModeEnum buildMode;

  /// Path to a local `wgpu-native` dynamic library when [buildMode] is [BuildModeEnum.local].
  final Uri? localPath;

  /// Path to a local `wgpu-native` Cargo checkout when [buildMode] is [BuildModeEnum.source].
  final Uri? checkoutPath;

  /// Creates a [BuildOptions] configuration.
  const BuildOptions({
    required this.buildMode,
    this.localPath,
    this.checkoutPath,
  });

  /// Parses [BuildOptions] from `pubspec.yaml` `hooks.user_defines.gpuarray`
  /// with optional environment variable overrides (`GPUARRAY_BUILD_MODE`,
  /// `SCIENTIFIC_DART_BUILD_MODE`, `LOCAL_GPUARRAY_BINARY`).
  factory BuildOptions.fromDefines(HookInputUserDefines defines) {
    final envMode =
        Platform.environment['GPUARRAY_BUILD_MODE'] ??
        Platform.environment['SCIENTIFIC_DART_BUILD_MODE'];
    final rawMode = envMode ?? defines['buildMode'];
    final buildMode = switch (rawMode) {
      null || 'fetch' => BuildModeEnum.fetch,
      'local' => BuildModeEnum.local,
      'source' || 'checkout' => BuildModeEnum.source,
      final other => throw ArgumentError(
        'Unknown buildMode "$other" for package:gpuarray.',
      ),
    };

    final envLocalPath = Platform.environment['LOCAL_GPUARRAY_BINARY'];
    final localPath = envLocalPath != null && envLocalPath.isNotEmpty
        ? Uri.file(envLocalPath)
        : defines.path('localPath');

    final envCheckoutPath = Platform.environment['LOCAL_GPUARRAY_CHECKOUT'];
    final checkoutPath = envCheckoutPath != null && envCheckoutPath.isNotEmpty
        ? Uri.directory(envCheckoutPath)
        : defines.path('checkoutPath');

    return BuildOptions(
      buildMode: buildMode,
      localPath: localPath,
      checkoutPath: checkoutPath,
    );
  }

  /// Returns a formatted usage message for `pubspec.yaml` configuration.
  static String usageError(Object error) =>
      '''
Error: $error

Set the build mode for `gpuarray` with either `fetch`, `local`, or `source` in your workspace `pubspec.yaml`:

* fetch: Download the precompiled wgpu-native binary (verified via SHA-256).
```yaml
hooks:
  user_defines:
    gpuarray:
      buildMode: fetch
```

* local: Use a locally existing wgpu-native binary.
```yaml
hooks:
  user_defines:
    gpuarray:
      buildMode: local
      localPath: path/to/libwgpu_native.so
```

* source: Build wgpu-native from a local git checkout via Cargo.
```yaml
hooks:
  user_defines:
    gpuarray:
      buildMode: source
      checkoutPath: path/to/wgpu-native
```
''';

  @override
  String toString() =>
      'BuildOptions(buildMode: ${buildMode.name}, localPath: $localPath, checkoutPath: $checkoutPath)';
}
