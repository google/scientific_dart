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

/// Build mode for resolving the native `symbolic_dart` libraries (`flint` and `symengine`).
enum BuildModeEnum {
  /// Fetch precompiled and provenance-attested binaries from GitHub Releases.
  fetch,

  /// Use locally existing binaries from `localPath` directory.
  local,

  /// Compile MPFR, FLINT, and SymEngine from source on the host machine.
  source,
}

/// Configuration options for the `symbolic_dart` native build hook.
final class BuildOptions {
  /// Selected build mode (`fetch`, `local`, or `source`).
  final BuildModeEnum buildMode;

  /// Path to a directory containing `libflint` and `libsymengine` when [buildMode] is [BuildModeEnum.local].
  final Uri? localPath;

  /// Path to a local package/source checkout when [buildMode] is [BuildModeEnum.source].
  final Uri? checkoutPath;

  /// Creates a [BuildOptions] configuration.
  const BuildOptions({
    required this.buildMode,
    this.localPath,
    this.checkoutPath,
  });

  /// Parses [BuildOptions] from `pubspec.yaml` `hooks.user_defines.symbolic_dart`
  /// with optional environment variable overrides (`SYMBOLIC_DART_BUILD_MODE`,
  /// `SCIENTIFIC_DART_BUILD_MODE`, `LOCAL_SYMBOLIC_DART_BINARY`).
  factory BuildOptions.fromDefines(HookInputUserDefines defines) {
    final envMode =
        Platform.environment['SYMBOLIC_DART_BUILD_MODE'] ??
        Platform.environment['SCIENTIFIC_DART_BUILD_MODE'];
    final rawMode = envMode ?? defines['buildMode'];
    final buildMode = switch (rawMode) {
      null || 'fetch' => BuildModeEnum.fetch,
      'local' => BuildModeEnum.local,
      'source' || 'checkout' => BuildModeEnum.source,
      final other => throw ArgumentError(
        'Unknown buildMode "$other" for package:symbolic_dart.',
      ),
    };

    final envLocalPath = Platform.environment['LOCAL_SYMBOLIC_DART_BINARY'];
    final localPath = envLocalPath != null && envLocalPath.isNotEmpty
        ? Uri.directory(envLocalPath)
        : defines.path('localPath');

    final envCheckoutPath =
        Platform.environment['LOCAL_SYMBOLIC_DART_CHECKOUT'];
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

Set the build mode for `symbolic_dart` with either `fetch`, `local`, or `source` in your workspace `pubspec.yaml`:

* fetch: Download the precompiled binaries from GitHub Releases (verified via SHA-256 & SLSA provenance).
```yaml
hooks:
  user_defines:
    symbolic_dart:
      buildMode: fetch
```

* local: Use locally existing binaries in a directory.
```yaml
hooks:
  user_defines:
    symbolic_dart:
      buildMode: local
      localPath: path/to/lib_dir/
```

* source: Compile MPFR, FLINT, and SymEngine from source on the host machine.
```yaml
hooks:
  user_defines:
    symbolic_dart:
      buildMode: source
```
''';

  @override
  String toString() =>
      'BuildOptions(buildMode: ${buildMode.name}, localPath: $localPath, checkoutPath: $checkoutPath)';
}
