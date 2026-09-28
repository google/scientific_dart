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
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:gpuarray/src/hook_helpers/build_options.dart';
import 'package:test/test.dart';

void main() {
  group('Native Assets wgpu-native Hook Configuration', () {
    test('SHA-256 checksums are valid 64-character hex strings', () {
      final sha256Regex = RegExp(r'^[a-f0-9]{64}$');

      const expectedHashes = {
        'linux-x64':
            '95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a',
        'linux-arm64':
            '015fcdf1dbae82e614a783cc38017e5399ae0927a889fe9b69c9b664bc61b47a',
        'macos-arm64':
            'a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e',
        'macos-x64':
            '8e2f7378548ddd0e2cf21e7d864dda46e953f0af724855a33778b85ead206d41',
        'windows-x64':
            '7e67d7445c42aeb85e30f88930fd8d7d83ee769e3390aeb1ada75ebf3cf78132',
        'windows-arm64':
            '4a876421a8c1e5fe72f849b3722214280fe485cb1c56f77f8b0c82414be5b29f',
        'windows-ia32':
            'ad59d4eadfcfe667999a37e096cc551ecf3f56c387b5a7fd5f61baebf105f54a',
        'android-arm64':
            '721741f1b05a20c1738166bedf7a5efb2ba4b382da689526d3fc33de22bdd573',
        'android-armv7':
            'f9d76c77b3fda3f7121476884eb16ec067f7dada83276298a3cc8bf6a8403d60',
        'android-x86_64':
            'ef16fc0644bf0e308a39ac4516742da8e22d8c201d3a542cc5baf533d272c491',
        'android-i686':
            '593b94875bc4fcc1506ea0b6714dd12b96b7c852921caa63f45eb61517793312',
        'ios-arm64':
            'e36c9913b9e5095a530fa9121c50b16a4e3dd020e1eebf601f2f47ce24d56941',
        'ios-x64-simulator':
            '94f67e1b268e8dd31b8e59b32f211ac469f09ed7950fceee52bd84f0623da3d9',
      };

      for (final entry in expectedHashes.entries) {
        expect(
          sha256Regex.hasMatch(entry.value),
          isTrue,
          reason:
              'Hash for ${entry.key} must be valid 64-character lowercase hex',
        );
      }
    });

    test(
      'Live download and SHA-256 verification of Linux x64 wgpu-native release',
      () async {
        final url =
            'https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/wgpu-linux-x86_64-release.zip';
        final expectedSha =
            '95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a';

        final client = HttpClient();
        try {
          var currentUrl = Uri.parse(url);
          for (var i = 0; i < 5; i++) {
            final req = await client.getUrl(currentUrl);
            final res = await req.close();
            if (res.statusCode >= 300 &&
                res.statusCode < 400 &&
                res.headers.value(HttpHeaders.locationHeader) != null) {
              currentUrl = currentUrl.resolve(
                res.headers.value(HttpHeaders.locationHeader)!,
              );
              continue;
            }
            expect(res.statusCode, equals(200));
            final builder = BytesBuilder();
            await for (final chunk in res) {
              builder.add(chunk);
            }
            final bytes = builder.toBytes();
            final computedHash = sha256.convert(bytes).toString().toLowerCase();
            expect(computedHash, equals(expectedSha));
            break;
          }
        } finally {
          client.close();
        }
      },
      timeout: const Timeout(Duration(minutes: 2)),
    );

    test(
      'BuildOptions and BuildModeEnum configuration and error formatting',
      () {
        const defaultOpts = BuildOptions(buildMode: BuildModeEnum.fetch);
        expect(defaultOpts.buildMode, equals(BuildModeEnum.fetch));
        expect(defaultOpts.localPath, isNull);
        expect(defaultOpts.checkoutPath, isNull);
        expect(defaultOpts.toString(), contains('buildMode: fetch'));

        final localUri = Uri.file('/tmp/libwgpu_native.so');
        final localOpts = BuildOptions(
          buildMode: BuildModeEnum.local,
          localPath: localUri,
        );
        expect(localOpts.buildMode, equals(BuildModeEnum.local));
        expect(localOpts.localPath, equals(localUri));

        final errStr = BuildOptions.usageError(
          const FormatException('Invalid buildMode'),
        );
        expect(errStr, contains('Invalid buildMode'));
        expect(errStr, contains('buildMode: fetch'));
      },
    );
  });
}
