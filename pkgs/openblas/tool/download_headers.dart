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

void main() async {
  final buildFile = File('hook/build.dart');
  if (!buildFile.existsSync()) {
    print(
      'Error: hook/build.dart not found. Run this script from the package root.',
    );
    exit(1);
  }

  final buildContent = buildFile.readAsStringSync();
  final versionRegex = RegExp(r'OpenBLAS-(\d+\.\d+\.\d+)');
  final match = versionRegex.firstMatch(buildContent);

  if (match == null) {
    print('Error: Could not find OpenBLAS version in hook/build.dart');
    exit(1);
  }

  final version = match.group(1);
  print('Detected OpenBLAS version: $version');

  final files = ['cblas.h', 'common.h'];
  final client = HttpClient();

  for (final fileName in files) {
    final url =
        'https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/v$version/$fileName';
    print('Downloading from: $url');

    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();

    if (response.statusCode != 200) {
      print(
        'Error: Failed to download $fileName. Status code: ${response.statusCode}',
      );
      exit(1);
    }

    final targetFile = File('third_party/openblas/$fileName');
    if (!targetFile.parent.existsSync()) {
      targetFile.parent.createSync(recursive: true);
    }

    final bytes = <int>[];
    await for (final chunk in response) {
      bytes.addAll(chunk);
    }
    targetFile.writeAsBytesSync(bytes);
    print('Successfully downloaded $fileName to ${targetFile.path}');
  }
  client.close();

  print('Running ffigen...');
  final ffigenResult = await Process.run('dart', ['run', 'ffigen']);
  if (ffigenResult.exitCode != 0) {
    print('Error: Failed to run ffigen. Exit code: ${ffigenResult.exitCode}');
    print('Stdout: ${ffigenResult.stdout}');
    print('Stderr: ${ffigenResult.stderr}');
    exit(1);
  }
  print('Successfully ran ffigen.');
}
