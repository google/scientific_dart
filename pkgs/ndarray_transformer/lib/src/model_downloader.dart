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

import 'package:http/http.dart' as http;

/// Default Hugging Face repository ID for the 22M-parameter MiniLM embedder.
const String defaultMiniLmRepo = 'sentence-transformers/all-MiniLM-L6-v2';

/// Files required from a Hugging Face BERT/MiniLM repository.
const List<String> requiredModelFiles = <String>[
  'config.json',
  'tokenizer.json',
  'model.safetensors',
];

/// Downloads `config.json`, `tokenizer.json`, and `model.safetensors` from a
/// Hugging Face repository into [targetDir], skipping files that already exist.
///
/// Streams payloads to temporary `.part` files before atomic rename so an
/// interrupted download never leaves a corrupt cached file.
Future<Directory> downloadHuggingFaceModel({
  required Directory targetDir,
  String repoId = defaultMiniLmRepo,
  String revision = 'main',
  void Function(String message)? onProgress,
  http.Client? client,
}) async {
  targetDir.createSync(recursive: true);
  final ownedClient = client == null;
  final httpClient = client ?? http.Client();
  try {
    for (final filename in requiredModelFiles) {
      final dest = File('${targetDir.path}/$filename');
      if (dest.existsSync() && dest.lengthSync() > 0) {
        onProgress?.call('Cached: ${dest.path}');
        continue;
      }
      final uri = Uri.https(
        'huggingface.co',
        '/$repoId/resolve/$revision/$filename',
      );
      onProgress?.call('Downloading $uri ...');
      final request = http.Request('GET', uri);
      final response = await httpClient.send(request);
      if (response.statusCode != 200) {
        await response.stream.drain<void>();
        throw HttpException(
          'Failed to download $uri (HTTP ${response.statusCode})',
          uri: uri,
        );
      }

      final partFile = File('${dest.path}.part');
      final sink = partFile.openWrite();
      var bytesWritten = 0;
      try {
        await for (final chunk in response.stream) {
          sink.add(chunk);
          bytesWritten += chunk.length;
        }
        await sink.flush();
      } catch (_) {
        await sink.close();
        if (partFile.existsSync()) partFile.deleteSync();
        rethrow;
      }
      await sink.close();
      partFile.renameSync(dest.path);
      onProgress?.call(
        'Saved ${dest.path} (${(bytesWritten / (1024 * 1024)).toStringAsFixed(2)} MiB)',
      );
    }
    return targetDir;
  } finally {
    if (ownedClient) httpClient.close();
  }
}
