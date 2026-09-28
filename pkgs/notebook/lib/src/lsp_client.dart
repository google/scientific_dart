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

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;

class LspCompletionResult {
  final String label;
  final String kind;
  final String? detail;

  LspCompletionResult({required this.label, required this.kind, this.detail});

  Map<String, dynamic> toJson() => {
    'label': label,
    'type': kind,
    if (detail != null) 'detail': detail,
  };
}

class LspClient {
  final String dartSdkPath;
  final String rootPath;

  Process? _process;
  int _nextRequestId = 1;
  final Map<int, Completer<Map<String, dynamic>>> _pendingRequests = {};
  final _stdoutBuffer = <int>[];
  int? _expectedLength;

  LspClient({required this.dartSdkPath, required this.rootPath});

  Future<void> start() async {
    final dartExec = p.join(dartSdkPath, 'bin', 'dart');
    _process = await Process.start(dartExec, ['language-server']);

    _process!.stdout.listen(_handleStdoutData);
    _process!.stderr.transform(utf8.decoder).listen((err) {
      // Ignore or log stderr warnings
    });

    // Send initialize
    final rootUri = p.toUri(rootPath).toString();
    final initResult = await _sendRequest('initialize', {
      'processId': pid,
      'rootUri': rootUri,
      'capabilities': {
        'textDocument': {
          'completion': {
            'completionItem': {
              'labelDetailsSupport': true,
              'snippetSupport': false,
            },
          },
        },
      },
    });

    if (initResult.containsKey('error')) {
      throw StateError('LSP Initialize failed: ${initResult['error']}');
    }

    _sendNotification('initialized', {});
  }

  void _handleStdoutData(List<int> chunk) {
    _stdoutBuffer.addAll(chunk);
    _processBuffer();
  }

  void _processBuffer() {
    while (true) {
      if (_expectedLength == null) {
        final headerEnd = _findSequence(_stdoutBuffer, [
          13,
          10,
          13,
          10,
        ]); // \r\n\r\n
        if (headerEnd == -1) return;

        final headerStr = utf8.decode(_stdoutBuffer.sublist(0, headerEnd));
        final match = RegExp(r'Content-Length:\s*(\d+)').firstMatch(headerStr);
        if (match != null) {
          _expectedLength = int.parse(match.group(1)!);
        } else {
          _expectedLength = 0;
        }

        _stdoutBuffer.removeRange(0, headerEnd + 4);
      }

      if (_stdoutBuffer.length >= _expectedLength!) {
        final bodyBytes = _stdoutBuffer.sublist(0, _expectedLength);
        _stdoutBuffer.removeRange(0, _expectedLength!);
        _expectedLength = null;

        try {
          final jsonStr = utf8.decode(bodyBytes);
          final msg = jsonDecode(jsonStr) as Map<String, dynamic>;
          _handleMessage(msg);
        } catch (e) {
          // Parse error handling
        }
      } else {
        break;
      }
    }
  }

  int _findSequence(List<int> list, List<int> seq) {
    for (var i = 0; i <= list.length - seq.length; i++) {
      var found = true;
      for (var j = 0; j < seq.length; j++) {
        if (list[i + j] != seq[j]) {
          found = false;
          break;
        }
      }
      if (found) return i;
    }
    return -1;
  }

  void _handleMessage(Map<String, dynamic> msg) {
    if (msg.containsKey('id') && _pendingRequests.containsKey(msg['id'])) {
      final completer = _pendingRequests.remove(msg['id'])!;
      completer.complete(msg);
    }
  }

  Future<Map<String, dynamic>> _sendRequest(
    String method,
    Map<String, dynamic> params,
  ) {
    final reqId = _nextRequestId++;
    final completer = Completer<Map<String, dynamic>>();
    _pendingRequests[reqId] = completer;

    final msg = {
      'jsonrpc': '2.0',
      'id': reqId,
      'method': method,
      'params': params,
    };

    _sendRaw(msg);
    return completer.future.timeout(
      const Duration(seconds: 5),
      onTimeout: () {
        _pendingRequests.remove(reqId);
        throw TimeoutException('LSP request $method timed out');
      },
    );
  }

  void _sendNotification(String method, Map<String, dynamic> params) {
    final msg = {'jsonrpc': '2.0', 'method': method, 'params': params};
    _sendRaw(msg);
  }

  void _sendRaw(Map<String, dynamic> msg) {
    final body = jsonEncode(msg);
    final bodyBytes = utf8.encode(body);
    final header = 'Content-Length: ${bodyBytes.length}\r\n\r\n';
    _process?.stdin.write(header);
    _process?.stdin.add(bodyBytes);
  }

  void didOpen(String fileUri, String content) {
    _sendNotification('textDocument/didOpen', {
      'textDocument': {
        'uri': fileUri,
        'languageId': 'dart',
        'version': 1,
        'text': content,
      },
    });
  }

  void didChange(String fileUri, String content, int version) {
    _sendNotification('textDocument/didChange', {
      'textDocument': {'uri': fileUri, 'version': version},
      'contentChanges': [
        {'text': content},
      ],
    });
  }

  Future<List<LspCompletionResult>> getCompletions(
    String fileUri,
    int line,
    int character,
  ) async {
    try {
      final response = await _sendRequest('textDocument/completion', {
        'textDocument': {'uri': fileUri},
        'position': {'line': line, 'character': character},
      });

      if (!response.containsKey('result')) return [];
      final result = response['result'];

      List<dynamic> itemsRaw = [];
      if (result is List) {
        itemsRaw = result;
      } else if (result is Map && result.containsKey('items')) {
        itemsRaw = result['items'] as List<dynamic>;
      }

      final results = <LspCompletionResult>[];
      for (final item in itemsRaw) {
        if (item is Map) {
          final label = item['label'] as String;
          final kindInt = item['kind'] as int? ?? 0;
          final detail = item['detail'] as String?;
          results.add(
            LspCompletionResult(
              label: label,
              kind: _mapLspKind(kindInt),
              detail: detail,
            ),
          );
        }
      }
      return results;
    } catch (e) {
      return [];
    }
  }

  String _mapLspKind(int kind) {
    switch (kind) {
      case 2:
        return 'method';
      case 3:
        return 'function';
      case 4:
        return 'constructor';
      case 5:
        return 'field';
      case 6:
        return 'variable';
      case 7:
        return 'class';
      case 8:
        return 'interface';
      case 13:
        return 'enum';
      case 14:
        return 'keyword';
      default:
        return 'property';
    }
  }

  Future<String?> getHover(String fileUri, int line, int character) async {
    try {
      final response = await _sendRequest('textDocument/hover', {
        'textDocument': {'uri': fileUri},
        'position': {'line': line, 'character': character},
      });

      if (!response.containsKey('result') || response['result'] == null) {
        return null;
      }
      final result = response['result'] as Map<String, dynamic>;
      if (!result.containsKey('contents')) return null;

      final contents = result['contents'];
      if (contents is String) return contents;
      if (contents is Map && contents.containsKey('value')) {
        return contents['value'] as String;
      }
      if (contents is List) {
        final parts = <String>[];
        for (final item in contents) {
          if (item is String) {
            parts.add(item);
          } else if (item is Map && item.containsKey('value')) {
            parts.add(item['value'] as String);
          }
        }
        return parts.isEmpty ? null : parts.join('\n\n');
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  Future<void> stop() async {
    _process?.kill();
  }
}
