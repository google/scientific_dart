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
import 'package:notebook/src/notebook_kernel.dart';
import 'package:notebook/src/ipynb.dart';
import 'package:markdown/markdown.dart' as md;

class NotebookServer {
  final String workspaceDir;
  final String dartSdkPath;
  final int port;
  final bool serverlessWasm;
  final String? staticBundleDir;

  HttpServer? _server;
  NotebookKernel? _kernel;
  final Set<WebSocket> _connectedClients = {};

  NotebookServer({
    required this.workspaceDir,
    required this.dartSdkPath,
    this.port = 8080,
    this.serverlessWasm = false,
    this.staticBundleDir,
  });

  int get actualPort => _server?.port ?? port;

  List<Map<String, dynamic>> _sessionCells = [];

  File get _sessionIpynbFile =>
      File(p.join(workspaceDir, 'notebook_session.ipynb'));
  File get _sessionJsonFile =>
      File(p.join(workspaceDir, 'notebook_session.json'));

  void _loadSessionData() {
    try {
      if (_sessionIpynbFile.existsSync()) {
        final content = _sessionIpynbFile.readAsStringSync();
        if (content.trim().isNotEmpty) {
          final nb = IpynbNotebook.fromJsonString(content);
          _sessionCells = nb.toSessionCells();
          return;
        }
      }

      if (_sessionJsonFile.existsSync()) {
        final content = _sessionJsonFile.readAsStringSync();
        if (content.trim().isNotEmpty) {
          final decoded = jsonDecode(content);
          if (decoded is List) {
            _sessionCells = List<Map<String, dynamic>>.from(
              decoded.map((e) => Map<String, dynamic>.from(e as Map)),
            );
          }
        }
      }
    } catch (e) {
      print('Warning: Failed to load session file: $e');
    }
  }

  void _saveSessionData([List<dynamic>? cells]) {
    try {
      if (cells != null) {
        _sessionCells = List<Map<String, dynamic>>.from(
          cells.map((e) => Map<String, dynamic>.from(e as Map)),
        );
      }
      final ipynb = IpynbNotebook.fromSessionCells(_sessionCells);
      _sessionIpynbFile.writeAsStringSync(ipynb.toJsonString(pretty: true));
      _sessionJsonFile.writeAsStringSync(jsonEncode(_sessionCells));
    } catch (e) {
      print('Warning: Failed to save session file: $e');
    }
  }

  void _broadcastSession() {
    final payload = jsonEncode({
      'type': 'init_session',
      'cells': _sessionCells,
    });
    for (final client in _connectedClients) {
      try {
        client.add(payload);
      } catch (_) {}
    }
  }

  void _updateOrAddCell({
    required String id,
    String? code,
    String? output,
    bool? isError,
    bool? evaluated,
    String? type,
  }) {
    final index = _sessionCells.indexWhere((c) => c['id'] == id);
    if (index != -1) {
      if (code != null) _sessionCells[index]['code'] = code;
      if (output != null) _sessionCells[index]['output'] = output;
      if (isError != null) _sessionCells[index]['isError'] = isError;
      if (evaluated != null) _sessionCells[index]['evaluated'] = evaluated;
      if (type != null) _sessionCells[index]['type'] = type;
    } else {
      _sessionCells.add({
        'id': id,
        'type': type ?? 'code',
        'code': code ?? '',
        'output': output ?? '',
        'isError': isError ?? false,
        'evaluated': evaluated ?? false,
      });
    }
    _saveSessionData();
  }

  Future<void> start() async {
    if (!serverlessWasm) {
      _loadSessionData();

      _kernel = NotebookKernel(
        workspaceDir: workspaceDir,
        dartSdkPath: dartSdkPath,
      );

      print('Starting notebook kernel...');
      await _kernel!.start();
      print('Kernel started successfully.');
    } else {
      print(
        'Starting in Serverless Wasm mode (static files only, no active VM kernel)...',
      );
    }

    _server = await HttpServer.bind(InternetAddress.anyIPv4, port);
    print('Server listening on http://localhost:$actualPort');

    _server!.listen((HttpRequest request) async {
      final path = request.uri.path;

      if (!serverlessWasm && path == '/ws') {
        if (WebSocketTransformer.isUpgradeRequest(request)) {
          final socket = await WebSocketTransformer.upgrade(request);
          _handleWebSocket(socket);
        } else {
          request.response.statusCode = HttpStatus.badRequest;
          await request.response.close();
        }
      } else if (!serverlessWasm && path == '/api/export/ipynb') {
        final ipynbJson = IpynbNotebook.fromSessionCells(
          _sessionCells,
        ).toJsonString(pretty: true);
        request.response.headers.set(
          'content-type',
          'application/x-ipynb+json',
        );
        request.response.headers.set(
          'content-disposition',
          'attachment; filename="notebook.ipynb"',
        );
        request.response.write(ipynbJson);
        await request.response.close();
      } else if (!serverlessWasm &&
          path == '/api/import/ipynb' &&
          request.method == 'POST') {
        try {
          final body = await utf8.decoder.bind(request).join();
          final nb = IpynbNotebook.fromJsonString(body);
          _sessionCells = nb.toSessionCells();
          _saveSessionData();
          _broadcastSession();
          request.response.statusCode = HttpStatus.ok;
          request.response.headers.contentType = ContentType.json;
          request.response.write(
            jsonEncode({'status': 'ok', 'cellCount': _sessionCells.length}),
          );
        } catch (e) {
          request.response.statusCode = HttpStatus.badRequest;
          request.response.write(
            jsonEncode({'status': 'error', 'message': '$e'}),
          );
        }
        await request.response.close();
      } else if (path == '/favicon.ico') {
        request.response.statusCode = HttpStatus.noContent;
        await request.response.close();
      } else {
        var requestedPath = request.uri.path;
        if (requestedPath == '/') requestedPath = '/index.html';
        final relativePath = requestedPath.startsWith('/')
            ? requestedPath.substring(1)
            : requestedPath;

        File? targetFile;
        if (staticBundleDir != null) {
          final f = File(p.join(staticBundleDir!, relativePath));
          if (await f.exists()) targetFile = f;
        }
        if (targetFile == null) {
          final webFile = File(p.join(workspaceDir, 'web', relativePath));
          if (await webFile.exists()) {
            targetFile = webFile;
          } else {
            final repoRoot = p.normalize(p.join(workspaceDir, '..', '..'));
            final bundleFile = File(
              p.join(
                repoRoot,
                '.dart_tool',
                'wasm_notebook_bundle',
                relativePath,
              ),
            );
            if (await bundleFile.exists()) {
              targetFile = bundleFile;
            }
          }
        }

        if (targetFile != null) {
          final ext = p.extension(targetFile.path).toLowerCase();
          switch (ext) {
            case '.wasm':
              request.response.headers.set('content-type', 'application/wasm');
              break;
            case '.mjs':
            case '.js':
              request.response.headers.set('content-type', 'text/javascript');
              break;
            case '.json':
              request.response.headers.set('content-type', 'application/json');
              break;
            case '.html':
              request.response.headers.contentType = ContentType.html;
              break;
            case '.css':
              request.response.headers.set('content-type', 'text/css');
              break;
            case '.ipynb':
              request.response.headers.set(
                'content-type',
                'application/x-ipynb+json',
              );
              break;
            default:
              request.response.headers.contentType = ContentType.binary;
          }
          await request.response.addStream(targetFile.openRead());
        } else {
          request.response.statusCode = HttpStatus.notFound;
          request.response.write('404 Not Found');
        }
        await request.response.close();
      }
    });
  }

  void _handleWebSocket(WebSocket socket) {
    print('Client connected to WebSocket.');
    _connectedClients.add(socket);
    socket.add(jsonEncode({'type': 'init_session', 'cells': _sessionCells}));

    socket.listen(
      (data) async {
        try {
          final msg = jsonDecode(data as String) as Map<String, dynamic>;
          final msgType = msg['type'];

          if (msgType == 'save_session') {
            if (msg['cells'] is List) {
              _saveSessionData(msg['cells'] as List);
            }
          } else if (msgType == 'update_cell') {
            final cellId = msg['id'] as String;
            final code = msg['code'] as String?;
            final type =
                msg['cellType'] as String? ?? msg['type_name'] as String?;
            _updateOrAddCell(
              id: cellId,
              code: code,
              type: type,
              evaluated: false,
            );
          } else if (msgType == 'delete_cell') {
            final cellId = msg['id'] as String;
            _sessionCells.removeWhere((c) => c['id'] == cellId);
            _saveSessionData();
          } else if (msgType == 'import_ipynb') {
            final ipynbContent = msg['data'] as String;
            final nb = IpynbNotebook.fromJsonString(ipynbContent);
            _sessionCells = nb.toSessionCells();
            _saveSessionData();
            _broadcastSession();
          } else if (msgType == 'export_ipynb') {
            final ipynbJson = IpynbNotebook.fromSessionCells(
              _sessionCells,
            ).toJsonString(pretty: true);
            socket.add(jsonEncode({'type': 'ipynb_data', 'data': ipynbJson}));
          } else if (msgType == 'execute') {
            final reqId = msg['id'] as String;
            final code = msg['code'] as String;
            final cellId = msg['cellId'] as String? ?? reqId.split('-')[0];

            _updateOrAddCell(
              id: cellId,
              code: code,
              output: 'Running...',
              evaluated: false,
            );

            try {
              final formattedCode = _kernel!.formatCode(code);
              final output = await _kernel!.execute(code);
              List<dynamic> outputsList;
              try {
                outputsList = jsonDecode(output) as List<dynamic>;
              } catch (_) {
                outputsList = [
                  {'mimeType': 'text/plain', 'data': output},
                ];
              }
              _updateOrAddCell(
                id: cellId,
                code: formattedCode,
                output: output,
                isError: false,
                evaluated: true,
              );
              final vars = await _kernel!.getVariableInspectorData();
              socket.add(
                jsonEncode({
                  'type': 'result',
                  'id': reqId,
                  'output': output,
                  'outputs': outputsList,
                  'formattedCode': formattedCode,
                  'isError': false,
                  'variables': vars,
                }),
              );
            } catch (e) {
              final errStr = '$e';
              _updateOrAddCell(
                id: cellId,
                code: code,
                output: errStr,
                isError: true,
                evaluated: false,
              );
              final vars = await _kernel!.getVariableInspectorData();
              socket.add(
                jsonEncode({
                  'type': 'result',
                  'id': reqId,
                  'output': errStr,
                  'outputs': [
                    {'mimeType': 'text/plain', 'data': errStr},
                  ],
                  'isError': true,
                  'variables': vars,
                }),
              );
            }
          } else if (msgType == 'inspect_variables') {
            final vars = await _kernel!.getVariableInspectorData();
            socket.add(
              jsonEncode({'type': 'variables_update', 'variables': vars}),
            );
          } else if (msgType == 'format_code') {
            final cellId = msg['cellId'] as String;
            final code = msg['code'] as String;
            final formatted = _kernel!.formatCode(code);
            socket.add(
              jsonEncode({
                'type': 'format_result',
                'cellId': cellId,
                'formattedCode': formatted,
              }),
            );
          } else if (msgType == 'complete') {
            final reqId = msg['id'] as String;
            final code = msg['code'] as String;
            final offset = (msg['offset'] as num?)?.toInt() ?? code.length;

            final completions = await _kernel!.getCompletions(code, offset);
            socket.add(
              jsonEncode({
                'type': 'completion',
                'id': reqId,
                'completions': completions.map((c) => c.toJson()).toList(),
              }),
            );
          } else if (msgType == 'hover') {
            final reqId = msg['id'] as String;
            final code = msg['code'] as String;
            final offset = (msg['offset'] as num?)?.toInt() ?? code.length;

            final hoverInfo = await _kernel!.getHover(code, offset);
            final hoverHtml = hoverInfo != null
                ? md.markdownToHtml(
                    hoverInfo,
                    extensionSet: md.ExtensionSet.gitHubFlavored,
                  )
                : null;
            socket.add(
              jsonEncode({
                'type': 'hover',
                'id': reqId,
                'hover': hoverInfo,
                'hoverHtml': hoverHtml,
              }),
            );
          }
        } catch (e) {
          print('Error handling WebSocket message: $e');
        }
      },
      onDone: () {
        _connectedClients.remove(socket);
        print('Client disconnected from WebSocket.');
      },
    );
  }

  Future<void> stop() async {
    for (final client in _connectedClients) {
      try {
        await client.close();
      } catch (_) {}
    }
    _connectedClients.clear();
    await _server?.close();
    await _kernel?.stop();
    print('Notebook server stopped.');
  }
}
