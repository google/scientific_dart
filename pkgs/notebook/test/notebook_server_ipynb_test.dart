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
import 'package:notebook/notebook.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

void main() {
  late NotebookServer server;
  String? savedSessionIpynb;
  String? savedSessionJson;

  setUp(() async {
    final workspaceDir = Directory.current.path;
    final ipynbFile = File(p.join(workspaceDir, 'notebook_session.ipynb'));
    final jsonFile = File(p.join(workspaceDir, 'notebook_session.json'));
    savedSessionIpynb = ipynbFile.existsSync()
        ? ipynbFile.readAsStringSync()
        : null;
    savedSessionJson = jsonFile.existsSync()
        ? jsonFile.readAsStringSync()
        : null;

    final sdkPath =
        Platform.environment['DART_SDK'] ??
        p.dirname(p.dirname(Platform.resolvedExecutable));
    server = NotebookServer(
      workspaceDir: workspaceDir,
      dartSdkPath: sdkPath,
      port: 0,
    );
    await server.start();
  });

  tearDown(() async {
    await server.stop();
    final workspaceDir = Directory.current.path;
    final ipynbFile = File(p.join(workspaceDir, 'notebook_session.ipynb'));
    final jsonFile = File(p.join(workspaceDir, 'notebook_session.json'));
    if (savedSessionIpynb != null) {
      ipynbFile.writeAsStringSync(savedSessionIpynb!);
    } else if (ipynbFile.existsSync()) {
      ipynbFile.deleteSync();
    }
    if (savedSessionJson != null) {
      jsonFile.writeAsStringSync(savedSessionJson!);
    } else if (jsonFile.existsSync()) {
      jsonFile.deleteSync();
    }
  });

  test('Exports and imports .ipynb through HTTP endpoints', () async {
    final client = HttpClient();

    // 1. Initial export should return valid empty notebook or default session
    final exportReq = await client.getUrl(
      Uri.parse('http://localhost:${server.actualPort}/api/export/ipynb'),
    );
    final exportRes = await exportReq.close();
    expect(exportRes.statusCode, HttpStatus.ok);
    expect(exportRes.headers.value('content-type'), 'application/x-ipynb+json');

    final exportedBody = await utf8.decoder.bind(exportRes).join();
    final nb = IpynbNotebook.fromJsonString(exportedBody);
    expect(nb.nbformat, 4);
    expect(nb.metadata['kernelspec']['language'], 'dart');

    // 2. Import a multi-cell Jupyter notebook
    final sampleNotebook = IpynbNotebook(
      cells: [
        IpynbCell(
          id: 'test-cell-1',
          cellType: IpynbCellType.markdown,
          source: '# Sample Notebook Title\nDescription of calculation.',
        ),
        IpynbCell(
          id: 'test-cell-2',
          cellType: IpynbCellType.code,
          source: 'final a = 100;\nfinal b = 200;\na + b',
          executionCount: 1,
          outputs: [
            IpynbOutput.executeResult(
              executionCount: 1,
              data: {
                'text/plain': '300',
                'text/html': '<span class="katex">300</span>',
              },
            ),
          ],
        ),
      ],
    );

    final importReq = await client.postUrl(
      Uri.parse('http://localhost:${server.actualPort}/api/import/ipynb'),
    );
    importReq.headers.contentType = ContentType.json;
    importReq.write(sampleNotebook.toJsonString());
    final importRes = await importReq.close();
    expect(importRes.statusCode, HttpStatus.ok);

    final importResBody =
        jsonDecode(await utf8.decoder.bind(importRes).join())
            as Map<String, dynamic>;
    expect(importResBody['status'], 'ok');
    expect(importResBody['cellCount'], 2);

    // 3. Verify session file was written to notebook_session.ipynb
    final sessionFile = File(
      p.join(Directory.current.path, 'notebook_session.ipynb'),
    );
    expect(sessionFile.existsSync(), isTrue);

    final savedNb = IpynbNotebook.fromJsonString(
      sessionFile.readAsStringSync(),
    );
    expect(savedNb.cells.length, 2);
    expect(savedNb.cells[0].cellType, IpynbCellType.markdown);
    expect(savedNb.cells[1].cellType, IpynbCellType.code);
    expect(savedNb.cells[1].outputs.first.data!['text/plain'], '300');

    client.close();
  });
}
