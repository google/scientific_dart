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

/// Represents the cell type in a Jupyter Notebook.
enum IpynbCellType {
  code('code'),
  markdown('markdown'),
  raw('raw');

  final String value;
  const IpynbCellType(this.value);

  static IpynbCellType fromString(String val) {
    switch (val) {
      case 'code':
        return IpynbCellType.code;
      case 'markdown':
        return IpynbCellType.markdown;
      case 'raw':
        return IpynbCellType.raw;
      default:
        return IpynbCellType.code;
    }
  }
}

/// Represents the output type in a Jupyter Notebook code cell.
enum IpynbOutputType {
  stream('stream'),
  executeResult('execute_result'),
  displayData('display_data'),
  error('error');

  final String value;
  const IpynbOutputType(this.value);

  static IpynbOutputType fromString(String val) {
    switch (val) {
      case 'stream':
        return IpynbOutputType.stream;
      case 'execute_result':
        return IpynbOutputType.executeResult;
      case 'display_data':
        return IpynbOutputType.displayData;
      case 'error':
        return IpynbOutputType.error;
      default:
        return IpynbOutputType.displayData;
    }
  }
}

/// An individual output element within a Jupyter Notebook code cell.
final class IpynbOutput {
  final IpynbOutputType outputType;
  final String? name; // 'stdout' or 'stderr' for stream
  final String? text; // for stream
  final Map<String, dynamic>?
  data; // MIME bundle for execute_result / display_data
  final Map<String, dynamic>? metadata;
  final int? executionCount;
  final String? ename;
  final String? evalue;
  final List<String>? traceback;

  IpynbOutput({
    required this.outputType,
    this.name,
    this.text,
    this.data,
    this.metadata,
    this.executionCount,
    this.ename,
    this.evalue,
    this.traceback,
  });

  factory IpynbOutput.stream({required String text, String name = 'stdout'}) {
    return IpynbOutput(
      outputType: IpynbOutputType.stream,
      name: name,
      text: text,
    );
  }

  factory IpynbOutput.displayData({
    required Map<String, dynamic> data,
    Map<String, dynamic>? metadata,
  }) {
    return IpynbOutput(
      outputType: IpynbOutputType.displayData,
      data: data,
      metadata: metadata ?? {},
    );
  }

  factory IpynbOutput.executeResult({
    required Map<String, dynamic> data,
    int? executionCount,
    Map<String, dynamic>? metadata,
  }) {
    return IpynbOutput(
      outputType: IpynbOutputType.executeResult,
      data: data,
      executionCount: executionCount,
      metadata: metadata ?? {},
    );
  }

  factory IpynbOutput.error({
    required String ename,
    required String evalue,
    List<String>? traceback,
  }) {
    return IpynbOutput(
      outputType: IpynbOutputType.error,
      ename: ename,
      evalue: evalue,
      traceback: traceback ?? [evalue],
    );
  }

  factory IpynbOutput.fromJson(Map<String, dynamic> json) {
    final outputType = IpynbOutputType.fromString(
      json['output_type'] as String? ?? 'display_data',
    );
    final name = json['name'] as String?;
    final textRaw = json['text'];
    final text = textRaw is List ? textRaw.join('') : textRaw as String?;
    final dataRaw = json['data'] as Map<String, dynamic>?;
    Map<String, dynamic>? normalizedData;
    if (dataRaw != null) {
      normalizedData = {};
      dataRaw.forEach((k, v) {
        if (v is List) {
          normalizedData![k] = v.join('');
        } else {
          normalizedData![k] = v;
        }
      });
    }
    final metadata = json['metadata'] as Map<String, dynamic>?;
    final executionCount = json['execution_count'] as int?;
    final ename = json['ename'] as String?;
    final evalue = json['evalue'] as String?;
    final tracebackRaw = json['traceback'] as List?;
    final traceback = tracebackRaw?.map((e) => e.toString()).toList();

    return IpynbOutput(
      outputType: outputType,
      name: name,
      text: text,
      data: normalizedData,
      metadata: metadata,
      executionCount: executionCount,
      ename: ename,
      evalue: evalue,
      traceback: traceback,
    );
  }

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{'output_type': outputType.value};

    switch (outputType) {
      case IpynbOutputType.stream:
        map['name'] = name ?? 'stdout';
        map['text'] = _splitLines(text ?? '');
        break;
      case IpynbOutputType.displayData:
        map['data'] = _formatMimeBundle(data ?? {});
        map['metadata'] = metadata ?? {};
        break;
      case IpynbOutputType.executeResult:
        map['execution_count'] = executionCount;
        map['data'] = _formatMimeBundle(data ?? {});
        map['metadata'] = metadata ?? {};
        break;
      case IpynbOutputType.error:
        map['ename'] = ename ?? 'Error';
        map['evalue'] = evalue ?? '';
        map['traceback'] = traceback ?? [evalue ?? ''];
        break;
    }

    return map;
  }
}

/// An individual cell within a Jupyter Notebook.
final class IpynbCell {
  final String id;
  final IpynbCellType cellType;
  final String source;
  final int? executionCount;
  final List<IpynbOutput> outputs;
  final Map<String, dynamic> metadata;

  IpynbCell({
    required this.id,
    required this.cellType,
    required this.source,
    this.executionCount,
    this.outputs = const [],
    this.metadata = const {},
  });

  factory IpynbCell.fromJson(
    Map<String, dynamic> json, [
    int fallbackIndex = 0,
  ]) {
    final cellType = IpynbCellType.fromString(
      json['cell_type'] as String? ?? 'code',
    );
    final id = json['id'] as String? ?? 'cell-$fallbackIndex';
    final sourceRaw = json['source'];
    final source = sourceRaw is List
        ? sourceRaw.join('')
        : (sourceRaw as String? ?? '');
    final executionCount = json['execution_count'] as int?;
    final metadata = Map<String, dynamic>.from(json['metadata'] as Map? ?? {});

    final outputsRaw = json['outputs'] as List? ?? [];
    final outputs = outputsRaw
        .map((e) => IpynbOutput.fromJson(Map<String, dynamic>.from(e as Map)))
        .toList();

    return IpynbCell(
      id: id,
      cellType: cellType,
      source: source,
      executionCount: executionCount,
      outputs: outputs,
      metadata: metadata,
    );
  }

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{
      'id': id,
      'cell_type': cellType.value,
      'metadata': metadata,
      'source': _splitLines(source),
    };

    if (cellType == IpynbCellType.code) {
      map['execution_count'] = executionCount;
      map['outputs'] = outputs.map((o) => o.toJson()).toList();
    }

    return map;
  }
}

/// Represents a full Jupyter Notebook document (v4 format).
final class IpynbNotebook {
  final List<IpynbCell> cells;
  final Map<String, dynamic> metadata;
  final int nbformat;
  final int nbformatMinor;

  IpynbNotebook({
    required this.cells,
    Map<String, dynamic>? metadata,
    this.nbformat = 4,
    this.nbformatMinor = 5,
  }) : metadata = metadata ?? _defaultMetadata();

  static Map<String, dynamic> _defaultMetadata({
    String dartVersion = '3.13.1',
  }) {
    return {
      'kernelspec': {
        'display_name': 'Dart',
        'language': 'dart',
        'name': 'dart',
      },
      'language_info': {
        'name': 'dart',
        'version': dartVersion,
        'mimetype': 'text/x-dart',
        'file_extension': '.dart',
      },
    };
  }

  factory IpynbNotebook.fromJson(Map<String, dynamic> json) {
    final nbformat = json['nbformat'] as int? ?? 4;
    final nbformatMinor = json['nbformat_minor'] as int? ?? 5;
    final metadata = Map<String, dynamic>.from(json['metadata'] as Map? ?? {});
    final cellsRaw = json['cells'] as List? ?? [];

    var index = 0;
    final cells = cellsRaw.map((e) {
      index++;
      return IpynbCell.fromJson(Map<String, dynamic>.from(e as Map), index);
    }).toList();

    return IpynbNotebook(
      cells: cells,
      metadata: metadata,
      nbformat: nbformat,
      nbformatMinor: nbformatMinor,
    );
  }

  factory IpynbNotebook.fromJsonString(String jsonString) {
    final decoded = jsonDecode(jsonString) as Map<String, dynamic>;
    return IpynbNotebook.fromJson(decoded);
  }

  Map<String, dynamic> toJson() {
    return {
      'nbformat': nbformat,
      'nbformat_minor': nbformatMinor,
      'metadata': metadata,
      'cells': cells.map((c) => c.toJson()).toList(),
    };
  }

  String toJsonString({bool pretty = true}) {
    if (pretty) {
      return const JsonEncoder.withIndent('  ').convert(toJson());
    }
    return jsonEncode(toJson());
  }

  /// Converts a legacy session cells list (from notebook_server.dart / web UI) to an [IpynbNotebook].
  factory IpynbNotebook.fromSessionCells(
    List<Map<String, dynamic>> sessionCells,
  ) {
    final cells = <IpynbCell>[];
    var execIndex = 1;

    for (final s in sessionCells) {
      final id = s['id']?.toString() ?? 'cell-${cells.length + 1}';
      final typeStr = s['type']?.toString() ?? 'code';
      final cellType = IpynbCellType.fromString(typeStr);
      final code = s['code']?.toString() ?? '';
      final outputRaw = s['output'];
      final outputsRawList = s['outputs'];
      final isError = s['isError'] == true;
      final evaluated = s['evaluated'] == true;

      final outputs = <IpynbOutput>[];

      if (cellType == IpynbCellType.code &&
          (evaluated ||
              isError ||
              (outputsRawList is List && outputsRawList.isNotEmpty) ||
              (outputRaw != null && outputRaw.toString().isNotEmpty))) {
        if (isError) {
          outputs.add(
            IpynbOutput.error(
              ename: 'ExecutionError',
              evalue: outputRaw?.toString() ?? 'Error',
            ),
          );
        } else if (outputsRawList is List && outputsRawList.isNotEmpty) {
          for (final item in outputsRawList) {
            if (item is Map) {
              final mime = item['mimeType']?.toString() ?? 'text/plain';
              final data = item['data']?.toString() ?? '';
              if (mime == 'text/html') {
                outputs.add(
                  IpynbOutput.displayData(
                    data: {
                      'text/html': data,
                      'text/plain': outputRaw?.toString() ?? '',
                    },
                  ),
                );
              } else {
                outputs.add(
                  IpynbOutput.executeResult(
                    executionCount: execIndex,
                    data: {'text/plain': data},
                  ),
                );
              }
            }
          }
        } else if (outputRaw != null && outputRaw.toString().isNotEmpty) {
          final outStr = outputRaw.toString();
          // Check if output is a JSON array of CellOutputItem
          List<dynamic>? parsedList;
          try {
            if (outStr.startsWith('[')) {
              parsedList = jsonDecode(outStr) as List<dynamic>?;
            }
          } catch (_) {}

          if (parsedList != null && parsedList.isNotEmpty) {
            for (final item in parsedList) {
              if (item is Map) {
                final mime = item['mimeType']?.toString() ?? 'text/plain';
                final data = item['data']?.toString() ?? '';
                if (mime == 'text/html') {
                  outputs.add(
                    IpynbOutput.displayData(
                      data: {'text/html': data, 'text/plain': data},
                    ),
                  );
                } else {
                  outputs.add(
                    IpynbOutput.executeResult(
                      executionCount: execIndex,
                      data: {'text/plain': data},
                    ),
                  );
                }
              }
            }
          } else {
            outputs.add(
              IpynbOutput.executeResult(
                executionCount: execIndex,
                data: {'text/plain': outStr},
              ),
            );
          }
        }
      }

      cells.add(
        IpynbCell(
          id: id,
          cellType: cellType,
          source: code,
          executionCount: evaluated && cellType == IpynbCellType.code
              ? execIndex++
              : null,
          outputs: outputs,
        ),
      );
    }

    return IpynbNotebook(cells: cells);
  }

  /// Converts this [IpynbNotebook] into session cells compatible with the web UI and server.
  List<Map<String, dynamic>> toSessionCells() {
    final list = <Map<String, dynamic>>[];

    for (final cell in cells) {
      var outputText = '';
      var isError = false;
      final outputsList = <Map<String, dynamic>>[];

      for (final out in cell.outputs) {
        if (out.outputType == IpynbOutputType.error) {
          isError = true;
          outputText = out.evalue ?? out.traceback?.join('\n') ?? 'Error';
        } else if (out.outputType == IpynbOutputType.stream) {
          outputText += (out.text ?? '');
        } else if (out.data != null) {
          final html = out.data!['text/html'];
          final plain = out.data!['text/plain'];
          if (html != null) {
            final htmlStr = html is List ? html.join('') : html.toString();
            outputsList.add({'mimeType': 'text/html', 'data': htmlStr});
            if (outputText.isEmpty) outputText = plain?.toString() ?? htmlStr;
          } else if (plain != null) {
            final plainStr = plain is List ? plain.join('') : plain.toString();
            outputsList.add({'mimeType': 'text/plain', 'data': plainStr});
            if (outputText.isEmpty) outputText = plainStr;
          }
        }
      }

      list.add({
        'id': cell.id,
        'type': cell.cellType.value,
        'code': cell.source,
        'output': outputText,
        'outputs': outputsList,
        'isError': isError,
        'evaluated': cell.outputs.isNotEmpty || cell.executionCount != null,
      });
    }

    return list;
  }
}

/// Helper to split text into lines preserving newline characters per the Jupyter nbformat specification.
List<String> _splitLines(String text) {
  if (text.isEmpty) return [];
  final lines = <String>[];
  var start = 0;
  for (var i = 0; i < text.length; i++) {
    if (text[i] == '\n') {
      lines.add(text.substring(start, i + 1));
      start = i + 1;
    }
  }
  if (start < text.length) {
    lines.add(text.substring(start));
  }
  return lines;
}

/// Helper to format MIME bundle items as strings or multi-line string arrays.
Map<String, dynamic> _formatMimeBundle(Map<String, dynamic> data) {
  final result = <String, dynamic>{};
  data.forEach((mime, val) {
    if (val is String) {
      result[mime] = _splitLines(val);
    } else if (val is List) {
      result[mime] = val;
    } else {
      result[mime] = val;
    }
  });
  return result;
}
