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

/// Immutable LSP 3.17 data models for code editor integration.
library;

class LspPosition implements Comparable<LspPosition> {
  final int line;
  final int character; // 0-based UTF-16 code units offset

  const LspPosition(this.line, this.character);

  @override
  int compareTo(LspPosition other) {
    if (line != other.line) return line.compareTo(other.line);
    return character.compareTo(other.character);
  }

  Map<String, dynamic> toJson() => {'line': line, 'character': character};

  factory LspPosition.fromJson(Map<String, dynamic> json) {
    return LspPosition(json['line'] as int, json['character'] as int);
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is LspPosition &&
          runtimeType == other.runtimeType &&
          line == other.line &&
          character == other.character;

  @override
  int get hashCode => Object.hash(line, character);

  @override
  String toString() => 'LspPosition($line:$character)';
}

class LspRange {
  final LspPosition start;
  final LspPosition end;

  const LspRange(this.start, this.end);

  Map<String, dynamic> toJson() => {
    'start': start.toJson(),
    'end': end.toJson(),
  };

  factory LspRange.fromJson(Map<String, dynamic> json) {
    return LspRange(
      LspPosition.fromJson(json['start'] as Map<String, dynamic>),
      LspPosition.fromJson(json['end'] as Map<String, dynamic>),
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is LspRange &&
          runtimeType == other.runtimeType &&
          start == other.start &&
          end == other.end;

  @override
  int get hashCode => Object.hash(start, end);

  @override
  String toString() => 'LspRange($start -> $end)';
}

class LspTextDocumentIdentifier {
  final String uri;
  const LspTextDocumentIdentifier(this.uri);

  Map<String, dynamic> toJson() => {'uri': uri};
}

class LspVersionedTextDocumentIdentifier extends LspTextDocumentIdentifier {
  final int version;
  const LspVersionedTextDocumentIdentifier(super.uri, this.version);

  @override
  Map<String, dynamic> toJson() => {'uri': uri, 'version': version};
}

class LspTextDocumentContentChangeEvent {
  final LspRange? range;
  final int? rangeLength;
  final String text;

  const LspTextDocumentContentChangeEvent({
    this.range,
    this.rangeLength,
    required this.text,
  });

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{'text': text};
    if (range != null) map['range'] = range!.toJson();
    if (rangeLength != null) map['rangeLength'] = rangeLength;
    return map;
  }
}

class LspCompletionItem {
  final String label;
  final String? insertText;
  final String? detail;
  final String? documentation;
  final int? kind; // CompletionItemKind

  const LspCompletionItem(
    this.label, {
    this.insertText,
    this.detail,
    this.documentation,
    this.kind,
  });
}

class LspHover {
  final String contents;
  final LspRange? range;

  const LspHover(this.contents, {this.range});
}

class LspDiagnostic {
  final LspRange range;
  final String message;
  final int? severity; // 1: Error, 2: Warning, 3: Information, 4: Hint
  final String? code;
  final String? source;

  const LspDiagnostic(
    this.range,
    this.message, {
    this.severity,
    this.code,
    this.source,
  });
}

class LspTextEdit {
  final LspRange range;
  final String newText;

  const LspTextEdit(this.range, this.newText);

  Map<String, dynamic> toJson() => {
    'range': range.toJson(),
    'newText': newText,
  };
}

class LspDocumentSymbol {
  final String name;
  final int kind;
  final LspRange range;
  final LspRange selectionRange;
  final List<LspDocumentSymbol> children;

  const LspDocumentSymbol({
    required this.name,
    required this.kind,
    required this.range,
    required this.selectionRange,
    this.children = const [],
  });
}

class LspCodeAction {
  final String title;
  final String? kind;
  final List<LspTextEdit>? edits;
  final bool isPreferred;

  const LspCodeAction(
    this.title, {
    this.kind,
    this.edits,
    this.isPreferred = false,
  });
}
