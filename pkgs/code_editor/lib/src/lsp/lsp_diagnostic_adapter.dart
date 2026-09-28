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

/// LSP 3.17 Diagnostic Severity enumeration and visual Squiggle geometry model.
library;

/// Diagnostic severity levels corresponding to LSP 3.17 diagnostic severity numbers (1-4).
enum DiagnosticSeverity {
  /// Error diagnostic severity (1).
  error,

  /// Warning diagnostic severity (2).
  warning,

  /// Information diagnostic severity (3).
  info,

  /// Hint diagnostic severity (4).
  hint,
}

/// Visual color representations associated with diagnostic squiggles.
enum SquiggleColor {
  /// Red color for error squiggles.
  red,

  /// Yellow color for warning squiggles.
  yellow,

  /// Blue color for info squiggles.
  blue,

  /// Gray color for hint squiggles.
  gray,
}

/// Represents the visual rendering geometry and severity of an LSP diagnostic squiggle.
final class DiagnosticSquiggle {
  /// Zero-indexed line number where the diagnostic squiggle starts.
  final int line;

  /// Zero-indexed starting column index on [line].
  final int startColumn;

  /// Zero-indexed ending column index on [line].
  final int endColumn;

  /// Severity level of the diagnostic.
  final DiagnosticSeverity severity;

  /// Description message of the diagnostic.
  final String message;

  /// Optional source identifier (e.g. 'dart', 'eslint').
  final String? source;

  /// Optional rule or error code identifier.
  final String? code;

  /// Creates a new [DiagnosticSquiggle] geometry model.
  const DiagnosticSquiggle({
    required this.line,
    required this.startColumn,
    required this.endColumn,
    required this.severity,
    required this.message,
    this.source,
    this.code,
  });

  /// The visual color associated with this diagnostic's severity level.
  SquiggleColor get color {
    switch (severity) {
      case DiagnosticSeverity.error:
        return SquiggleColor.red;
      case DiagnosticSeverity.warning:
        return SquiggleColor.yellow;
      case DiagnosticSeverity.info:
        return SquiggleColor.blue;
      case DiagnosticSeverity.hint:
        return SquiggleColor.gray;
    }
  }

  /// Hex color string corresponding to the severity color representation.
  String get colorHex {
    switch (severity) {
      case DiagnosticSeverity.error:
        return '#FF0000';
      case DiagnosticSeverity.warning:
        return '#FFFF00';
      case DiagnosticSeverity.info:
        return '#0000FF';
      case DiagnosticSeverity.hint:
        return '#808080';
    }
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is DiagnosticSquiggle &&
          runtimeType == other.runtimeType &&
          line == other.line &&
          startColumn == other.startColumn &&
          endColumn == other.endColumn &&
          severity == other.severity &&
          message == other.message &&
          source == other.source &&
          code == other.code;

  @override
  int get hashCode => Object.hash(
    line,
    startColumn,
    endColumn,
    severity,
    message,
    source,
    code,
  );
}

/// Adapter utility to translate raw LSP diagnostic maps into [DiagnosticSquiggle] objects.
final class LspDiagnosticAdapter {
  /// Converts raw LSP `Diagnostic` maps into [DiagnosticSquiggle] geometries.
  static List<DiagnosticSquiggle> fromLspDiagnostics(
    List<dynamic> diagnostics, {
    List<String>? documentLines,
  }) {
    final squiggles = <DiagnosticSquiggle>[];

    for (final rawDiag in diagnostics) {
      if (rawDiag is! Map) continue;
      final Map<String, dynamic> diag = Map<String, dynamic>.from(rawDiag);
      final range = diag['range'] as Map?;
      if (range == null) continue;

      final start = range['start'] as Map?;
      final end = range['end'] as Map?;
      if (start == null || end == null) continue;

      final startLine = (start['line'] as num?)?.toInt() ?? 0;
      final startCol = (start['character'] as num?)?.toInt() ?? 0;
      final endLine = (end['line'] as num?)?.toInt() ?? startLine;
      final endCol = (end['character'] as num?)?.toInt() ?? startCol;

      final severityInt = (diag['severity'] as num?)?.toInt() ?? 1;
      final severity = switch (severityInt) {
        1 => DiagnosticSeverity.error,
        2 => DiagnosticSeverity.warning,
        3 => DiagnosticSeverity.info,
        4 => DiagnosticSeverity.hint,
        _ => DiagnosticSeverity.error,
      };

      final message = diag['message'] as String? ?? '';
      final source = diag['source'] as String?;
      final code = diag['code']?.toString();

      for (var l = startLine; l <= endLine; l++) {
        final colStart = (l == startLine) ? startCol : 0;
        final maxLineLen = (documentLines != null && l < documentLines.length)
            ? documentLines[l].length
            : 999;
        final colEnd = (l == endLine) ? endCol : maxLineLen;

        squiggles.add(
          DiagnosticSquiggle(
            line: l,
            startColumn: colStart,
            endColumn: colEnd,
            severity: severity,
            message: message,
            source: source,
            code: code,
          ),
        );
      }
    }
    return squiggles;
  }
}
