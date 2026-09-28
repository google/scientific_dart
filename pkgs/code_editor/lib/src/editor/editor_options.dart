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

import 'package:meta/meta.dart';

/// Configuration options for the code editor.
@immutable
final class EditorOptions {
  /// Font family name (e.g. `'Fira Code'`, `'JetBrains Mono'`).
  final String fontFamily;

  /// Font size in CSS/logical pixels.
  final double fontSize;

  /// Line height in CSS/logical pixels.
  final double lineHeight;

  /// Tab size in spaces (default: 2).
  final int tabSize;

  /// Whether soft line wrapping is enabled.
  final bool wordWrap;

  /// Whether line numbers gutter is visible.
  final bool lineNumbers;

  /// Whether code folding markers are enabled.
  final bool folding;

  /// Whether auto-closing brackets and quotes are enabled.
  final bool autoClosingBrackets;

  /// Whether smart indentation on Enter is enabled.
  final bool smartIndent;

  /// Whether bracket pair matching highlight is enabled.
  final bool matchBrackets;

  /// Whether active line is highlighted.
  final bool highlightActiveLine;

  /// Whether the editor is in read-only mode.
  final bool readOnly;

  /// Creates an [EditorOptions] configuration.
  const EditorOptions({
    this.fontFamily = "'Fira Code', 'JetBrains Mono', 'Consolas', monospace",
    this.fontSize = 14.0,
    this.lineHeight = 24.0,
    this.tabSize = 2,
    this.wordWrap = false,
    this.lineNumbers = true,
    this.folding = true,
    this.autoClosingBrackets = true,
    this.smartIndent = true,
    this.matchBrackets = true,
    this.highlightActiveLine = true,
    this.readOnly = false,
  });

  /// Copies this options object with updated fields.
  EditorOptions copyWith({
    String? fontFamily,
    double? fontSize,
    double? lineHeight,
    int? tabSize,
    bool? wordWrap,
    bool? lineNumbers,
    bool? folding,
    bool? autoClosingBrackets,
    bool? smartIndent,
    bool? matchBrackets,
    bool? highlightActiveLine,
    bool? readOnly,
  }) {
    return EditorOptions(
      fontFamily: fontFamily ?? this.fontFamily,
      fontSize: fontSize ?? this.fontSize,
      lineHeight: lineHeight ?? this.lineHeight,
      tabSize: tabSize ?? this.tabSize,
      wordWrap: wordWrap ?? this.wordWrap,
      lineNumbers: lineNumbers ?? this.lineNumbers,
      folding: folding ?? this.folding,
      autoClosingBrackets: autoClosingBrackets ?? this.autoClosingBrackets,
      smartIndent: smartIndent ?? this.smartIndent,
      matchBrackets: matchBrackets ?? this.matchBrackets,
      highlightActiveLine: highlightActiveLine ?? this.highlightActiveLine,
      readOnly: readOnly ?? this.readOnly,
    );
  }
}
