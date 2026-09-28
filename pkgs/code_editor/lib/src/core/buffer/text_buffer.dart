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

/// Abstract contract for text buffer operations, line indexing, substring
/// extraction, and string mutation.
abstract class TextBuffer {
  /// Total length of the text in UTF-16 code units.
  int get length;

  /// Total number of lines in the buffer (1-indexed count, minimum 1).
  int get lineCount;

  /// Full text content of the buffer.
  String get text;

  /// Returns a substring from [startOffset] of [length] UTF-16 code units.
  String getTextInRange(int startOffset, int length);

  /// Returns the text content of line [lineIndex] (0-based index),
  /// including any trailing newline character(s).
  String getLine(int lineIndex);

  /// Returns the starting character offset (UTF-16 code unit offset) for [lineIndex] (0-based).
  int getLineOffset(int lineIndex);

  /// Returns the length of [lineIndex] in UTF-16 code units (including line breaks).
  int getLineLength(int lineIndex);

  /// Returns the line and column (0-based line, 0-based code unit column) for [offset].
  (int line, int column) getLineAndColumnAt(int offset);

  /// Returns the UTF-16 code unit offset for position ([line], [column]).
  int getOffsetAt(int line, int column);

  /// Inserts [text] at [offset].
  void insert(int offset, String text);

  /// Deletes [length] UTF-16 code units starting at [offset].
  void delete(int offset, int length);
}
