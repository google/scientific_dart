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

enum TextAffinity { upstream, downstream }

/// Represents an immutable 0-indexed position within a text document.
class TextPosition implements Comparable<TextPosition> {
  final int line;
  final int column;
  final TextAffinity affinity;

  const TextPosition(
    this.line,
    this.column, {
    this.affinity = TextAffinity.downstream,
  });

  @override
  int compareTo(TextPosition other) {
    if (line != other.line) {
      return line.compareTo(other.line);
    }
    return column.compareTo(other.column);
  }

  /// Returns `true` if this position comes before [other].
  bool operator <(TextPosition other) => compareTo(other) < 0;

  /// Returns `true` if this position comes before or at [other].
  bool operator <=(TextPosition other) => compareTo(other) <= 0;

  /// Returns `true` if this position comes after [other].
  bool operator >(TextPosition other) => compareTo(other) > 0;

  /// Returns `true` if this position comes after or at [other].
  bool operator >=(TextPosition other) => compareTo(other) >= 0;

  TextPosition copyWith({int? line, int? column, TextAffinity? affinity}) {
    return TextPosition(
      line ?? this.line,
      column ?? this.column,
      affinity: affinity ?? this.affinity,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is TextPosition &&
          runtimeType == other.runtimeType &&
          line == other.line &&
          column == other.column &&
          affinity == other.affinity;

  @override
  int get hashCode => Object.hash(line, column, affinity);

  @override
  String toString() => 'TextPosition($line, $column, affinity: $affinity)';
}
