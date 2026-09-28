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

enum BufferType { original, add }

/// Buffer storage for original text content.
class OriginalBuffer {
  final String content;
  OriginalBuffer(this.content);

  int get length => content.length;
}

/// Append-only buffer storage for added text insertions.
class AddBuffer {
  final StringBuffer _buffer = StringBuffer();
  String _cachedText = '';

  int get length => _cachedText.length;

  /// Appends [text] to the add buffer and returns the start offset in add buffer.
  int append(String text) {
    final startOffset = _cachedText.length;
    _buffer.write(text);
    _cachedText += text;
    return startOffset;
  }

  /// Returns the text content starting at [start] of [length].
  String getText(int start, int length) {
    return _cachedText.substring(start, start + length);
  }

  /// Returns full text content.
  String get content => _cachedText;
}
