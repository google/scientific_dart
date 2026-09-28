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

import '../buffer/text_buffer.dart';

/// Abstract invertible edit operation.
abstract class EditOperation {
  int get offset;

  /// Applies the operation to [buffer].
  void apply(TextBuffer buffer);

  /// Returns the inverse operation that undoes this change.
  EditOperation invert();
}

/// Insertion edit operation.
class InsertOperation implements EditOperation {
  @override
  final int offset;
  final String text;

  InsertOperation(this.offset, this.text);

  @override
  void apply(TextBuffer buffer) {
    buffer.insert(offset, text);
  }

  @override
  EditOperation invert() {
    return DeleteOperation(offset, text.length, deletedText: text);
  }
}

/// Deletion edit operation.
class DeleteOperation implements EditOperation {
  @override
  final int offset;
  final int length;
  final String deletedText;

  DeleteOperation(this.offset, this.length, {required this.deletedText});

  @override
  void apply(TextBuffer buffer) {
    buffer.delete(offset, length);
  }

  @override
  EditOperation invert() {
    return InsertOperation(offset, deletedText);
  }
}
