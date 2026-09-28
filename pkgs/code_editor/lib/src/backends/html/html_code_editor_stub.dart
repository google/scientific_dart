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

import '../../editor/code_editor_controller.dart';

/// Stub implementation for non-web environments.
final class HtmlCodeEditor {
  final Object hostElement;
  final CodeEditorController controller;

  void Function()? onExecute;
  void Function(int offset, int clientX, int clientY)? onCompletionRequested;
  void Function(int offset, int clientX, int clientY)? onHoverRequested;

  HtmlCodeEditor({
    required this.hostElement,
    required this.controller,
    this.onExecute,
    this.onCompletionRequested,
    this.onHoverRequested,
  });

  Object get rootElement => Object();

  void focus() {}
  void render() {}
}
