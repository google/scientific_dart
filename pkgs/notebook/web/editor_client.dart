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

import 'dart:js_interop';
import 'package:web/web.dart' as web;
import 'package:code_editor/code_editor.dart';
import 'package:code_editor/backends/html.dart';

@JS('globalThis')
external JSObject get globalThis;

@JS('runCell')
external void _runCellJS(JSString cellId);

@JS('requestCompletionCm')
external void _requestCompletion(JSObject cm, JSString cellId);

extension type _GlobalScope(JSObject _) implements JSObject {
  // ignore: non_constant_identifier_names
  external set DartEditor(JSFunction fn);
}

extension type _CursorPos(JSObject _) implements JSObject {
  external int? get line;
  external int? get ch;
  external set line(int? val);
  external set ch(int? val);
}

extension type _ChangeObj(JSObject _) implements JSObject {
  external String? get origin;
  external set origin(String? val);
}

void main() {
  JSObject createEditor(web.HTMLElement element, [JSObject? options]) {
    try {
      final web.HTMLTextAreaElement? textarea;
      final web.HTMLElement parent;
      final String rawId;
      final String initialCode;

      if (element.isA<web.HTMLTextAreaElement>()) {
        textarea = element as web.HTMLTextAreaElement;
        rawId = textarea.id;
        initialCode = textarea.value;
        parent = (textarea.parentElement as web.HTMLElement?) ?? textarea;
        textarea.style.display = 'none';
      } else {
        rawId = element.id;
        final existingTextarea = element.querySelector('textarea');
        if (existingTextarea != null &&
            existingTextarea.isA<web.HTMLTextAreaElement>()) {
          textarea = existingTextarea as web.HTMLTextAreaElement;
          initialCode = textarea.value;
          textarea.style.display = 'none';
        } else {
          textarea = null;
          initialCode = element.textContent ?? '';
        }
        parent = element;
      }

      final cellId = rawId.startsWith('editor-')
          ? rawId.substring('editor-'.length)
          : rawId;

      // Clean up any existing mount container in parent
      final existingMount = parent.querySelector(
        '.dart-editor-mount-container',
      );
      if (existingMount != null) {
        existingMount.remove();
      }

      // Mount container
      final mountContainer =
          web.document.createElement('div') as web.HTMLElement;
      mountContainer.className = 'dart-editor-mount-container';
      mountContainer.style.width = '100%';
      mountContainer.style.margin = '4px 0';
      parent.appendChild(mountContainer);

      final controller = CodeEditorController(
        initialText: initialCode,
        options: const EditorOptions(
          fontFamily: "'Fira Code', 'JetBrains Mono', 'Consolas', monospace",
          fontSize: 14.0,
          lineHeight: 24.0,
          tabSize: 2,
          autoClosingBrackets: true,
          smartIndent: true,
          matchBrackets: true,
          highlightActiveLine: true,
        ),
      );

      late final _DartEditorBridge bridge;

      final editor = HtmlCodeEditor(
        hostElement: mountContainer,
        controller: controller,
        onExecute: () {
          _runCellJS(cellId.toJS);
        },
        onCompletionRequested: (offset, x, y) {
          _requestCompletion(bridge, cellId.toJS);
        },
      );

      final changeCallbacks = <JSFunction>[];

      controller.addListener(() {
        if (textarea != null) {
          textarea.value = controller.text;
        }
        for (final cb in changeCallbacks) {
          try {
            final changeObj = _ChangeObj(JSObject());
            changeObj.origin = '+input';
            cb.callAsFunction(bridge, bridge, changeObj);
          } catch (_) {}
        }
      });

      JSString getValue() => controller.text.toJS;
      void setValue(JSString newCode) {
        controller.text = newCode.toDart;
      }

      void focus() {
        editor.focus();
      }

      void formatCode() {
        controller.formatDocument();
      }

      void setDiagnostic(
        JSNumber line,
        JSNumber startColumn,
        JSNumber endColumn,
        JSString message,
      ) {
        controller.setDiagnostics([
          DiagnosticSquiggle(
            line: line.toDartInt,
            startColumn: startColumn.toDartInt,
            endColumn: endColumn.toDartInt,
            severity: DiagnosticSeverity.error,
            message: message.toDart,
          ),
        ]);
      }

      void clearDiagnostics() {
        controller.clearDiagnostics();
      }

      JSObject getCursor() {
        final pos = controller.selection.extent;
        final obj = _CursorPos(JSObject());
        obj.line = pos.line;
        obj.ch = pos.column;
        return obj;
      }

      void setCursor(JSObject cursorObj) {
        final cur = _CursorPos(cursorObj);
        final lineVal = cur.line;
        final chVal = cur.ch;
        if (lineVal != null && chVal != null) {
          controller.selection = TextSelection.collapsed(
            TextPosition(lineVal, chVal),
          );
        }
      }

      JSNumber indexFromPos(JSObject posObj) {
        final cur = _CursorPos(posObj);
        final lineVal = cur.line;
        final chVal = cur.ch;
        if (lineVal != null && chVal != null) {
          final offset = controller.buffer.getOffsetAt(lineVal, chVal);
          return offset.toJS;
        }
        return 0.toJS;
      }

      JSString getLine(JSNumber lineNum) {
        final line = lineNum.toDartInt;
        if (line >= 0 && line < controller.lineCount) {
          return controller.buffer.getLine(line).toJS;
        }
        return ''.toJS;
      }

      void replaceRange(
        JSString replacement,
        JSObject fromPos, [
        JSObject? toPos,
      ]) {
        final fromCur = _CursorPos(fromPos);
        final fromLine = fromCur.line ?? 0;
        final fromCh = fromCur.ch ?? 0;

        final toCur = toPos != null ? _CursorPos(toPos) : null;
        final toLine = toCur?.line ?? fromLine;
        final toCh = toCur?.ch ?? fromCh;

        final fromOffset = controller.buffer.getOffsetAt(fromLine, fromCh);
        final toOffset = controller.buffer.getOffsetAt(toLine, toCh);
        final fullText = controller.text;
        final clampedStart = fromOffset.clamp(0, fullText.length);
        final clampedEnd = toOffset.clamp(clampedStart, fullText.length);
        final updated =
            fullText.substring(0, clampedStart) +
            replacement.toDart +
            fullText.substring(clampedEnd);
        controller.text = updated;
      }

      void on(JSString eventName, JSFunction callback) {
        if (eventName.toDart == 'change') {
          changeCallbacks.add(callback);
        }
      }

      JSObject getWrapperElement() => editor.rootElement as JSObject;

      bridge = _DartEditorBridge._(JSObject());
      bridge.getValue = getValue.toJS;
      bridge.setValue = setValue.toJS;
      bridge.focus = focus.toJS;
      bridge.formatCode = formatCode.toJS;
      bridge.setDiagnostic = setDiagnostic.toJS;
      bridge.clearDiagnostics = clearDiagnostics.toJS;
      bridge.getCursor = getCursor.toJS;
      bridge.setCursor = setCursor.toJS;
      bridge.indexFromPos = indexFromPos.toJS;
      bridge.getLine = getLine.toJS;
      bridge.replaceRange = replaceRange.toJS;
      bridge.on = on.toJS;
      bridge.getWrapperElement = getWrapperElement.toJS;

      return bridge;
    } catch (e, st) {
      print('Error in DartEditor initialization: $e\n$st');
      rethrow;
    }
  }

  final fn = createEditor.toJS;
  _GlobalScope(globalThis).DartEditor = fn;
}

@JS()
@anonymous
extension type _DartEditorBridge._(JSObject _) implements JSObject {
  external set getValue(JSFunction fn);
  external set setValue(JSFunction fn);
  external set focus(JSFunction fn);
  external set formatCode(JSFunction fn);
  external set setDiagnostic(JSFunction fn);
  external set clearDiagnostics(JSFunction fn);
  external set getCursor(JSFunction fn);
  external set setCursor(JSFunction fn);
  external set indexFromPos(JSFunction fn);
  external set getLine(JSFunction fn);
  external set replaceRange(JSFunction fn);
  external set on(JSFunction fn);
  external set getWrapperElement(JSFunction fn);
}
