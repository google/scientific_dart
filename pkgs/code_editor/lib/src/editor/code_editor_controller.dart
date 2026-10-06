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

import 'dart:math' as math;
import '../core/buffer/piece_tree.dart';
import '../core/buffer/text_buffer.dart';
import '../core/history/edit_operation.dart';
import '../core/history/editor_transaction.dart';
import '../core/history/undo_manager.dart';
import '../core/selection/selection_model.dart';
import '../features/bracket_matching/bracket_matcher.dart';
import '../features/find_replace/find_replace_controller.dart';
import '../features/find_replace/search_match.dart';
import '../features/formatting/dart_formatter_engine.dart';
import '../features/smart_editing/auto_close_pairs.dart';
import '../features/smart_editing/line_operations.dart';
import '../features/smart_editing/smart_indent_engine.dart';
import '../features/snippets/snippet_engine.dart';
import '../lsp/lsp_diagnostic_adapter.dart';
import '../lsp/lsp_sync_manager.dart';
import '../syntax/color_theme.dart';
import '../syntax/grammars/dart_grammar.dart';
import '../syntax/incremental_tokenizer.dart';
import '../syntax/syntax_token.dart';
import '../viewport/folding_manager.dart';
import 'editor_options.dart';

/// Cursor navigation direction movement types.
enum CursorMovement {
  left,
  right,
  up,
  down,
  wordLeft,
  wordRight,
  lineStart,
  lineEnd,
  pageUp,
  pageDown,
  documentStart,
  documentEnd,
}

/// Central controller orchestrating document buffers, undo/redo transactions,
/// multi-cursor selections, incremental syntax tokenization, smart editing, and search.
final class CodeEditorController {
  final TextBuffer _buffer;
  final SelectionModel _selectionModel;
  final UndoManager _undoManager;
  final IncrementalTokenizer _tokenizer;
  final BracketMatcher _bracketMatcher;
  final AutoCloseEngine _autoCloseEngine;
  final SmartIndentEngine _smartIndentEngine;
  final LineOperations _lineOperations;
  final FindReplaceController _findReplace;
  final SnippetEngine _snippetEngine;
  final FoldingManager _foldingManager;
  final DartFormatterEngine _formatter;
  final LspDocumentSyncManager _lspSync;

  ExpandedSnippetResult? _activeSnippet;
  int _activeTabStopIndex = 0;
  final List<DiagnosticSquiggle> _diagnostics = [];

  EditorOptions _options;
  final List<void Function()> _changeListeners = [];

  /// Creates a [CodeEditorController] with initial text and configuration.
  CodeEditorController({
    String initialText = '',
    EditorOptions options = const EditorOptions(),
    IncrementalTokenizer? tokenizer,
    ColorTheme? theme,
    AutoCloseEngine? autoCloseEngine,
    SmartIndentEngine? smartIndentEngine,
    BracketMatcher? bracketMatcher,
    LineOperations? lineOperations,
    SnippetEngine? snippetEngine,
    FoldingManager? foldingManager,
    DartFormatterEngine? formatter,
    String documentUri = 'untitled:document.dart',
  }) : _buffer = PieceTreeTextBuffer(initialText),
       _options = options,
       _selectionModel = SelectionModel(),
       _undoManager = UndoManager(),
       _tokenizer =
           tokenizer ??
           IncrementalTokenizer(
             tokenizer: DartGrammar.createLexer(),
             styleCache: StyleCache(theme ?? ColorTheme.catppuccinMocha()),
           ),
       _autoCloseEngine = autoCloseEngine ?? const AutoCloseEngine(),
       _smartIndentEngine =
           smartIndentEngine ?? SmartIndentEngine(tabSize: options.tabSize),
       _bracketMatcher = bracketMatcher ?? const BracketMatcher(),
       _lineOperations = lineOperations ?? const LineOperations(),
       _findReplace = FindReplaceController(),
       _snippetEngine = snippetEngine ?? SnippetEngine(),
       _foldingManager = foldingManager ?? FoldingManager(),
       _formatter = formatter ?? DartFormatterEngine(tabSize: options.tabSize),
       _lspSync = LspDocumentSyncManager(uri: documentUri) {
    _tokenizeAll();
  }

  /// The underlying text buffer.
  TextBuffer get buffer => _buffer;

  /// Current full text string of the document.
  String get text => _buffer.text;

  /// Active diagnostic squiggles.
  List<DiagnosticSquiggle> get diagnostics => List.unmodifiable(_diagnostics);

  /// Sets the list of active diagnostic squiggles.
  void setDiagnostics(List<DiagnosticSquiggle> squiggles) {
    _diagnostics.clear();
    _diagnostics.addAll(squiggles);
    _notifyListeners();
  }

  /// Adds a single diagnostic squiggle.
  void addDiagnostic(DiagnosticSquiggle squiggle) {
    _diagnostics.add(squiggle);
    _notifyListeners();
  }

  /// Clears all diagnostic squiggles.
  void clearDiagnostics() {
    if (_diagnostics.isNotEmpty) {
      _diagnostics.clear();
      _notifyListeners();
    }
  }

  /// Formats the active buffer using [DartFormatterEngine] and records an undo step.
  void formatDocument() {
    if (_options.readOnly) return;
    final oldText = _buffer.text;
    final formatted = _formatter.formatCode(oldText);
    if (formatted == oldText) return;

    final oldLen = _buffer.length;
    _buffer.delete(0, oldLen);
    _buffer.insert(0, formatted);

    final tx = EditorTransaction(
      operations: [
        DeleteOperation(0, oldLen, deletedText: oldText),
        InsertOperation(0, formatted),
      ],
      selectionsBefore: [_selectionModel.primarySelection],
      selectionsAfter: [
        TextSelection.collapsed(
          TextPosition(
            _buffer.lineCount - 1,
            _buffer.getLineLength(_buffer.lineCount - 1),
          ),
        ),
      ],
    );

    _undoManager.commitTransaction(tx);
    _lspSync.createChangeEvents(_buffer, tx);

    _tokenizeAll();
    _notifyListeners();
  }

  /// Snippet engine.
  SnippetEngine get snippetEngine => _snippetEngine;

  /// Folding manager.
  FoldingManager get foldingManager => _foldingManager;

  /// Sets the full text of the document, resetting transaction history.
  set text(String newText) {
    final oldText = _buffer.text;
    _buffer.delete(0, _buffer.length);
    _buffer.insert(0, newText);
    _selectionModel.collapseTo(const TextPosition(0, 0));
    _undoManager.clear();
    _tokenizeAll();
    final tx = EditorTransaction(
      operations: [
        if (oldText.isNotEmpty)
          DeleteOperation(0, oldText.length, deletedText: oldText),
        if (newText.isNotEmpty) InsertOperation(0, newText),
      ],
      selectionsBefore: const [
        TextSelection(base: TextPosition(0, 0), extent: TextPosition(0, 0)),
      ],
      selectionsAfter: const [
        TextSelection(base: TextPosition(0, 0), extent: TextPosition(0, 0)),
      ],
    );
    _lspSync.createChangeEvents(_buffer, tx);
    _notifyListeners();
  }

  /// Total number of lines in the document.
  int get lineCount => _buffer.lineCount;

  /// Total character length of the document.
  int get length => _buffer.length;

  /// Multi-cursor selection model.
  SelectionModel get selectionModel => _selectionModel;

  /// Primary selection.
  TextSelection get selection => _selectionModel.primarySelection;

  /// Sets the primary selection.
  set selection(TextSelection sel) {
    _selectionModel.primarySelection = sel;
    _notifyListeners();
  }

  /// Current editor options.
  EditorOptions get options => _options;

  /// Updates editor options.
  set options(EditorOptions opts) {
    _options = opts;
    _notifyListeners();
  }

  /// Undo manager.
  UndoManager get undoManager => _undoManager;

  /// Find and replace controller.
  FindReplaceController get findReplace => _findReplace;

  /// Cached syntax tokens per line.
  List<List<SyntaxToken>> get lineTokens => _tokenizer.cachedLineTokens;

  /// Adds a listener callback invoked when the buffer or selection changes.
  void addListener(void Function() listener) {
    _changeListeners.add(listener);
  }

  /// Removes a registered change listener.
  void removeListener(void Function() listener) {
    _changeListeners.remove(listener);
  }

  void _notifyListeners() {
    for (final l in List<void Function()>.from(_changeListeners)) {
      l();
    }
  }

  void _tokenizeAll() {
    final lines = <String>[];
    for (int i = 0; i < _buffer.lineCount; i++) {
      final start = _buffer.getLineOffset(i);
      final len = _buffer.getLineLength(i);
      lines.add(_buffer.getTextInRange(start, len));
    }
    _tokenizer.setDocument(lines);
  }

  void _retokenizeFrom(int lineIndex) {
    final lines = <String>[];
    for (int i = 0; i < _buffer.lineCount; i++) {
      final start = _buffer.getLineOffset(i);
      final len = _buffer.getLineLength(i);
      lines.add(_buffer.getTextInRange(start, len));
    }
    _tokenizer.updateDocument(lines, startLineIndex: lineIndex);
  }

  // --- Document Edit Actions ---

  /// Inserts [inputText] at the current selection(s), handling auto-closing pairs.
  void insertText(String inputText) {
    if (_options.readOnly || inputText.isEmpty) return;

    final primarySel = _selectionModel.primarySelection;

    // Check auto-close behavior for single-character typing
    if (_options.autoClosingBrackets && inputText.length == 1) {
      final action = _autoCloseEngine.handleType(
        typedChar: inputText,
        buffer: _buffer,
        selection: primarySel,
      );

      if (action is SkipCloseAction) {
        // Step over closing bracket without inserting duplicate
        final newPos = TextPosition(
          primarySel.extent.line,
          primarySel.extent.column + 1,
        );
        _selectionModel.collapseTo(newPos);
        _notifyListeners();
        return;
      } else if (action is InsertPairAction) {
        final pairText = '${action.open}${action.close}';
        _applyInsert(pairText, caretOffsetDelta: action.open.length);
        return;
      } else if (action is WrapSelectionAction) {
        final startOffset =
            _buffer.getLineOffset(primarySel.start.line) +
            primarySel.start.column;
        final endOffset =
            _buffer.getLineOffset(primarySel.end.line) + primarySel.end.column;
        final selectedText = _buffer.getTextInRange(
          startOffset,
          endOffset - startOffset,
        );
        final wrappedText = '${action.open}$selectedText${action.close}';

        _applyReplace(
          startOffset: startOffset,
          length: endOffset - startOffset,
          replacementText: wrappedText,
          newSelection: TextSelection(
            base: TextPosition(
              primarySel.base.line,
              primarySel.base.column + action.open.length,
            ),
            extent: TextPosition(
              primarySel.extent.line,
              primarySel.extent.column + action.open.length,
            ),
          ),
        );
        return;
      }
    }

    _applyInsert(inputText);
  }

  void _applyInsert(String textToInsert, {int? caretOffsetDelta}) {
    final selectionsBefore = List<TextSelection>.from(
      _selectionModel.selections,
    );
    // Sort indices by offset descending so modifying lower lines doesn't invalidate upper offsets
    final indices = List<int>.generate(selectionsBefore.length, (i) => i);
    indices.sort((i, j) {
      final offI =
          _buffer.getLineOffset(selectionsBefore[i].start.line) +
          selectionsBefore[i].start.column;
      final offJ =
          _buffer.getLineOffset(selectionsBefore[j].start.line) +
          selectionsBefore[j].start.column;
      return offJ.compareTo(offI);
    });

    final ops = <EditOperation>[];
    final updatedSelections = List<TextSelection?>.filled(
      selectionsBefore.length,
      null,
    );
    int minModifiedLine = _buffer.lineCount;

    for (final idx in indices) {
      final sel = selectionsBefore[idx];
      final startOffset =
          _buffer.getLineOffset(sel.start.line) + sel.start.column;
      final endOffset = _buffer.getLineOffset(sel.end.line) + sel.end.column;
      final deleteLen = endOffset - startOffset;

      if (deleteLen > 0) {
        final deletedText = _buffer.getTextInRange(startOffset, deleteLen);
        ops.add(
          DeleteOperation(startOffset, deleteLen, deletedText: deletedText),
        );
        _buffer.delete(startOffset, deleteLen);
      }

      ops.add(InsertOperation(startOffset, textToInsert));
      _buffer.insert(startOffset, textToInsert);

      final finalOffset =
          startOffset + (caretOffsetDelta ?? textToInsert.length);
      final (newLine, newCol) = _buffer.getLineAndColumnAt(finalOffset);
      updatedSelections[idx] = TextSelection.collapsed(
        TextPosition(newLine, newCol),
      );
      if (sel.start.line < minModifiedLine) {
        minModifiedLine = sel.start.line;
      }
    }

    final nonNullSelections = updatedSelections
        .whereType<TextSelection>()
        .toList();
    final tx = EditorTransaction(
      operations: ops,
      selectionsBefore: selectionsBefore,
      selectionsAfter: nonNullSelections,
    );
    _undoManager.commitTransaction(tx);
    _lspSync.createChangeEvents(_buffer, tx);

    if (nonNullSelections.isNotEmpty) {
      _selectionModel.setSelections(nonNullSelections);
    }
    _retokenizeFrom(math.max(0, minModifiedLine));
    _notifyListeners();
  }

  void _applyReplace({
    required int startOffset,
    required int length,
    required String replacementText,
    required TextSelection newSelection,
  }) {
    final sel = _selectionModel.primarySelection;
    final ops = <EditOperation>[];

    if (length > 0) {
      final deletedText = _buffer.getTextInRange(startOffset, length);
      ops.add(DeleteOperation(startOffset, length, deletedText: deletedText));
      _buffer.delete(startOffset, length);
    }
    if (replacementText.isNotEmpty) {
      ops.add(InsertOperation(startOffset, replacementText));
      _buffer.insert(startOffset, replacementText);
    }

    final tx = EditorTransaction(
      operations: ops,
      selectionsBefore: [sel],
      selectionsAfter: [newSelection],
    );
    _undoManager.commitTransaction(tx);
    _lspSync.createChangeEvents(_buffer, tx);
    _selectionModel.primarySelection = newSelection;
    final (startLine, _) = _buffer.getLineAndColumnAt(startOffset);
    _retokenizeFrom(startLine);
    _notifyListeners();
  }

  /// Deletes backward from the cursor (Backspace key).
  void deleteBackward() {
    if (_options.readOnly || _buffer.length == 0) return;

    final sel = _selectionModel.primarySelection;
    if (!sel.isCollapsed) {
      _applyInsert('');
      return;
    }

    final pos = sel.extent;
    if (pos.line == 0 && pos.column == 0) return;

    // Check if backspace should delete matching bracket pair
    int deleteCount = 1;
    if (_options.autoClosingBrackets) {
      deleteCount = _autoCloseEngine.checkBackspacePairDeletion(
        buffer: _buffer,
        position: pos,
      );
    }

    final currentOffset = _buffer.getLineOffset(pos.line) + pos.column;
    final startOffset = math.max(0, currentOffset - (deleteCount == 2 ? 1 : 1));
    final len = (deleteCount == 2) ? 2 : 1;

    final deletedText = _buffer.getTextInRange(startOffset, len);
    _buffer.delete(startOffset, len);

    final (newLine, newCol) = _buffer.getLineAndColumnAt(startOffset);
    final newSel = TextSelection.collapsed(TextPosition(newLine, newCol));

    final tx = EditorTransaction(
      operations: [DeleteOperation(startOffset, len, deletedText: deletedText)],
      selectionsBefore: [sel],
      selectionsAfter: [newSel],
    );
    _undoManager.commitTransaction(tx);
    _lspSync.createChangeEvents(_buffer, tx);
    _selectionModel.primarySelection = newSel;
    _retokenizeFrom(newLine);
    _notifyListeners();
  }

  /// Deletes forward from the cursor (Delete key).
  void deleteForward() {
    if (_options.readOnly || _buffer.length == 0) return;

    final sel = _selectionModel.primarySelection;
    if (!sel.isCollapsed) {
      _applyInsert('');
      return;
    }

    final pos = sel.extent;
    final currentOffset = _buffer.getLineOffset(pos.line) + pos.column;
    if (currentOffset >= _buffer.length) return;

    final deletedText = _buffer.getTextInRange(currentOffset, 1);
    _buffer.delete(currentOffset, 1);

    final tx = EditorTransaction(
      operations: [DeleteOperation(currentOffset, 1, deletedText: deletedText)],
      selectionsBefore: [sel],
      selectionsAfter: [sel],
    );
    _undoManager.commitTransaction(tx);
    _lspSync.createChangeEvents(_buffer, tx);
    _retokenizeFrom(pos.line);
    _notifyListeners();
  }

  /// Inserts a newline with smart indentation (Enter key).
  void insertNewline() {
    if (_options.readOnly) return;

    if (!_options.smartIndent) {
      _applyInsert('\n');
      return;
    }

    final sel = _selectionModel.primarySelection;
    final res = _smartIndentEngine.calculateEnter(
      buffer: _buffer,
      position: sel.extent,
    );

    _applyInsert(res.textToInsert, caretOffsetDelta: res.relativeCaretOffset);
  }

  /// Handles Tab key: cycles active snippet tab stop, expands snippet prefix, or indents selection.
  void tabPressed() {
    if (_options.readOnly) return;

    if (_activeSnippet != null && _activeSnippet!.tabStops.isNotEmpty) {
      _activeTabStopIndex =
          (_activeTabStopIndex + 1) % _activeSnippet!.tabStops.length;
      final nextStop = _activeSnippet!.tabStops[_activeTabStopIndex];
      final (startL, startC) = _buffer.getLineAndColumnAt(nextStop.startOffset);
      final (endL, endC) = _buffer.getLineAndColumnAt(
        nextStop.startOffset + nextStop.length,
      );
      _selectionModel.primarySelection = TextSelection(
        base: TextPosition(startL, startC),
        extent: TextPosition(endL, endC),
      );
      _notifyListeners();
      return;
    }

    final sel = _selectionModel.primarySelection;
    if (sel.isCollapsed) {
      // Check if word immediately left of cursor matches a snippet prefix
      final lineOffset = _buffer.getLineOffset(sel.extent.line);
      final lineLen = _buffer.getLineLength(sel.extent.line);
      final lineText = _buffer.getTextInRange(lineOffset, lineLen);
      final wordStartCol = SelectionModel.getWordStartColumn(
        lineText,
        sel.extent.column,
      );
      final candidateWord = lineText.substring(wordStartCol, sel.extent.column);

      final snippet = _snippetEngine.findSnippet(candidateWord);
      if (snippet != null) {
        final insertOffset = lineOffset + wordStartCol;
        // Delete candidate prefix first
        _buffer.delete(insertOffset, candidateWord.length);

        final res = _snippetEngine.expandSnippet(
          body: snippet.body,
          insertOffset: insertOffset,
          buffer: _buffer,
        );

        _applyInsert(res.insertedText);
        _activeSnippet = res;
        _activeTabStopIndex = 0;
        _selectionModel.primarySelection = res.initialSelection;
        _notifyListeners();
        return;
      }
    }

    indentSelection();
  }

  /// Handles Shift+Tab key: cycles previous snippet tab stop or outdents selection.
  void shiftTabPressed() {
    if (_options.readOnly) return;

    if (_activeSnippet != null && _activeSnippet!.tabStops.isNotEmpty) {
      _activeTabStopIndex =
          (_activeTabStopIndex - 1 + _activeSnippet!.tabStops.length) %
          _activeSnippet!.tabStops.length;
      final prevStop = _activeSnippet!.tabStops[_activeTabStopIndex];
      final (startL, startC) = _buffer.getLineAndColumnAt(prevStop.startOffset);
      final (endL, endC) = _buffer.getLineAndColumnAt(
        prevStop.startOffset + prevStop.length,
      );
      _selectionModel.primarySelection = TextSelection(
        base: TextPosition(startL, startC),
        extent: TextPosition(endL, endC),
      );
      _notifyListeners();
      return;
    }

    outdentSelection();
  }

  /// Indents selected lines or inserts tab spaces at cursor (Tab key fallback).
  void indentSelection() {
    if (_options.readOnly) return;

    final sel = _selectionModel.primarySelection;
    if (sel.isCollapsed) {
      final tabSpaces = ' ' * _options.tabSize;
      _applyInsert(tabSpaces);
    } else {
      final startLine = sel.start.line;
      final endLine = sel.end.line;
      final lines = <String>[];
      for (int l = startLine; l <= endLine; l++) {
        final off = _buffer.getLineOffset(l);
        final len = _buffer.getLineLength(l);
        lines.add(_buffer.getTextInRange(off, len));
      }
      final indented = _smartIndentEngine.indentLines(lines);
      final startOffset = _buffer.getLineOffset(startLine);
      final endOffset = (endLine + 1 < _buffer.lineCount)
          ? _buffer.getLineOffset(endLine + 1)
          : _buffer.length;

      final replacement =
          indented.join('\n') + (endLine + 1 < _buffer.lineCount ? '\n' : '');
      _applyReplace(
        startOffset: startOffset,
        length: endOffset - startOffset,
        replacementText: replacement,
        newSelection: TextSelection(
          base: TextPosition(sel.base.line, sel.base.column + _options.tabSize),
          extent: TextPosition(
            sel.extent.line,
            sel.extent.column + _options.tabSize,
          ),
        ),
      );
    }
  }

  /// Outdents selected lines (Shift+Tab).
  void outdentSelection() {
    if (_options.readOnly) return;

    final sel = _selectionModel.primarySelection;
    final startLine = sel.start.line;
    final endLine = sel.end.line;
    final lines = <String>[];
    for (int l = startLine; l <= endLine; l++) {
      final off = _buffer.getLineOffset(l);
      final len = _buffer.getLineLength(l);
      lines.add(_buffer.getTextInRange(off, len));
    }
    final outdented = _smartIndentEngine.outdentLines(lines);
    final startOffset = _buffer.getLineOffset(startLine);
    final endOffset = (endLine + 1 < _buffer.lineCount)
        ? _buffer.getLineOffset(endLine + 1)
        : _buffer.length;

    final replacement =
        outdented.join('\n') + (endLine + 1 < _buffer.lineCount ? '\n' : '');
    _applyReplace(
      startOffset: startOffset,
      length: endOffset - startOffset,
      replacementText: replacement,
      newSelection: sel,
    );
  }

  // --- Line Operations ---

  /// Duplicates current line(s) down.
  void duplicateLinesDown() {
    if (_options.readOnly) return;
    final res = _lineOperations.duplicateLinesDown(
      buffer: _buffer,
      selection: _selectionModel.primarySelection,
    );
    _applyReplace(
      startOffset: res.replaceStartOffset,
      length: res.replaceLength,
      replacementText: res.newText,
      newSelection: res.newSelection,
    );
  }

  /// Moves current line(s) up.
  void moveLinesUp() {
    if (_options.readOnly) return;
    final res = _lineOperations.moveLinesUp(
      buffer: _buffer,
      selection: _selectionModel.primarySelection,
    );
    if (res != null) {
      _applyReplace(
        startOffset: res.replaceStartOffset,
        length: res.replaceLength,
        replacementText: res.newText,
        newSelection: res.newSelection,
      );
    }
  }

  /// Moves current line(s) down.
  void moveLinesDown() {
    if (_options.readOnly) return;
    final res = _lineOperations.moveLinesDown(
      buffer: _buffer,
      selection: _selectionModel.primarySelection,
    );
    if (res != null) {
      _applyReplace(
        startOffset: res.replaceStartOffset,
        length: res.replaceLength,
        replacementText: res.newText,
        newSelection: res.newSelection,
      );
    }
  }

  /// Deletes current line(s).
  void deleteLines() {
    if (_options.readOnly) return;
    final res = _lineOperations.deleteLines(
      buffer: _buffer,
      selection: _selectionModel.primarySelection,
    );
    _applyReplace(
      startOffset: res.replaceStartOffset,
      length: res.replaceLength,
      replacementText: res.newText,
      newSelection: res.newSelection,
    );
  }

  /// Toggles line comments (`//`).
  void toggleLineComment() {
    if (_options.readOnly) return;
    final res = _lineOperations.toggleLineComment(
      buffer: _buffer,
      selection: _selectionModel.primarySelection,
    );
    _applyReplace(
      startOffset: res.replaceStartOffset,
      length: res.replaceLength,
      replacementText: res.newText,
      newSelection: res.newSelection,
    );
  }

  // --- Undo / Redo ---

  /// Undoes the last transaction.
  bool undo() {
    if (_options.readOnly) return false;
    final invTx = _undoManager.undo(_buffer, _selectionModel);
    if (invTx != null) {
      _lspSync.createChangeEvents(_buffer, invTx);
      _tokenizeAll();
      _notifyListeners();
      return true;
    }
    return false;
  }

  /// Redoes the next undone transaction.
  bool redo() {
    if (_options.readOnly) return false;
    final redoTx = _undoManager.redo(_buffer, _selectionModel);
    if (redoTx != null) {
      _lspSync.createChangeEvents(_buffer, redoTx);
      _tokenizeAll();
      _notifyListeners();
      return true;
    }
    return false;
  }

  // --- Bracket Matching ---

  /// Finds the matching bracket adjacent to current cursor.
  BracketMatchResult? getMatchingBracket() {
    if (!_options.matchBrackets) return null;
    return _bracketMatcher.findMatchingBracket(
      _buffer,
      _selectionModel.primarySelection.extent,
    );
  }

  // --- Navigation Movements ---

  /// Moves the cursor according to [movement], optionally extending selection if [select] is true.
  void moveCursor(CursorMovement movement, {bool select = false}) {
    final sel = _selectionModel.primarySelection;
    final currentPos = sel.extent;
    TextPosition newPos = currentPos;

    switch (movement) {
      case CursorMovement.left:
        if (!select && !sel.isCollapsed) {
          newPos = sel.start;
        } else if (currentPos.column > 0) {
          newPos = TextPosition(currentPos.line, currentPos.column - 1);
        } else if (currentPos.line > 0) {
          final prevLine = currentPos.line - 1;
          final prevLineLen = _buffer.getLineLength(prevLine);
          newPos = TextPosition(prevLine, prevLineLen);
        }
        break;

      case CursorMovement.right:
        if (!select && !sel.isCollapsed) {
          newPos = sel.end;
        } else {
          final curLineLen = _buffer.getLineLength(currentPos.line);
          if (currentPos.column < curLineLen) {
            newPos = TextPosition(currentPos.line, currentPos.column + 1);
          } else if (currentPos.line < _buffer.lineCount - 1) {
            newPos = TextPosition(currentPos.line + 1, 0);
          }
        }
        break;

      case CursorMovement.up:
        if (currentPos.line > 0) {
          final targetLine = currentPos.line - 1;
          final targetLineLen = _buffer.getLineLength(targetLine);
          newPos = TextPosition(
            targetLine,
            math.min(currentPos.column, targetLineLen),
          );
        } else {
          newPos = const TextPosition(0, 0);
        }
        break;

      case CursorMovement.down:
        if (currentPos.line < _buffer.lineCount - 1) {
          final targetLine = currentPos.line + 1;
          final targetLineLen = _buffer.getLineLength(targetLine);
          newPos = TextPosition(
            targetLine,
            math.min(currentPos.column, targetLineLen),
          );
        } else {
          final lastLine = _buffer.lineCount - 1;
          final lastLineLen = _buffer.getLineLength(lastLine);
          newPos = TextPosition(lastLine, lastLineLen);
        }
        break;

      case CursorMovement.wordLeft:
        final lineOffset = _buffer.getLineOffset(currentPos.line);
        final lineLen = _buffer.getLineLength(currentPos.line);
        final lineText = _buffer.getTextInRange(lineOffset, lineLen);
        final wordStart = SelectionModel.getWordStartColumn(
          lineText,
          currentPos.column,
        );
        newPos = TextPosition(currentPos.line, wordStart);
        break;

      case CursorMovement.wordRight:
        final lineOffset = _buffer.getLineOffset(currentPos.line);
        final lineLen = _buffer.getLineLength(currentPos.line);
        final lineText = _buffer.getTextInRange(lineOffset, lineLen);
        final wordEnd = SelectionModel.getWordEndColumn(
          lineText,
          currentPos.column,
        );
        newPos = TextPosition(currentPos.line, wordEnd);
        break;

      case CursorMovement.lineStart:
        final lineOffset = _buffer.getLineOffset(currentPos.line);
        final lineLen = _buffer.getLineLength(currentPos.line);
        final lineText = _buffer.getTextInRange(lineOffset, lineLen);
        newPos = SelectionModel.getSmartLineStart(lineText, currentPos);
        break;

      case CursorMovement.lineEnd:
        final lineLen = _buffer.getLineLength(currentPos.line);
        newPos = TextPosition(currentPos.line, lineLen);
        break;

      case CursorMovement.pageUp:
        newPos = SelectionModel.movePageUp(currentPos, 20);
        break;

      case CursorMovement.pageDown:
        newPos = SelectionModel.movePageDown(currentPos, _buffer.lineCount, 20);
        break;

      case CursorMovement.documentStart:
        newPos = const TextPosition(0, 0);
        break;

      case CursorMovement.documentEnd:
        final lastLine = _buffer.lineCount - 1;
        final lastLineLen = _buffer.getLineLength(lastLine);
        newPos = TextPosition(lastLine, lastLineLen);
        break;
    }

    if (select) {
      _selectionModel.primarySelection = TextSelection(
        base: sel.base,
        extent: newPos,
      );
    } else {
      _selectionModel.collapseTo(newPos);
    }
    _notifyListeners();
  }

  /// Selects the entire document.
  void selectAll() {
    final lines = <String>[];
    for (int i = 0; i < _buffer.lineCount; i++) {
      final off = _buffer.getLineOffset(i);
      final len = _buffer.getLineLength(i);
      lines.add(_buffer.getTextInRange(off, len));
    }
    _selectionModel.selectAll(lines);
    _notifyListeners();
  }

  /// Adds a secondary collapsed caret at [position] (for Alt+Click multi-caret editing).
  void addSecondaryCaret(TextPosition position) {
    final cur = _selectionModel.secondarySelections;
    _selectionModel.secondarySelections = [
      ...cur,
      TextSelection.collapsed(position),
    ];
    _notifyListeners();
  }

  /// Selects the next occurrence of the current selection (or word at cursor)
  /// and adds a secondary caret (for Ctrl+D / Cmd+D multi-caret editing).
  void selectNextOccurrence() {
    if (_selectionModel.primarySelection.isCollapsed) {
      selectWordAt(_selectionModel.primarySelection.extent);
      return;
    }

    final sel = _selectionModel.primarySelection;
    final startOff = _buffer.getLineOffset(sel.start.line) + sel.start.column;
    final endOff = _buffer.getLineOffset(sel.end.line) + sel.end.column;
    final targetText = _buffer.getTextInRange(startOff, endOff - startOff);
    if (targetText.isEmpty) return;

    final lastSel = _selectionModel.selections.last;
    final searchStartOffset =
        _buffer.getLineOffset(lastSel.end.line) + lastSel.end.column;
    final nextIdx = _buffer.text.indexOf(targetText, searchStartOffset);

    if (nextIdx != -1) {
      final (startLine, startCol) = _buffer.getLineAndColumnAt(nextIdx);
      final (endLine, endCol) = _buffer.getLineAndColumnAt(
        nextIdx + targetText.length,
      );
      final newSel = TextSelection(
        base: TextPosition(startLine, startCol),
        extent: TextPosition(endLine, endCol),
      );
      _selectionModel.secondarySelections = [
        ..._selectionModel.secondarySelections,
        newSel,
      ];
      _notifyListeners();
    }
  }

  /// Selects the word at [position].
  void selectWordAt(TextPosition position) {
    final lines = <String>[];
    for (int i = 0; i < _buffer.lineCount; i++) {
      final off = _buffer.getLineOffset(i);
      final len = _buffer.getLineLength(i);
      lines.add(_buffer.getTextInRange(off, len));
    }
    _selectionModel.selectWordAt(lines, position);
    _notifyListeners();
  }

  /// Selects the full line at [lineIndex].
  void selectLineAt(int lineIndex) {
    final lines = <String>[];
    for (int i = 0; i < _buffer.lineCount; i++) {
      final off = _buffer.getLineOffset(i);
      final len = _buffer.getLineLength(i);
      lines.add(_buffer.getTextInRange(off, len));
    }
    _selectionModel.selectLineAt(lines, lineIndex);
    _notifyListeners();
  }

  // --- Search & Replace ---

  /// Finds all occurrences matching [options].
  void find(SearchOptions options) {
    _findReplace.find(_buffer, options);
    if (_findReplace.activeMatch != null) {
      _selectionModel.primarySelection = _findReplace.activeMatch!.range;
    }
    _notifyListeners();
  }

  /// Advances to the next search match.
  SearchMatch? findNext() {
    final match = _findReplace.findNext();
    if (match != null) {
      _selectionModel.primarySelection = match.range;
      _notifyListeners();
    }
    return match;
  }

  /// Moves to the previous search match.
  SearchMatch? findPrevious() {
    final match = _findReplace.findPrevious();
    if (match != null) {
      _selectionModel.primarySelection = match.range;
      _notifyListeners();
    }
    return match;
  }

  /// Replaces the currently active search match with [replacement].
  bool replaceCurrent(String replacement) {
    if (_options.readOnly) return false;
    final match = _findReplace.activeMatch;
    if (match == null) return false;

    final (afterLine, afterCol) = _buffer.getLineAndColumnAt(
      match.startOffset + replacement.length,
    );
    _applyReplace(
      startOffset: match.startOffset,
      length: match.length,
      replacementText: replacement,
      newSelection: TextSelection.collapsed(TextPosition(afterLine, afterCol)),
    );

    // Re-run search
    find(_findReplace.options);
    return true;
  }

  /// Replaces all search matches with [replacement].
  int replaceAll(String replacement) {
    if (_options.readOnly || _findReplace.matches.isEmpty) return 0;
    final matches = List<SearchMatch>.from(_findReplace.matches);
    int count = 0;

    // Replace in reverse order so offsets remain stable
    for (int i = matches.length - 1; i >= 0; i--) {
      final m = matches[i];
      final deletedText = _buffer.getTextInRange(m.startOffset, m.length);
      _buffer.delete(m.startOffset, m.length);
      if (replacement.isNotEmpty) {
        _buffer.insert(m.startOffset, replacement);
      }
      final (afterLine, afterCol) = _buffer.getLineAndColumnAt(
        m.startOffset + replacement.length,
      );
      final tx = EditorTransaction(
        operations: [
          DeleteOperation(m.startOffset, m.length, deletedText: deletedText),
          if (replacement.isNotEmpty)
            InsertOperation(m.startOffset, replacement),
        ],
        selectionsBefore: [m.range],
        selectionsAfter: [
          TextSelection.collapsed(TextPosition(afterLine, afterCol)),
        ],
      );
      _undoManager.commitTransaction(tx);
      _lspSync.createChangeEvents(_buffer, tx);
      count++;
    }

    _tokenizeAll();
    find(_findReplace.options);
    _notifyListeners();
    return count;
  }
}
