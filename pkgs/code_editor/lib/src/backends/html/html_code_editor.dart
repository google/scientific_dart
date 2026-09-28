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

import 'dart:async';
import 'dart:js_interop';
import 'dart:math' as math;
import 'package:web/web.dart' as web;
import '../../editor/code_editor_controller.dart';
import '../../core/selection/selection_model.dart';
import '../../features/find_replace/search_match.dart';
import '../../syntax/syntax_token.dart';

/// Full interactive HTML/DOM code editor component.
final class HtmlCodeEditor {
  final web.HTMLElement hostElement;
  final CodeEditorController controller;

  late final web.HTMLDivElement _root;
  late final web.HTMLDivElement _gutter;
  late final web.HTMLDivElement _linesLayer;
  late final web.HTMLDivElement _selectionsLayer;
  late final web.HTMLDivElement _caretsLayer;
  late final web.HTMLDivElement _activeLineLayer;
  late final web.HTMLDivElement _popupsLayer;
  late final web.HTMLTextAreaElement _hiddenInput;

  // Rich UI widgets
  late final web.HTMLDivElement _findReplacePanel;
  late final web.HTMLInputElement _searchInput;
  late final web.HTMLInputElement _replaceInput;
  late final web.HTMLElement _matchCountBadge;
  late final web.HTMLDivElement _completionMenu;
  late final web.HTMLDivElement _hoverCard;

  bool _isFindOpen = false;
  bool get isFindOpen => _isFindOpen;
  bool _isCompletionOpen = false;
  List<_CompletionItem> _currentCompletions = [];
  int _selectedCompletionIndex = 0;
  Timer? _hoverTimer;

  void Function()? onExecute;
  void Function(int offset, int clientX, int clientY)? onCompletionRequested;
  void Function(int offset, int clientX, int clientY)? onHoverRequested;

  bool _isDragging = false;
  bool _isFocused = false;

  /// The root DOM element for this editor.
  web.HTMLDivElement get rootElement => _root;

  /// Creates and mounts an [HtmlCodeEditor] into [hostElement].
  HtmlCodeEditor({
    required this.hostElement,
    required this.controller,
    this.onExecute,
    this.onCompletionRequested,
    this.onHoverRequested,
  }) {
    _initDom();
    _bindEvents();
    controller.addListener(render);
    render();
  }

  void _initDom() {
    hostElement.innerHTML = ''.toJS;

    _root = web.document.createElement('div') as web.HTMLDivElement;
    _root.className = 'dart-code-editor-root';
    _root.style.position = 'relative';
    _root.style.width = '100%';
    _root.style.minHeight = '150px';
    _root.style.backgroundColor = '#1e1e2e';
    _root.style.color = '#cdd6f4';
    _root.style.fontFamily = controller.options.fontFamily;
    _root.style.fontSize = '${controller.options.fontSize}px';
    _root.style.lineHeight = '${controller.options.lineHeight}px';
    _root.style.borderRadius = '8px';
    _root.style.border = '1px solid #313244';
    _root.style.overflow = 'hidden';
    _root.style.userSelect = 'none';

    // Hidden input for capturing physical keystrokes and IME composition
    _hiddenInput =
        web.document.createElement('textarea') as web.HTMLTextAreaElement;
    _hiddenInput.className = 'dart-editor-hidden-input';
    _hiddenInput.style.position = 'absolute';
    _hiddenInput.style.left = '-9999px';
    _hiddenInput.style.top = '0px';
    _hiddenInput.style.width = '1px';
    _hiddenInput.style.height = '1px';
    _hiddenInput.style.opacity = '0';
    _hiddenInput.style.zIndex = '-1';
    _hiddenInput.setAttribute('autocapitalize', 'off');
    _hiddenInput.setAttribute('autocomplete', 'off');
    _hiddenInput.setAttribute('autocorrect', 'off');
    _hiddenInput.setAttribute('spellcheck', 'false');

    // Gutter layer
    _gutter = web.document.createElement('div') as web.HTMLDivElement;
    _gutter.className = 'editor-gutter';
    _gutter.style.position = 'absolute';
    _gutter.style.left = '0px';
    _gutter.style.top = '0px';
    _gutter.style.bottom = '0px';
    _gutter.style.width = '52px';
    _gutter.style.backgroundColor = '#181825';
    _gutter.style.borderRight = '1px solid #313244';
    _gutter.style.color = '#6c7086';
    _gutter.style.textAlign = 'right';
    _gutter.style.paddingRight = '8px';
    _gutter.style.boxSizing = 'border-box';
    _gutter.style.userSelect = 'none';

    // Active line highlight layer
    _activeLineLayer = web.document.createElement('div') as web.HTMLDivElement;
    _activeLineLayer.className = 'editor-active-line-layer';
    _activeLineLayer.style.position = 'absolute';
    _activeLineLayer.style.left = '52px';
    _activeLineLayer.style.right = '0px';
    _activeLineLayer.style.pointerEvents = 'none';

    // Selection highlight layer
    _selectionsLayer = web.document.createElement('div') as web.HTMLDivElement;
    _selectionsLayer.className = 'editor-selections-layer';
    _selectionsLayer.style.position = 'absolute';
    _selectionsLayer.style.left = '52px';
    _selectionsLayer.style.top = '0px';
    _selectionsLayer.style.right = '0px';
    _selectionsLayer.style.bottom = '0px';
    _selectionsLayer.style.pointerEvents = 'none';

    // Lines text content layer
    _linesLayer = web.document.createElement('div') as web.HTMLDivElement;
    _linesLayer.className = 'editor-lines-layer';
    _linesLayer.style.position = 'absolute';
    _linesLayer.style.left = '60px';
    _linesLayer.style.top = '0px';
    _linesLayer.style.right = '0px';
    _linesLayer.style.bottom = '0px';
    _linesLayer.style.whiteSpace = 'pre';
    _linesLayer.style.fontFamily = controller.options.fontFamily;
    _linesLayer.style.cursor = 'text';

    // Caret blinking layer
    _caretsLayer = web.document.createElement('div') as web.HTMLDivElement;
    _caretsLayer.className = 'editor-carets-layer';
    _caretsLayer.style.position = 'absolute';
    _caretsLayer.style.left = '60px';
    _caretsLayer.style.top = '0px';
    _caretsLayer.style.right = '0px';
    _caretsLayer.style.bottom = '0px';
    _caretsLayer.style.pointerEvents = 'none';

    // Popups overlay layer
    _popupsLayer = web.document.createElement('div') as web.HTMLDivElement;
    _popupsLayer.className = 'editor-popups-layer';
    _popupsLayer.style.position = 'absolute';
    _popupsLayer.style.left = '0px';
    _popupsLayer.style.top = '0px';
    _popupsLayer.style.width = '100%';
    _popupsLayer.style.height = '100%';
    _popupsLayer.style.pointerEvents = 'none';

    // Build Find/Replace UI Panel
    _buildFindReplacePanel();
    // Build Autocomplete Menu
    _buildCompletionMenu();
    // Build Hover Doc Card
    _buildHoverCard();

    _root.appendChild(_hiddenInput);
    _root.appendChild(_gutter);
    _root.appendChild(_activeLineLayer);
    _root.appendChild(_selectionsLayer);
    _root.appendChild(_linesLayer);
    _root.appendChild(_caretsLayer);
    _root.appendChild(_popupsLayer);
    _root.appendChild(_findReplacePanel);
    _root.appendChild(_completionMenu);
    _root.appendChild(_hoverCard);

    hostElement.appendChild(_root);
  }

  void _buildFindReplacePanel() {
    _findReplacePanel = web.document.createElement('div') as web.HTMLDivElement;
    _findReplacePanel.className = 'editor-find-replace-panel';
    _findReplacePanel.style.position = 'absolute';
    _findReplacePanel.style.top = '8px';
    _findReplacePanel.style.right = '16px';
    _findReplacePanel.style.zIndex = '100';
    _findReplacePanel.style.backgroundColor = '#181825';
    _findReplacePanel.style.border = '1px solid #45475a';
    _findReplacePanel.style.borderRadius = '6px';
    _findReplacePanel.style.padding = '8px 12px';
    _findReplacePanel.style.boxShadow = '0 6px 16px rgba(0,0,0,0.45)';
    _findReplacePanel.style.display = 'none';
    _findReplacePanel.style.flexDirection = 'column';
    _findReplacePanel.style.gap = '6px';

    final row1 = web.document.createElement('div') as web.HTMLDivElement;
    row1.style.display = 'flex';
    row1.style.alignItems = 'center';
    row1.style.gap = '6px';

    _searchInput = web.document.createElement('input') as web.HTMLInputElement;
    _searchInput.type = 'text';
    _searchInput.placeholder = 'Find (Ctrl+F)...';
    _searchInput.style.backgroundColor = '#1e1e2e';
    _searchInput.style.border = '1px solid #313244';
    _searchInput.style.borderRadius = '4px';
    _searchInput.style.color = '#cdd6f4';
    _searchInput.style.padding = '4px 8px';
    _searchInput.style.fontSize = '12px';
    _searchInput.style.outline = 'none';
    _searchInput.style.width = '160px';

    _matchCountBadge = web.document.createElement('span') as web.HTMLElement;
    _matchCountBadge.style.fontSize = '11px';
    _matchCountBadge.style.color = '#a6adc8';
    _matchCountBadge.style.minWidth = '42px';
    _matchCountBadge.style.textAlign = 'center';
    _matchCountBadge.textContent = '0/0';

    final prevBtn = _createToolbarBtn('▲', 'Previous match (Shift+Enter)', () {
      controller.findPrevious();
    });
    final nextBtn = _createToolbarBtn('▼', 'Next match (Enter)', () {
      controller.findNext();
    });
    final closeBtn = _createToolbarBtn('×', 'Close Find/Replace (Esc)', () {
      _closeFindReplace();
    });

    row1.appendChild(_searchInput);
    row1.appendChild(_matchCountBadge);
    row1.appendChild(prevBtn);
    row1.appendChild(nextBtn);
    row1.appendChild(closeBtn);

    final row2 = web.document.createElement('div') as web.HTMLDivElement;
    row2.style.display = 'flex';
    row2.style.alignItems = 'center';
    row2.style.gap = '6px';

    _replaceInput = web.document.createElement('input') as web.HTMLInputElement;
    _replaceInput.type = 'text';
    _replaceInput.placeholder = 'Replace with...';
    _replaceInput.style.backgroundColor = '#1e1e2e';
    _replaceInput.style.border = '1px solid #313244';
    _replaceInput.style.borderRadius = '4px';
    _replaceInput.style.color = '#cdd6f4';
    _replaceInput.style.padding = '4px 8px';
    _replaceInput.style.fontSize = '12px';
    _replaceInput.style.outline = 'none';
    _replaceInput.style.width = '160px';

    final replaceBtn = _createTextBtn('Replace', 'Replace active match', () {
      controller.replaceCurrent(_replaceInput.value);
    });
    final replaceAllBtn = _createTextBtn(
      'Replace All',
      'Replace all occurrences',
      () {
        controller.replaceAll(_replaceInput.value);
      },
    );

    row2.appendChild(_replaceInput);
    row2.appendChild(replaceBtn);
    row2.appendChild(replaceAllBtn);

    _findReplacePanel.appendChild(row1);
    _findReplacePanel.appendChild(row2);

    _searchInput.addEventListener(
      'input',
      ((web.Event e) {
        _updateSearch();
      }.toJS),
    );

    _searchInput.addEventListener(
      'keydown',
      ((web.KeyboardEvent e) {
        if (e.key == 'Enter') {
          e.preventDefault();
          if (e.shiftKey) {
            controller.findPrevious();
          } else {
            controller.findNext();
          }
        } else if (e.key == 'Escape') {
          _closeFindReplace();
        }
      }.toJS),
    );
  }

  void _updateSearch() {
    final query = _searchInput.value;
    if (query.isEmpty) {
      controller.findReplace.clear();
      _matchCountBadge.textContent = '0/0';
      render();
      return;
    }
    controller.find(SearchOptions(query: query));
    final count = controller.findReplace.matchCount;
    final active = controller.findReplace.activeMatchIndex + 1;
    _matchCountBadge.textContent = count > 0 ? '$active/$count' : '0/0';
    render();
  }

  void _openFindReplace() {
    _isFindOpen = true;
    _findReplacePanel.style.display = 'flex';
    _searchInput.focus();
    _searchInput.select();
  }

  void _closeFindReplace() {
    _isFindOpen = false;
    _findReplacePanel.style.display = 'none';
    controller.findReplace.clear();
    focus();
    render();
  }

  web.HTMLElement _createToolbarBtn(
    String label,
    String tooltip,
    void Function() onClick,
  ) {
    final btn = web.document.createElement('button') as web.HTMLElement;
    btn.textContent = label;
    btn.title = tooltip;
    btn.style.backgroundColor = '#313244';
    btn.style.border = 'none';
    btn.style.borderRadius = '4px';
    btn.style.color = '#cdd6f4';
    btn.style.padding = '3px 7px';
    btn.style.cursor = 'pointer';
    btn.style.fontSize = '11px';
    btn.addEventListener(
      'click',
      ((web.Event e) {
        e.stopPropagation();
        onClick();
      }.toJS),
    );
    return btn;
  }

  web.HTMLElement _createTextBtn(
    String label,
    String tooltip,
    void Function() onClick,
  ) {
    final btn = web.document.createElement('button') as web.HTMLElement;
    btn.textContent = label;
    btn.title = tooltip;
    btn.style.backgroundColor = '#89b4fa';
    btn.style.color = '#11111b';
    btn.style.border = 'none';
    btn.style.borderRadius = '4px';
    btn.style.padding = '3px 8px';
    btn.style.fontWeight = 'bold';
    btn.style.cursor = 'pointer';
    btn.style.fontSize = '11px';
    btn.addEventListener(
      'click',
      ((web.Event e) {
        e.stopPropagation();
        onClick();
      }.toJS),
    );
    return btn;
  }

  void _buildCompletionMenu() {
    _completionMenu = web.document.createElement('div') as web.HTMLDivElement;
    _completionMenu.className = 'editor-completion-menu';
    _completionMenu.style.position = 'absolute';
    _completionMenu.style.zIndex = '200';
    _completionMenu.style.backgroundColor = '#181825';
    _completionMenu.style.border = '1px solid #45475a';
    _completionMenu.style.borderRadius = '6px';
    _completionMenu.style.boxShadow = '0 6px 20px rgba(0,0,0,0.5)';
    _completionMenu.style.maxHeight = '220px';
    _completionMenu.style.overflowY = 'auto';
    _completionMenu.style.display = 'none';
    _completionMenu.style.minWidth = '240px';
  }

  void _buildHoverCard() {
    _hoverCard = web.document.createElement('div') as web.HTMLDivElement;
    _hoverCard.className = 'editor-hover-card';
    _hoverCard.style.position = 'absolute';
    _hoverCard.style.zIndex = '180';
    _hoverCard.style.backgroundColor = '#181825';
    _hoverCard.style.border = '1px solid #585b70';
    _hoverCard.style.borderRadius = '6px';
    _hoverCard.style.padding = '8px 12px';
    _hoverCard.style.boxShadow = '0 6px 18px rgba(0,0,0,0.5)';
    _hoverCard.style.maxWidth = '360px';
    _hoverCard.style.display = 'none';
    _hoverCard.style.pointerEvents = 'none';
  }

  /// Trigger rich autocomplete suggestions menu at cursor position.
  void showCompletions() {
    final sel = controller.selection.extent;
    final lineText = controller.buffer.getLine(sel.line);
    final wordStart = SelectionModel.getWordStartColumn(lineText, sel.column);
    final prefix = lineText.substring(wordStart, sel.column).toLowerCase();

    // Determine completion items based on context (dot after NDArray vs general words)
    final items = <_CompletionItem>[];

    if (wordStart > 0 && lineText[wordStart - 1] == '.') {
      // NDArray / object property & method suggestions
      items.addAll([
        _CompletionItem(
          'shape',
          'List<int>',
          'Dimensions of the NDArray',
          'Property',
        ),
        _CompletionItem('dtype', 'DType', 'Data type of elements', 'Property'),
        _CompletionItem(
          'strides',
          'List<int>',
          'Step sizes per dimension',
          'Property',
        ),
        _CompletionItem('scalar', 'T', 'Single 0D scalar value', 'Property'),
        _CompletionItem(
          'copy()',
          'NDArray<T>',
          'Create a deep decoupled copy',
          'Method',
        ),
        _CompletionItem(
          'fill(value)',
          'void',
          'Fill array with value',
          'Method',
        ),
        _CompletionItem(
          'getCell([coords])',
          'T',
          'Get element at multi-dim indices',
          'Method',
        ),
        _CompletionItem(
          'setCell([coords], v)',
          'void',
          'Set element at indices',
          'Method',
        ),
        _CompletionItem(
          'reshape([newShape])',
          'NDArray<T>',
          'Reshape array view',
          'Method',
        ),
        _CompletionItem(
          'broadcastTo([shape])',
          'NDArray<T>',
          'Broadcast array to target shape',
          'Method',
        ),
        _CompletionItem(
          'transpose()',
          'NDArray<T>',
          'Reverse axis order',
          'Method',
        ),
        _CompletionItem(
          'scope(...)',
          'NDArray<T>',
          'Attach array to memory scope',
          'Method',
        ),
      ]);
    } else {
      // Built-in types, keywords, and snippets
      items.addAll([
        _CompletionItem(
          'NDArray.create([shape], DType.float64)',
          'NDArray',
          'Create new uninitialized array',
          'Class',
        ),
        _CompletionItem(
          'NDArray.fromList([data], [shape], DType.float64)',
          'NDArray',
          'Create array from Dart list',
          'Class',
        ),
        _CompletionItem(
          'DType.float64',
          'DType',
          '64-bit IEEE double float',
          'Enum',
        ),
        _CompletionItem('DType.float32', 'DType', '32-bit IEEE float', 'Enum'),
        _CompletionItem(
          'DType.int32',
          'DType',
          '32-bit signed integer',
          'Enum',
        ),
        _CompletionItem(
          'ScratchArena.scope((arena) => ...)',
          'ScratchArena',
          'Scoped temporary allocation arena',
          'Class',
        ),
        _CompletionItem(
          'Complex(real, imag)',
          'Complex',
          'Complex number with real & imag',
          'Class',
        ),
        _CompletionItem(
          'for (var i = 0; i < n; i++)',
          'Snippet',
          'Standard for loop',
          'Snippet',
        ),
        _CompletionItem(
          'if (cond) { ... }',
          'Snippet',
          'Conditional block',
          'Snippet',
        ),
      ]);
    }

    _currentCompletions = items
        .where((item) => item.label.toLowerCase().contains(prefix))
        .toList();

    if (_currentCompletions.isEmpty) {
      _closeCompletions();
      return;
    }

    _selectedCompletionIndex = 0;
    _isCompletionOpen = true;

    final charWidth = _getCharWidth();
    final left = 60 + sel.column * charWidth;
    final top = (sel.line + 1) * controller.options.lineHeight;

    _completionMenu.style.left = '${left}px';
    _completionMenu.style.top = '${top}px';
    _completionMenu.style.display = 'block';

    _renderCompletionMenu();
  }

  void _renderCompletionMenu() {
    _completionMenu.innerHTML = ''.toJS;

    for (int i = 0; i < _currentCompletions.length; i++) {
      final item = _currentCompletions[i];
      final itemEl = web.document.createElement('div') as web.HTMLDivElement;
      itemEl.style.padding = '6px 10px';
      itemEl.style.display = 'flex';
      itemEl.style.justifyContent = 'space-between';
      itemEl.style.alignItems = 'center';
      itemEl.style.cursor = 'pointer';
      itemEl.style.fontSize = '12px';
      itemEl.style.fontFamily = controller.options.fontFamily;

      if (i == _selectedCompletionIndex) {
        itemEl.style.backgroundColor = '#313244';
        itemEl.style.color = '#cdd6f4';
      } else {
        itemEl.style.backgroundColor = 'transparent';
        itemEl.style.color = '#a6adc8';
      }

      final labelSpan = web.document.createElement('span') as web.HTMLElement;
      labelSpan.style.fontWeight = 'bold';
      labelSpan.textContent = item.label;

      final badgeSpan = web.document.createElement('span') as web.HTMLElement;
      badgeSpan.style.fontSize = '10px';
      badgeSpan.style.color = '#89b4fa';
      badgeSpan.style.marginLeft = '12px';
      badgeSpan.textContent = item.detail;

      itemEl.appendChild(labelSpan);
      itemEl.appendChild(badgeSpan);

      itemEl.addEventListener(
        'click',
        ((web.Event e) {
          e.stopPropagation();
          _selectedCompletionIndex = i;
          _commitCompletion();
        }.toJS),
      );

      _completionMenu.appendChild(itemEl);
    }
  }

  void _commitCompletion() {
    if (_currentCompletions.isEmpty) return;
    final item = _currentCompletions[_selectedCompletionIndex];

    final sel = controller.selection.extent;
    final lineText = controller.buffer.getLine(sel.line);
    final wordStart = SelectionModel.getWordStartColumn(lineText, sel.column);
    final prefixLen = sel.column - wordStart;
    if (prefixLen > 0) {
      controller.selection = TextSelection(
        base: TextPosition(sel.line, wordStart),
        extent: sel,
      );
    }

    // Extract identifier before parens/brackets
    final insertStr = item.label.split('(').first;
    controller.insertText(insertStr);
    _closeCompletions();
  }

  void _closeCompletions() {
    _isCompletionOpen = false;
    _completionMenu.style.display = 'none';
  }

  /// Shows hover documentation popup card for hovered line & column.
  void showHoverCardAt(int line, int column) {
    if (line < 0 || line >= controller.lineCount) return;
    final lineText = controller.buffer.getLine(line);
    final wordStart = SelectionModel.getWordStartColumn(lineText, column);
    final wordEnd = SelectionModel.getWordEndColumn(lineText, column);

    if (wordStart >= wordEnd) {
      _hoverCard.style.display = 'none';
      return;
    }

    final word = lineText.substring(wordStart, wordEnd);
    final doc = _getHoverDocumentation(word);
    if (doc == null) {
      _hoverCard.style.display = 'none';
      return;
    }

    _hoverCard.innerHTML = doc.toJS;
    final charWidth = _getCharWidth();
    final left = 60 + wordStart * charWidth;
    final top = (line + 1) * controller.options.lineHeight + 4;

    _hoverCard.style.left = '${left}px';
    _hoverCard.style.top = '${top}px';
    _hoverCard.style.display = 'block';
  }

  String? _getHoverDocumentation(String word) {
    switch (word) {
      case 'NDArray':
        return '<div style="font-weight:bold;color:#89b4fa;">class NDArray&lt;T&gt;</div>'
            '<div style="font-size:11px;color:#a6adc8;margin-top:4px;">'
            'N-dimensional array with C-contiguous or strided memory layouts, SIMD operations, and automatic scope management.</div>';
      case 'DType':
        return '<div style="font-weight:bold;color:#f9e2af;">enum DType</div>'
            '<div style="font-size:11px;color:#a6adc8;margin-top:4px;">'
            'Data types supported by NDArray (float64, float32, int32, int64, complex64, complex128, boolean).</div>';
      case 'shape':
        return '<div style="font-weight:bold;color:#89dceb;">List&lt;int&gt; get shape</div>'
            '<div style="font-size:11px;color:#a6adc8;margin-top:4px;">'
            'Returns the dimension sizes of this NDArray.</div>';
      case 'copy':
        return '<div style="font-weight:bold;color:#a6e3a1;">NDArray&lt;T&gt; copy({NDArray&lt;T&gt;? out})</div>'
            '<div style="font-size:11px;color:#a6adc8;margin-top:4px;">'
            'Creates a deep copy of the array elements into new memory or into optional [out] array.</div>';
      case 'ScratchArena':
        return '<div style="font-weight:bold;color:#cba6f7;">class ScratchArena</div>'
            '<div style="font-size:11px;color:#a6adc8;margin-top:4px;">'
            'High-speed temporary native memory allocation arena for NDArray operations.</div>';
      default:
        return null;
    }
  }

  void _bindEvents() {
    _root.addEventListener(
      'click',
      ((web.MouseEvent e) {
        focus();
      }.toJS),
    );

    _linesLayer.addEventListener(
      'mousedown',
      ((web.MouseEvent e) {
        _isDragging = true;
        focus();
        final pos = _calculatePositionFromMouse(e);
        if (e.altKey) {
          controller.addSecondaryCaret(pos);
          render();
          return;
        }
        if (e.detail == 2) {
          // Double click word selection
          controller.selectWordAt(pos);
        } else if (e.detail == 3) {
          // Triple click line selection
          controller.selectLineAt(pos.line);
        } else {
          if (e.shiftKey) {
            controller.selection = TextSelection(
              base: controller.selection.base,
              extent: pos,
            );
          } else {
            controller.selection = TextSelection.collapsed(pos);
          }
        }
      }.toJS),
    );

    web.window.addEventListener(
      'mousemove',
      ((web.MouseEvent e) {
        if (_isDragging) {
          final pos = _calculatePositionFromMouse(e);
          controller.selection = TextSelection(
            base: controller.selection.base,
            extent: pos,
          );
        }
      }.toJS),
    );

    web.window.addEventListener(
      'mouseup',
      ((web.MouseEvent e) {
        _isDragging = false;
      }.toJS),
    );

    _gutter.addEventListener(
      'mousedown',
      ((web.MouseEvent e) {
        focus();
        final target = e.target as web.HTMLElement?;
        final pos = _calculatePositionFromMouse(e);
        if (target != null && target.classList.contains('fold-btn')) {
          controller.foldingManager.toggleFold(pos.line);
          render();
          return;
        }
        controller.selectLineAt(pos.line);
      }.toJS),
    );

    _linesLayer.addEventListener(
      'mousemove',
      ((web.MouseEvent e) {
        _hoverTimer?.cancel();
        _hoverTimer = Timer(const Duration(milliseconds: 400), () {
          final pos = _calculatePositionFromMouse(e);
          showHoverCardAt(pos.line, pos.column);
        });
      }.toJS),
    );

    _linesLayer.addEventListener(
      'mouseleave',
      ((web.MouseEvent e) {
        _hoverTimer?.cancel();
        _hoverCard.style.display = 'none';
      }.toJS),
    );

    _hiddenInput.addEventListener(
      'focus',
      ((web.Event e) {
        _isFocused = true;
        _root.style.borderColor = '#89b4fa';
        render();
      }.toJS),
    );

    _hiddenInput.addEventListener(
      'blur',
      ((web.Event e) {
        _isFocused = false;
        _root.style.borderColor = '#313244';
        render();
      }.toJS),
    );

    _hiddenInput.addEventListener(
      'keydown',
      ((web.KeyboardEvent e) {
        final key = e.key;
        final ctrlOrMeta = e.ctrlKey || e.metaKey;

        if (_isCompletionOpen) {
          if (key == 'ArrowDown') {
            e.preventDefault();
            _selectedCompletionIndex =
                (_selectedCompletionIndex + 1) % _currentCompletions.length;
            _renderCompletionMenu();
            return;
          } else if (key == 'ArrowUp') {
            e.preventDefault();
            _selectedCompletionIndex =
                (_selectedCompletionIndex - 1 + _currentCompletions.length) %
                _currentCompletions.length;
            _renderCompletionMenu();
            return;
          } else if (key == 'Enter' || key == 'Tab') {
            e.preventDefault();
            _commitCompletion();
            return;
          } else if (key == 'Escape') {
            e.preventDefault();
            _closeCompletions();
            return;
          }
        }

        if (key == 'Escape') {
          _closeFindReplace();
          if (controller.selectionModel.selections.length > 1) {
            controller.selection = TextSelection.collapsed(
              controller.selection.extent,
            );
          }
          return;
        }

        if (ctrlOrMeta && (key == 'd' || key == 'D')) {
          e.preventDefault();
          controller.selectNextOccurrence();
          return;
        }

        if ((e.shiftKey && e.altKey && (key == 'f' || key == 'F')) ||
            (ctrlOrMeta && e.shiftKey && (key == 'i' || key == 'I')) ||
            (ctrlOrMeta && e.altKey && (key == 'l' || key == 'L'))) {
          e.preventDefault();
          controller.formatDocument();
          return;
        }

        if (e.shiftKey && key == 'Enter') {
          e.preventDefault();
          onExecute?.call();
          return;
        }

        if (ctrlOrMeta &&
            (key == 'f' || key == 'F' || key == 'h' || key == 'H')) {
          e.preventDefault();
          _openFindReplace();
          return;
        }

        if (ctrlOrMeta && key == ' ') {
          e.preventDefault();
          showCompletions();
          final sel = controller.selection.extent;
          final offset = controller.buffer.getLineOffset(sel.line) + sel.column;
          final rect = _root.getBoundingClientRect();
          final charWidth = _getCharWidth();
          final x = (rect.left + 56 + sel.column * charWidth).toInt();
          final y = (rect.top + (sel.line + 1) * controller.options.lineHeight)
              .toInt();
          onCompletionRequested?.call(offset, x, y);
          return;
        }

        if (ctrlOrMeta && (key == 'z' || key == 'Z')) {
          e.preventDefault();
          if (e.shiftKey) {
            controller.redo();
          } else {
            controller.undo();
          }
          return;
        }

        if (ctrlOrMeta && (key == 'y' || key == 'Y')) {
          e.preventDefault();
          controller.redo();
          return;
        }

        if (ctrlOrMeta && (key == 'a' || key == 'A')) {
          e.preventDefault();
          controller.selectAll();
          return;
        }

        if (ctrlOrMeta && (key == '/' || key == '?')) {
          e.preventDefault();
          controller.toggleLineComment();
          return;
        }

        if (e.altKey && (key == 'ArrowUp')) {
          e.preventDefault();
          controller.moveLinesUp();
          return;
        }

        if (e.altKey && (key == 'ArrowDown')) {
          e.preventDefault();
          if (e.shiftKey) {
            controller.duplicateLinesDown();
          } else {
            controller.moveLinesDown();
          }
          return;
        }

        if (ctrlOrMeta && e.shiftKey && (key == 'k' || key == 'K')) {
          e.preventDefault();
          controller.deleteLines();
          return;
        }

        switch (key) {
          case 'ArrowLeft':
            e.preventDefault();
            final movement = e.altKey
                ? CursorMovement.wordLeft
                : (ctrlOrMeta ? CursorMovement.lineStart : CursorMovement.left);
            controller.moveCursor(movement, select: e.shiftKey);
            break;
          case 'ArrowRight':
            e.preventDefault();
            final movement = e.altKey
                ? CursorMovement.wordRight
                : (ctrlOrMeta ? CursorMovement.lineEnd : CursorMovement.right);
            controller.moveCursor(movement, select: e.shiftKey);
            break;
          case 'ArrowUp':
            e.preventDefault();
            final movement = ctrlOrMeta
                ? CursorMovement.documentStart
                : CursorMovement.up;
            controller.moveCursor(movement, select: e.shiftKey);
            break;
          case 'ArrowDown':
            e.preventDefault();
            final movement = ctrlOrMeta
                ? CursorMovement.documentEnd
                : CursorMovement.down;
            controller.moveCursor(movement, select: e.shiftKey);
            break;
          case 'Home':
            e.preventDefault();
            controller.moveCursor(
              ctrlOrMeta
                  ? CursorMovement.documentStart
                  : CursorMovement.lineStart,
              select: e.shiftKey,
            );
            break;
          case 'End':
            e.preventDefault();
            controller.moveCursor(
              ctrlOrMeta ? CursorMovement.documentEnd : CursorMovement.lineEnd,
              select: e.shiftKey,
            );
            break;
          case 'PageUp':
            e.preventDefault();
            controller.moveCursor(CursorMovement.pageUp, select: e.shiftKey);
            break;
          case 'PageDown':
            e.preventDefault();
            controller.moveCursor(CursorMovement.pageDown, select: e.shiftKey);
            break;
          case 'Backspace':
            e.preventDefault();
            controller.deleteBackward();
            break;
          case 'Delete':
            e.preventDefault();
            controller.deleteForward();
            break;
          case 'Enter':
            e.preventDefault();
            controller.insertNewline();
            break;
          case 'Tab':
            e.preventDefault();
            if (e.shiftKey) {
              controller.shiftTabPressed();
            } else {
              controller.tabPressed();
            }
            break;
        }
      }.toJS),
    );

    _hiddenInput.addEventListener(
      'input',
      ((web.Event e) {
        final val = _hiddenInput.value;
        if (val.isNotEmpty) {
          controller.insertText(val);
          _hiddenInput.value = '';
        }
      }.toJS),
    );
  }

  /// Focuses the editor input.
  void focus() {
    _hiddenInput.focus();
    _isFocused = true;
  }

  double _getCharWidth() {
    return controller.options.fontSize * 0.60;
  }

  TextPosition _calculatePositionFromMouse(web.MouseEvent e) {
    final rect = _linesLayer.getBoundingClientRect();
    final clickY = e.clientY - rect.top;
    final clickX = e.clientX - rect.left;

    final lineIndex = (clickY / controller.options.lineHeight).floor().clamp(
      0,
      controller.lineCount - 1,
    );
    final charWidth = _getCharWidth();
    final lineLen = controller.buffer.getLineLength(lineIndex);
    final colIndex = (clickX / charWidth).round().clamp(0, lineLen);

    return TextPosition(lineIndex, colIndex);
  }

  /// Re-renders all editor DOM layers.
  void render() {
    final lineCount = controller.lineCount;
    final lineHeight = controller.options.lineHeight;
    final charWidth = _getCharWidth();
    final totalHeight = math.max(100.0, lineCount * lineHeight + 16.0);
    _root.style.height = '${totalHeight}px';

    // 1. Render Gutter with Line Numbers, Fold Markers, & Diagnostic Badges
    final gutterHtml = StringBuffer();
    for (int i = 0; i < lineCount; i++) {
      if (controller.foldingManager.isLineHidden(i)) continue;

      final isFoldHeader = controller.foldingManager.isFoldHeader(i);
      final foldRegion = controller.foldingManager.getRegionAt(i);
      final foldIcon = isFoldHeader
          ? (foldRegion?.isCollapsed == true ? '▶' : '▼')
          : '';

      final hasError = controller.diagnostics.any((d) => d.line == i);
      final errBadge = hasError
          ? '<span style="color:#f38ba8; font-size:10px; margin-right:4px;" title="Diagnostic Error">●</span>'
          : '';

      gutterHtml.write(
        '<div style="height: ${lineHeight}px; line-height: ${lineHeight}px; display: flex; justify-content: space-between; align-items: center;" '
        'data-line="$i">'
        '<span style="cursor: pointer; color: #89b4fa; font-size: 9px; user-select: none;" class="fold-btn">$foldIcon</span>'
        '<span>$errBadge${i + 1}</span>'
        '</div>',
      );
    }
    _gutter.innerHTML = gutterHtml.toString().toJS;

    // 2. Render Active Line Highlight
    final primarySel = controller.selection;
    if (controller.options.highlightActiveLine && _isFocused) {
      final activeTop = primarySel.extent.line * lineHeight;
      _activeLineLayer.innerHTML =
          '<div style="position: absolute; top: ${activeTop}px; height: ${lineHeight}px; left: 0px; right: 0px; background-color: rgba(255, 255, 255, 0.04);"></div>'
              .toJS;
    } else {
      _activeLineLayer.innerHTML = ''.toJS;
    }

    // 3. Render Selections, Search Matches, & Diagnostic Squiggles
    final selHtml = StringBuffer();

    // All active selections
    for (final sel in controller.selectionModel.selections) {
      if (sel.isCollapsed) continue;
      final start = sel.start;
      final end = sel.end;

      for (int l = start.line; l <= end.line; l++) {
        if (controller.foldingManager.isLineHidden(l)) continue;
        final lineLen = controller.buffer.getLineLength(l);
        final startCol = (l == start.line) ? start.column : 0;
        final endCol = (l == end.line) ? end.column : lineLen;
        final top = l * lineHeight;
        final left = startCol * charWidth;
        final width = math.max(2.0, (endCol - startCol) * charWidth);

        selHtml.write(
          '<div style="position: absolute; top: ${top}px; left: ${left}px; width: ${width}px; height: ${lineHeight}px; background-color: rgba(137, 180, 250, 0.28); border-radius: 2px;"></div>',
        );
      }
    }

    // Highlight search matches
    for (int idx = 0; idx < controller.findReplace.matches.length; idx++) {
      final m = controller.findReplace.matches[idx];
      final l = m.range.start.line;
      if (controller.foldingManager.isLineHidden(l)) continue;
      final top = l * lineHeight;
      final left = m.range.start.column * charWidth;
      final width = math.max(
        4.0,
        (m.range.end.column - m.range.start.column) * charWidth,
      );
      final isActiveMatch = idx == controller.findReplace.activeMatchIndex;
      final borderColor = isActiveMatch ? '#f9e2af' : '#fab387';
      final bgColor = isActiveMatch
          ? 'rgba(249, 226, 175, 0.25)'
          : 'rgba(250, 179, 135, 0.15)';

      selHtml.write(
        '<div style="position: absolute; top: ${top}px; left: ${left}px; width: ${width}px; height: ${lineHeight}px; background-color: $bgColor; border: 1px solid $borderColor; border-radius: 2px; box-sizing: border-box;"></div>',
      );
    }

    // Render diagnostic squiggles
    for (final diag in controller.diagnostics) {
      if (controller.foldingManager.isLineHidden(diag.line)) continue;
      final top = diag.line * lineHeight + lineHeight - 3;
      final left = diag.startColumn * charWidth;
      final width = math.max(
        6.0,
        (diag.endColumn - diag.startColumn) * charWidth,
      );

      selHtml.write(
        '<div style="position: absolute; top: ${top}px; left: ${left}px; width: ${width}px; height: 2px; background-color: ${diag.colorHex}; box-shadow: 0 1px 2px rgba(0,0,0,0.5);"></div>',
      );
    }

    _selectionsLayer.innerHTML = selHtml.toString().toJS;

    // 4. Render Lines Text & Syntax Tokens
    final linesHtml = StringBuffer();
    final lineTokens = controller.lineTokens;

    for (int l = 0; l < lineCount; l++) {
      if (controller.foldingManager.isLineHidden(l)) continue;

      final top = l * lineHeight;
      final start = controller.buffer.getLineOffset(l);
      final len = controller.buffer.getLineLength(l);
      final lineStr = controller.buffer.getTextInRange(start, len);
      final tokens = (l < lineTokens.length) ? lineTokens[l] : <SyntaxToken>[];

      linesHtml.write(
        '<div style="position: absolute; top: ${top}px; height: ${lineHeight}px; line-height: ${lineHeight}px;">',
      );
      if (tokens.isEmpty) {
        linesHtml.write(_escapeHtml(lineStr.isEmpty ? ' ' : lineStr));
      } else {
        for (final t in tokens) {
          final color = _getScopeColor(t.type, t.text);
          final fontWeight = (t.type == TokenType.keyword) ? 'bold' : 'normal';
          linesHtml.write(
            '<span style="color: $color; font-weight: $fontWeight;">${_escapeHtml(t.text)}</span>',
          );
        }
      }

      // If this line is a fold header that is collapsed, append a visual badge
      final foldReg = controller.foldingManager.getRegionAt(l);
      if (foldReg != null && foldReg.isCollapsed) {
        linesHtml.write(
          '<span style="background-color: #313244; color: #a6adc8; padding: 0 4px; border-radius: 3px; font-size: 11px; margin-left: 6px; user-select: none;">... ${foldReg.endLine - foldReg.startLine} lines folded</span>',
        );
      }

      linesHtml.write('</div>');
    }
    _linesLayer.innerHTML = linesHtml.toString().toJS;

    // 5. Render Multi-Carets
    final caretsHtml = StringBuffer();
    if (_isFocused) {
      for (final sel in controller.selectionModel.selections) {
        if (!sel.isCollapsed) continue;
        final caretPos = sel.extent;
        if (controller.foldingManager.isLineHidden(caretPos.line)) continue;

        final caretTop = caretPos.line * lineHeight + 2;
        final caretLeft = caretPos.column * charWidth;
        caretsHtml.write(
          '<div style="position: absolute; top: ${caretTop}px; left: ${caretLeft}px; width: 2px; height: ${lineHeight - 4}px; background-color: #f5e0dc; animation: caret-blink 1s infinite;"></div>',
        );
      }
    }
    _caretsLayer.innerHTML = caretsHtml.toString().toJS;
  }

  String _escapeHtml(String text) {
    return text
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
  }

  String _getScopeColor(TokenType type, String text) {
    switch (type) {
      case TokenType.keyword:
        return '#cba6f7'; // Mauve / Purple
      case TokenType.identifier:
        if (RegExp(r'^[A-Z]').hasMatch(text)) {
          return '#f9e2af'; // Type / Class Gold
        }
        return '#89b4fa'; // Identifier Soft Blue
      case TokenType.number:
        return '#fab387'; // Peach / Orange
      case TokenType.string:
        return '#a6e3a1'; // Pastel Green
      case TokenType.comment:
        return '#6c7086'; // Muted Gray
      case TokenType.operator:
        return '#89dceb'; // Cyan / Teal
      case TokenType.punctuation:
        return '#9399b2'; // Lavender Gray
      default:
        return '#cdd6f4'; // Main Text
    }
  }
}

class _CompletionItem {
  final String label;
  final String detail;
  final String doc;
  final String kind;

  _CompletionItem(this.label, this.detail, this.doc, this.kind);
}
