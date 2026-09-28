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

/// Completion Popup Data Models for active completion dropdown UI overlays.
library;

/// Represents an individual item entry in a completion dropdown menu.
final class CompletionDropdownItem {
  /// The display label for the completion item.
  final String label;

  /// Optional detail string (e.g. signature, return type, or module path).
  final String? detail;

  /// Optional documentation string or markdown snippet.
  final String? documentation;

  /// LSP CompletionItemKind or human-readable category string (e.g. 'Function', 'Variable').
  final String? kind;

  /// Text to be inserted when this completion item is selected.
  final String insertText;

  /// Creates a new [CompletionDropdownItem].
  const CompletionDropdownItem({
    required this.label,
    this.detail,
    this.documentation,
    this.kind,
    String? insertText,
  }) : insertText = insertText ?? label;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is CompletionDropdownItem &&
          runtimeType == other.runtimeType &&
          label == other.label &&
          detail == other.detail &&
          documentation == other.documentation &&
          kind == other.kind &&
          insertText == other.insertText;

  @override
  int get hashCode =>
      Object.hash(label, detail, documentation, kind, insertText);
}

/// Represents the active state and layout model of an LSP completion dropdown overlay.
final class CompletionPopupModel {
  /// The X position (column index or pixel offset) of the caret anchor.
  final double x;

  /// The Y position (line index or pixel offset) of the caret anchor.
  final double y;

  /// List of completion items available in the popup.
  final List<CompletionDropdownItem> items;

  /// Index of the currently highlighted/selected completion item.
  final int selectedIndex;

  /// Search or trigger prefix typed by the user to filter completions.
  final String searchPrefix;

  /// Whether the popup menu is currently visible.
  final bool isVisible;

  /// Creates a new [CompletionPopupModel].
  const CompletionPopupModel({
    required this.x,
    required this.y,
    required this.items,
    this.selectedIndex = 0,
    this.searchPrefix = '',
    this.isVisible = true,
  });

  /// The currently selected item from filtered items, or `null` if empty or out of range.
  CompletionDropdownItem? get selectedItem {
    final list = filteredItems;
    return (list.isNotEmpty &&
            selectedIndex >= 0 &&
            selectedIndex < list.length)
        ? list[selectedIndex]
        : null;
  }

  /// Filtered items matching [searchPrefix].
  List<CompletionDropdownItem> get filteredItems {
    if (searchPrefix.isEmpty) return items;
    final prefixLower = searchPrefix.toLowerCase();
    return items
        .where((item) => item.label.toLowerCase().contains(prefixLower))
        .toList();
  }

  /// Creates a copy of this [CompletionPopupModel] with updated fields.
  CompletionPopupModel copyWith({
    double? x,
    double? y,
    List<CompletionDropdownItem>? items,
    int? selectedIndex,
    String? searchPrefix,
    bool? isVisible,
  }) {
    return CompletionPopupModel(
      x: x ?? this.x,
      y: y ?? this.y,
      items: items ?? this.items,
      selectedIndex: selectedIndex ?? this.selectedIndex,
      searchPrefix: searchPrefix ?? this.searchPrefix,
      isVisible: isVisible ?? this.isVisible,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is CompletionPopupModel &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y &&
          selectedIndex == other.selectedIndex &&
          searchPrefix == other.searchPrefix &&
          isVisible == other.isVisible &&
          _listEquals(items, other.items);

  @override
  int get hashCode => Object.hash(
    x,
    y,
    Object.hashAll(items),
    selectedIndex,
    searchPrefix,
    isVisible,
  );

  static bool _listEquals(
    List<CompletionDropdownItem> a,
    List<CompletionDropdownItem> b,
  ) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
