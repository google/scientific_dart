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

/// Hover Tooltip Data Model for LSP hover tooltips UI overlay.
library;

/// Represents an LSP markdown hover tooltip overlay anchored at text coordinates.
final class HoverTooltipModel {
  /// The X position (column index or pixel offset) of the hover target.
  final double x;

  /// The Y position (line index or pixel offset) of the hover target.
  final double y;

  /// Raw markdown string content of the tooltip.
  final String markdownContent;

  /// Optional signature or code header block (e.g. function signature or type signature).
  final String? signature;

  /// Whether the tooltip is currently visible.
  final bool isVisible;

  /// Creates a new [HoverTooltipModel].
  const HoverTooltipModel({
    required this.x,
    required this.y,
    required this.markdownContent,
    this.signature,
    this.isVisible = true,
  });

  /// Creates a copy of this model with modified parameters.
  HoverTooltipModel copyWith({
    double? x,
    double? y,
    String? markdownContent,
    String? signature,
    bool? isVisible,
  }) {
    return HoverTooltipModel(
      x: x ?? this.x,
      y: y ?? this.y,
      markdownContent: markdownContent ?? this.markdownContent,
      signature: signature ?? this.signature,
      isVisible: isVisible ?? this.isVisible,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is HoverTooltipModel &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y &&
          markdownContent == other.markdownContent &&
          signature == other.signature &&
          isVisible == other.isVisible;

  @override
  int get hashCode => Object.hash(x, y, markdownContent, signature, isVisible);
}
