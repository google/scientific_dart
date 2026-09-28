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

import 'text_position.dart';

/// Immutable representation of a text selection or cursor position.
class Selection {
  /// The position at which the selection originated (anchor / base).
  final TextPosition anchor;

  /// The position at which the selection currently terminates (position / extent).
  final TextPosition position;

  /// The affinity of the selection.
  final TextAffinity affinity;

  /// Creates a [Selection] with positional [anchor], optional [position], and optional [affinity].
  const Selection(
    this.anchor, [
    TextPosition? position,
    this.affinity = TextAffinity.downstream,
  ]) : position = position ?? anchor;

  /// Creates a [Selection] with named [base] and [extent] positions.
  const Selection.range({
    required TextPosition base,
    required TextPosition extent,
    this.affinity = TextAffinity.downstream,
  }) : anchor = base,
       position = extent;

  /// Creates a collapsed selection (caret position) at [pos].
  const Selection.collapsed(
    TextPosition pos, {
    this.affinity = TextAffinity.downstream,
  }) : anchor = pos,
       position = pos;

  /// Base position getter.
  TextPosition get base => anchor;

  /// Extent position getter.
  TextPosition get extent => position;

  /// Whether the selection is collapsed to a single caret position.
  bool get isCollapsed => anchor == position;

  /// Whether the selection is reversed (position comes before anchor).
  bool get isReversed => position.compareTo(anchor) < 0;

  /// The starting position of the selection (inclusive).
  TextPosition get start => isReversed ? position : anchor;

  /// The ending position of the selection (exclusive).
  TextPosition get end => isReversed ? anchor : position;

  /// Returns a new [Selection] with updated fields.
  Selection copyWith({
    TextPosition? anchor,
    TextPosition? position,
    TextPosition? base,
    TextPosition? extent,
    TextAffinity? affinity,
  }) {
    return Selection.range(
      base: base ?? anchor ?? this.anchor,
      extent: extent ?? position ?? this.position,
      affinity: affinity ?? this.affinity,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Selection &&
          runtimeType == other.runtimeType &&
          anchor == other.anchor &&
          position == other.position &&
          affinity == other.affinity;

  @override
  int get hashCode => Object.hash(anchor, position, affinity);

  @override
  String toString() => 'Selection(base: $anchor, extent: $position)';
}

/// Representation of a text selection using named parameters.
class TextSelection extends Selection {
  /// Creates a [TextSelection] with named [base] and [extent] positions.
  const TextSelection({
    required super.base,
    required super.extent,
    super.affinity,
  }) : super.range();

  /// Creates a collapsed [TextSelection] at [position].
  const TextSelection.collapsed(super.position, {super.affinity})
    : super.collapsed();

  @override
  TextSelection copyWith({
    TextPosition? anchor,
    TextPosition? position,
    TextPosition? base,
    TextPosition? extent,
    TextAffinity? affinity,
  }) {
    return TextSelection(
      base: base ?? anchor ?? this.anchor,
      extent: extent ?? position ?? this.position,
      affinity: affinity ?? this.affinity,
    );
  }
}
