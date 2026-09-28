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

import 'package:meta/meta.dart';

/// Contract representing the lexer's state at the end of a line.
@immutable
abstract class LineState {
  const LineState();

  /// Returns true if this state represents the default/initial state.
  bool get isInitial;
}

/// Default initial line state.
@immutable
class EmptyLineState extends LineState {
  const EmptyLineState();

  @override
  bool get isInitial => true;

  @override
  bool operator ==(Object other) => other is EmptyLineState;

  @override
  int get hashCode => 0;

  @override
  String toString() => 'EmptyLineState';
}

/// Stack-based line state representing active rules / scopes pushed during tokenization.
@immutable
class StackLineState extends LineState {
  final List<String> stack;

  const StackLineState(this.stack);

  @override
  bool get isInitial => stack.isEmpty;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! StackLineState) return false;
    if (stack.length != other.stack.length) return false;
    for (int i = 0; i < stack.length; i++) {
      if (stack[i] != other.stack[i]) return false;
    }
    return true;
  }

  @override
  int get hashCode => Object.hashAll(stack);

  @override
  String toString() => 'StackLineState($stack)';
}
