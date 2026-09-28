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

/// Platform-agnostic model representing a key combination shortcut.
@immutable
class KeyCombination {
  final String key;
  final bool ctrl;
  final bool alt;
  final bool shift;
  final bool meta;

  const KeyCombination(
    this.key, {
    this.ctrl = false,
    this.alt = false,
    this.shift = false,
    this.meta = false,
  });

  /// Parse a key combination string like "Ctrl+Shift+P" or "Alt+ArrowDown".
  factory KeyCombination.parse(String comboStr) {
    final parts = comboStr.split('+').map((s) => s.trim()).toList();
    if (parts.isEmpty) {
      return const KeyCombination('');
    }
    bool ctrl = false;
    bool alt = false;
    bool shift = false;
    bool meta = false;
    String key = '';

    for (var i = 0; i < parts.length; i++) {
      final part = parts[i];
      final lower = part.toLowerCase();
      if (i == parts.length - 1 &&
          lower != 'ctrl' &&
          lower != 'alt' &&
          lower != 'shift' &&
          lower != 'cmd' &&
          lower != 'meta') {
        key = part;
      } else {
        switch (lower) {
          case 'ctrl':
          case 'control':
            ctrl = true;
            break;
          case 'alt':
          case 'option':
            alt = true;
            break;
          case 'shift':
            shift = true;
            break;
          case 'cmd':
          case 'meta':
          case 'command':
            meta = true;
            break;
          default:
            key = part;
            break;
        }
      }
    }

    return KeyCombination(key, ctrl: ctrl, alt: alt, shift: shift, meta: meta);
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is KeyCombination &&
          runtimeType == other.runtimeType &&
          key.toLowerCase() == other.key.toLowerCase() &&
          ctrl == other.ctrl &&
          alt == other.alt &&
          shift == other.shift &&
          meta == other.meta;

  @override
  int get hashCode => Object.hash(key.toLowerCase(), ctrl, alt, shift, meta);

  @override
  String toString() {
    final buffer = <String>[];
    if (ctrl) buffer.add('Ctrl');
    if (alt) buffer.add('Alt');
    if (shift) buffer.add('Shift');
    if (meta) buffer.add('Meta');
    buffer.add(key);
    return buffer.join('+');
  }
}

/// Keybinding entry binding a key shortcut to an action command.
class KeyCommandBinding {
  final KeyCombination combination;
  final String commandId;
  final void Function() action;

  KeyCommandBinding({
    required this.combination,
    required this.commandId,
    required this.action,
  });
}

/// Registry for managing and dispatching platform-agnostic keyboard shortcuts.
class KeybindingRegistry {
  final Map<KeyCombination, List<KeyCommandBinding>> _bindings = {};

  /// Register a command action for a given key combination.
  void register(
    KeyCombination combination,
    String commandId,
    void Function() action,
  ) {
    final binding = KeyCommandBinding(
      combination: combination,
      commandId: commandId,
      action: action,
    );
    _bindings.putIfAbsent(combination, () => []).add(binding);
  }

  /// Unregister all actions for a specific key combination or command ID.
  void unregister(KeyCombination combination, {String? commandId}) {
    if (commandId == null) {
      _bindings.remove(combination);
    } else {
      final list = _bindings[combination];
      if (list != null) {
        list.removeWhere((b) => b.commandId == commandId);
        if (list.isEmpty) {
          _bindings.remove(combination);
        }
      }
    }
  }

  /// Dispatch an incoming key combination event.
  /// Returns `true` if a binding was found and executed.
  bool dispatch(KeyCombination combination) {
    final list = _bindings[combination];
    if (list != null && list.isNotEmpty) {
      for (final binding in List<KeyCommandBinding>.from(list)) {
        binding.action();
      }
      return true;
    }
    return false;
  }

  /// Retrieve registered command IDs for a key combination.
  List<String> getCommands(KeyCombination combination) {
    final list = _bindings[combination];
    if (list == null) return const [];
    return list.map((b) => b.commandId).toList();
  }

  /// Clear all registered keybindings.
  void clear() {
    _bindings.clear();
  }
}
