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

import '../../events/keybinding_registry.dart';

export '../../events/keybinding_registry.dart';

/// Terminal mouse event decoded from SGR escape sequences (`\x1b[<flags;x;yM`).
class TerminalMouseEvent {
  final int button;
  final int x;
  final int y;
  final bool isRelease;
  final bool isMove;

  TerminalMouseEvent({
    required this.button,
    required this.x,
    required this.y,
    this.isRelease = false,
    this.isMove = false,
  });
}

/// Raw mode stdin byte stream & escape sequence decoder.
class TerminalInputParser {
  final List<void Function(KeyCombination combo)> _keyListeners = [];
  final List<void Function(TerminalMouseEvent mouse)> _mouseListeners = [];

  void addKeyListener(void Function(KeyCombination combo) listener) {
    _keyListeners.add(listener);
  }

  void addMouseListener(void Function(TerminalMouseEvent mouse) listener) {
    _mouseListeners.add(listener);
  }

  /// Process raw input data string or stdin chunk.
  void parseInput(String data) {
    if (data.isEmpty) return;

    var index = 0;
    while (index < data.length) {
      if (data[index] == '\x1b') {
        // Escape sequence start
        if (index + 1 < data.length && data[index + 1] == '[') {
          // CSI sequence
          final rest = data.substring(index + 2);
          // Check SGR mouse sequence: e.g. `<0;20;10M` or `<0;20;10m`
          if (rest.startsWith('<')) {
            final match = RegExp(r'^<(\d+);(\d+);(\d+)([Mm])').firstMatch(rest);
            if (match != null) {
              final flags = int.parse(match.group(1)!);
              final x = int.parse(match.group(2)!);
              final y = int.parse(match.group(3)!);
              final action = match.group(4)!;
              final isRelease = action == 'm';
              final mouseEvent = TerminalMouseEvent(
                button: flags & 3,
                x: x,
                y: y,
                isRelease: isRelease,
                isMove: (flags & 32) != 0,
              );
              _emitMouse(mouseEvent);
              index += 2 + match.group(0)!.length;
              continue;
            }
          }

          // Check Delete sequence: `\x1b[3~`
          if (rest.startsWith('3~')) {
            _emitKey(const KeyCombination('Delete'));
            index += 4;
            continue;
          }

          // Check Ctrl+Arrow sequences: `\x1b[1;5A`, `\x1b[1;5B`, `\x1b[1;5C`, `\x1b[1;5D`
          if (rest.startsWith('1;5A')) {
            _emitKey(const KeyCombination('ArrowUp', ctrl: true));
            index += 6;
            continue;
          } else if (rest.startsWith('1;5B')) {
            _emitKey(const KeyCombination('ArrowDown', ctrl: true));
            index += 6;
            continue;
          } else if (rest.startsWith('1;5C')) {
            _emitKey(const KeyCombination('ArrowRight', ctrl: true));
            index += 6;
            continue;
          } else if (rest.startsWith('1;5D')) {
            _emitKey(const KeyCombination('ArrowLeft', ctrl: true));
            index += 6;
            continue;
          }

          // Check standard cursor/key sequences
          if (rest.startsWith('A')) {
            _emitKey(const KeyCombination('ArrowUp'));
            index += 3;
            continue;
          } else if (rest.startsWith('B')) {
            _emitKey(const KeyCombination('ArrowDown'));
            index += 3;
            continue;
          } else if (rest.startsWith('C')) {
            _emitKey(const KeyCombination('ArrowRight'));
            index += 3;
            continue;
          } else if (rest.startsWith('D')) {
            _emitKey(const KeyCombination('ArrowLeft'));
            index += 3;
            continue;
          } else if (rest.startsWith('H')) {
            _emitKey(const KeyCombination('Home'));
            index += 3;
            continue;
          } else if (rest.startsWith('F')) {
            _emitKey(const KeyCombination('End'));
            index += 3;
            continue;
          } else if (rest.startsWith('5~')) {
            _emitKey(const KeyCombination('PageUp'));
            index += 4;
            continue;
          } else if (rest.startsWith('6~')) {
            _emitKey(const KeyCombination('PageDown'));
            index += 4;
            continue;
          }
        } else if (index + 1 < data.length && data[index + 1] != '\x1b') {
          // Alt + character sequence (e.g. `\x1b` + 'a')
          final altChar = data[index + 1];
          _emitKey(KeyCombination(altChar, alt: true));
          index += 2;
          continue;
        }
        // Standalone escape key
        _emitKey(const KeyCombination('Escape'));
        index++;
      } else {
        final charCode = data.codeUnitAt(index);
        if (charCode == 13 || charCode == 10) {
          _emitKey(const KeyCombination('Enter'));
        } else if (charCode == 9) {
          _emitKey(const KeyCombination('Tab'));
        } else if (charCode == 127 || charCode == 8) {
          _emitKey(const KeyCombination('Backspace'));
        } else if (charCode >= 1 && charCode <= 26) {
          // Ctrl+A to Ctrl+Z
          final ctrlChar = String.fromCharCode(charCode + 64);
          _emitKey(KeyCombination(ctrlChar, ctrl: true));
        } else {
          _emitKey(KeyCombination(data[index]));
        }
        index++;
      }
    }
  }

  void _emitKey(KeyCombination combo) {
    for (final listener in List<void Function(KeyCombination)>.from(
      _keyListeners,
    )) {
      listener(combo);
    }
  }

  void _emitMouse(TerminalMouseEvent mouse) {
    for (final listener in List<void Function(TerminalMouseEvent)>.from(
      _mouseListeners,
    )) {
      listener(mouse);
    }
  }
}
