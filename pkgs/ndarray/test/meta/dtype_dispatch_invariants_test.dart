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

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:analyzer/dart/analysis/features.dart';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/source/line_info.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_bindings.dart';
import 'package:ndarray/src/operations/native_pointer.dart';
import 'package:test/test.dart';

Directory _findPackageRoot() {
  var dir = Directory.current;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/hook').existsSync()) {
      return dir;
    }
    final sub = Directory('${dir.path}/pkgs/ndarray');
    if (sub.existsSync()) {
      return sub;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError('Could not locate pkgs/ndarray root directory.');
    }
    dir = parent;
  }
}

String _native(String path) => Uri.file(path).toFilePath();

String _posix(String path) => path.replaceAll(r'\', '/');

List<File> _dartFilesIn(Directory dir) {
  if (!dir.existsSync()) return const [];
  return dir
      .listSync(recursive: true)
      .whereType<File>()
      .where((f) => f.path.endsWith('.dart'))
      .toList()
    ..sort((a, b) => a.path.compareTo(b.path));
}

String _stripCppComments(String source) {
  final withoutBlock = source.replaceAll(
    RegExp(r'/\*[\s\S]*?\*/', multiLine: true),
    '',
  );
  return withoutBlock
      .split('\n')
      .map((line) {
        final idx = line.indexOf('//');
        return idx >= 0 ? line.substring(0, idx) : line;
      })
      .join('\n');
}

bool _isDTypeSwitch(SwitchStatement node) {
  final exprSource = node.expression.toSource();
  if (exprSource.contains('dtype') || exprSource.contains('DType')) {
    return true;
  }
  for (final member in node.members) {
    if (member is SwitchPatternCase) {
      if (member.guardedPattern.pattern.toSource().contains('DType.')) {
        return true;
      }
    } else if (member is SwitchCase) {
      if (member.expression.toSource().contains('DType.')) {
        return true;
      }
    }
  }
  return false;
}

bool _isRecordDTypeSwitch(SwitchStatement node) {
  if (node.expression is RecordLiteral) return true;
  for (final member in node.members) {
    if (member is SwitchPatternCase &&
        member.guardedPattern.pattern is RecordPattern) {
      return true;
    }
  }
  return false;
}

bool _statementsJustBreak(NodeList<Statement> statements) {
  if (statements.isEmpty) return true;
  if (statements.length == 1 && statements.first is BreakStatement) {
    return true;
  }
  return false;
}

bool _statementsAlwaysTerminate(NodeList<Statement> statements) {
  if (statements.isEmpty) return false;
  final last = statements.last;
  if (last is ReturnStatement) return true;
  if (last is ExpressionStatement && last.expression is ThrowExpression) {
    return true;
  }
  if (last is IfStatement && last.elseStatement != null) {
    final thenStmts = last.thenStatement is Block
        ? (last.thenStatement as Block).statements
        : null;
    final elseStmts = last.elseStatement is Block
        ? (last.elseStatement as Block).statements
        : null;
    if (thenStmts != null &&
        elseStmts != null &&
        _statementsAlwaysTerminate(thenStmts) &&
        _statementsAlwaysTerminate(elseStmts)) {
      return true;
    }
  }
  return false;
}

/// Finds the first statement executed after [stmt] completes normally (via
/// fallthrough or `break` out of a switch), walking up enclosing `Block`,
/// `IfStatement`, and `TryStatement` nodes.
Statement? _nextExecutedStatementAfterSwitch(SwitchStatement node) {
  AstNode? current = node;
  while (current != null && current is! FunctionBody) {
    final parent = current.parent;
    if (parent is Block) {
      final stmts = parent.statements;
      final idx = stmts.indexOf(current as Statement);
      if (idx >= 0 && idx + 1 < stmts.length) {
        return stmts[idx + 1];
      }
      current = parent;
    } else if (parent is IfStatement) {
      current = parent;
    } else if (parent is TryStatement) {
      // If current is inside the try body, after finally it continues after TryStatement.
      current = parent;
    } else {
      break;
    }
  }
  return null;
}

class _DTypeSwitchSafetyVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations = [];

  _DTypeSwitchSafetyVisitor(this.filePath, this.lineInfo);

  @override
  void visitSwitchStatement(SwitchStatement node) {
    if (_isDTypeSwitch(node)) {
      final posixPath = _posix(filePath);
      final isLinalg = posixPath.endsWith('/linalg.dart');
      final isOperations = posixPath.contains('/src/operations/');
      final isRecord = _isRecordDTypeSwitch(node);
      final nextStmt = _nextExecutedStatementAfterSwitch(node);
      final fallsThroughToReturn =
          nextStmt is ReturnStatement &&
          !(nextStmt.expression is MethodInvocation &&
              (nextStmt.expression as MethodInvocation).methodName.name ==
                  "scope");

      for (var i = 0; i < node.members.length; i++) {
        final member = node.members[i];
        final isSharedCaseLabel =
            member is! SwitchDefault &&
            member.statements.isEmpty &&
            i + 1 < node.members.length;
        if (isSharedCaseLabel) continue;
        if (member is SwitchDefault) {
          final loc = lineInfo.getLocation(member.offset);
          if (_statementsJustBreak(member.statements)) {
            if (fallsThroughToReturn) {
              violations.add(
                '$posixPath:${loc.lineNumber}: `default: break;` in DType '
                'switch (${node.expression.toSource()}) falls through directly '
                'to `${nextStmt.toSource()}` (unwritten buffer hazard).',
              );
            } else if (isOperations && !isLinalg) {
              violations.add(
                '$posixPath:${loc.lineNumber}: `default: break;` is forbidden '
                'in DType switches under lib/src/operations/ '
                '(${node.expression.toSource()}). Use an exhaustive switch '
                'over all DType values or `default: throw UnsupportedError(...)`.',
              );
            }
          } else if (isOperations && !isLinalg && !isRecord) {
            if (!_statementsAlwaysTerminate(member.statements)) {
              violations.add(
                '$posixPath:${loc.lineNumber}: single-DType switch '
                '(${node.expression.toSource()}) has a non-throwing `default:`. '
                'List all 15 DType cases explicitly for compiler exhaustiveness.',
              );
            }
          }
        } else if (!isLinalg &&
            _statementsJustBreak(member.statements) &&
            fallsThroughToReturn) {
          final loc = lineInfo.getLocation(member.offset);
          violations.add(
            '$posixPath:${loc.lineNumber}: `break;` case in DType switch '
            '(${node.expression.toSource()}) falls through directly to '
            '`${nextStmt.toSource()}` without writing output buffer.',
          );
        }
      }
    }
    super.visitSwitchStatement(node);
  }
}

void main() {
  final pkgRoot = _findPackageRoot();
  final libDir = Directory('${pkgRoot.path}/lib');
  final hookDir = Directory('${pkgRoot.path}/hook');
  final libFiles = _dartFilesIn(libDir);
  final featureSet = FeatureSet.latestLanguageVersion();

  group('DType Dispatch Safety Invariants', () {
    test(
      'No switch over DType (or record of DTypes) under lib/ has a default: break that falls through to returning an unwritten buffer',
      () {
        final violations = <String>[];
        for (final file in libFiles) {
          final result = parseFile(
            path: _native(file.path),
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final visitor = _DTypeSwitchSafetyVisitor(file.path, result.lineInfo);
          result.unit.accept(visitor);
          violations.addAll(visitor.violations);
        }
        expect(
          violations,
          isEmpty,
          reason:
              'Found unsafe DType switch dispatch patterns:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'Every C/C++ switch on a dtype code in hook/*.cpp has a failing default:',
      () {
        final cppFiles =
            hookDir
                .listSync()
                .whereType<File>()
                .where((f) => f.path.endsWith('.cpp'))
                .toList()
              ..sort((a, b) => a.path.compareTo(b.path));
        expect(cppFiles, isNotEmpty);

        final violations = <String>[];
        final switchRegex = RegExp(r'\bswitch\s*\(([^)]+)\)\s*\{');

        for (final file in cppFiles) {
          final raw = file.readAsStringSync();
          final stripped = _stripCppComments(raw);
          for (final match in switchRegex.allMatches(stripped)) {
            final expr = match.group(1)!.trim();
            final bodyStart = match.end;
            var depth = 1;
            var i = bodyStart;
            while (i < stripped.length && depth > 0) {
              final ch = stripped.codeUnitAt(i);
              if (ch == 0x7B /* { */ ) {
                depth++;
              } else if (ch == 0x7D /* } */ ) {
                depth--;
              }
              i++;
            }
            final body = stripped.substring(bodyStart, i - 1);
            final isDTypeSwitch =
                expr.contains('dtype') || body.contains('DTYPE_');
            if (!isDTypeSwitch) continue;

            final lineNum =
                '\n'.allMatches(stripped.substring(0, match.start)).length + 1;
            final defaultMatch = RegExp(
              r'\bdefault\s*:\s*([^}]*)',
            ).firstMatch(body);
            if (defaultMatch == null) {
              violations.add(
                '${_posix(file.path)}:$lineNum: C++ switch ($expr) on dtype '
                'is missing a `default:` branch.',
              );
              continue;
            }
            final afterDefault = defaultMatch.group(1)!;
            final hasFailingAction = RegExp(
              r'\b(abort\s*\(\s*\)|return\s+-\d+)',
            ).hasMatch(afterDefault);
            final hasSilentBreak = RegExp(
              r'^\s*break\s*;',
            ).hasMatch(afterDefault);
            if (!hasFailingAction || hasSilentBreak) {
              violations.add(
                '${_posix(file.path)}:$lineNum: C++ switch ($expr) `default:` '
                'must fail loudly (`abort()` or negative error code), found: '
                '`${afterDefault.split('\n').first.trim()}`.',
              );
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Every C/C++ dtype switch in hook/*.cpp must have a failing '
              'default:\n${violations.join('\n')}',
        );
      },
    );

    test(
      'NDArrayNativePointer.typedPointer and complexComponentPointer reject mismatched FFI element types at the Dart-to-native boundary',
      () {
        NDArray.scope(() {
          for (final dt in DType.values) {
            final arr = dt == DType.boolean
                ? NDArray.fromList([true], [1], DType.boolean)
                : (dt.isComplex
                      ? NDArray.fromList([Complex(1, 2)], [1], dt)
                      : NDArray.zeros([1], dt));

            void checkType<N extends ffi.NativeType>(
              ffi.Pointer<N> Function() getter,
              bool expectedValid,
            ) {
              if (expectedValid) {
                expect(getter().address, isNonZero);
              } else {
                expect(getter, throwsStateError);
              }
            }

            checkType<ffi.Double>(
              () => arr.typedPointer<ffi.Double>(),
              dt == DType.float64,
            );
            checkType<ffi.Float>(
              () => arr.typedPointer<ffi.Float>(),
              dt == DType.float32,
            );
            checkType<ffi.Int64>(
              () => arr.typedPointer<ffi.Int64>(),
              dt == DType.int64,
            );
            checkType<ffi.Uint64>(
              () => arr.typedPointer<ffi.Uint64>(),
              dt == DType.uint64,
            );
            checkType<ffi.Int32>(
              () => arr.typedPointer<ffi.Int32>(),
              dt == DType.int32,
            );
            checkType<ffi.Uint32>(
              () => arr.typedPointer<ffi.Uint32>(),
              dt == DType.uint32,
            );
            checkType<ffi.Int16>(
              () => arr.typedPointer<ffi.Int16>(),
              dt == DType.int16,
            );
            checkType<ffi.Uint16>(
              () => arr.typedPointer<ffi.Uint16>(),
              dt == DType.uint16 || dt == DType.float16 || dt == DType.bfloat16,
            );
            checkType<ffi.Int8>(
              () => arr.typedPointer<ffi.Int8>(),
              dt == DType.int8,
            );
            checkType<ffi.Uint8>(
              () => arr.typedPointer<ffi.Uint8>(),
              dt == DType.uint8 || dt == DType.boolean,
            );
            checkType<cpx_t>(
              () => arr.typedPointer<cpx_t>(),
              dt == DType.complex128,
            );
            checkType<cpx_f_t>(
              () => arr.typedPointer<cpx_f_t>(),
              dt == DType.complex64,
            );
            checkType<ffi.Double>(
              () => arr.complexComponentPointer<ffi.Double>(),
              dt == DType.complex128,
            );
            checkType<ffi.Float>(
              () => arr.complexComponentPointer<ffi.Float>(),
              dt == DType.complex64,
            );
          }
        });
      },
    );
  });
}
