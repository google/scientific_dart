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
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:analyzer/dart/analysis/features.dart';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray.dart' show BoolList, ComplexList;
import 'package:ndarray/src/ndarray_bindings.dart'
    show ndarray_consume_oom_flag, ndarray_set_oom_flag;
import 'package:test/test.dart';

@pragma('vm:never-inline')
(List<Object?>, Object?) _extractTypedDataAndDropOwner<T extends AnySpec>(
  DType<T> dtype, {
  required int count,
  required String mode,
}) {
  final a = NDArray<T>.create([count], dtype);
  final Object? expectedFirst;
  final Object? expectedLast;
  switch (dtype) {
    case DType.float64:
    case DType.float32:
      expectedFirst = 42.5;
      expectedLast = -19.25;
    case DType.float16:
    case DType.bfloat16:
      expectedFirst = 3.5;
      expectedLast = -7.0;
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.int8:
      expectedFirst = -42;
      expectedLast = 99;
    case DType.uint64:
    case DType.uint32:
    case DType.uint16:
    case DType.uint8:
      expectedFirst = 42;
      expectedLast = 200;
    case DType.complex128:
    case DType.complex64:
      expectedFirst = Complex(3.25, -4.5);
      expectedLast = Complex(-1.5, 8.75);
    case DType.boolean:
      expectedFirst = true;
      expectedLast = false;
  }
  a.setCell([0], expectedFirst);
  a.setCell([count - 1], expectedLast);

  final List<Object?> view = switch (mode) {
    'typed' => a.data.cast<Object?>(),
    'base' => (a as NDArray<DTypeTag>).data.cast<Object?>(),
    'dynamic' => ((a as dynamic).data as List).cast<Object?>(),
    'slice' => a.slice([Slice(start: 0, stop: count)]).data.cast<Object?>(),
    'reshape' => a.reshape([1, count]).data.cast<Object?>(),
    'ravel' => a.ravel().data.cast<Object?>(),
    'transpose' => a.transpose().data.cast<Object?>(),
    _ => throw StateError('Unknown mode: $mode'),
  };

  return (view, (expectedFirst, expectedLast));
}

@pragma('vm:never-inline')
void _createFromPointerAndDrop(
  ffi.Pointer<ffi.Double> raw,
  ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>
  finalizerFn,
) {
  final a = NDArray<Float64>.fromPointer(
    raw.cast(),
    [64],
    DType.float64,
    nativeFinalizer: finalizerFn,
  );
  expect(a.getCell([0]), equals(123.0));
}

final List<Object?> _gcRing = List<Object?>.filled(4096, null);

Future<void> _churnGcAndNativeHeap({int rounds = 8}) async {
  for (var round = 0; round < rounds; round++) {
    for (var i = 0; i < _gcRing.length; i++) {
      _gcRing[i] = List<int>.filled(256, i + round);
    }
    _gcRing.fillRange(0, _gcRing.length, null);

    final ptrs = <ffi.Pointer<ffi.Uint32>>[];
    for (var i = 0; i < 64; i++) {
      final p = malloc<ffi.Uint32>(256);
      for (var j = 0; j < 256; j++) {
        p[j] = 0xDEADBEEF;
      }
      ptrs.add(p);
    }
    for (final p in ptrs) {
      malloc.free(p);
    }
    await Future<void>.delayed(Duration.zero);
  }
}

void main() {
  group('H5: Escaping .data views retain root NDArray owner across GC', () {
    final allDTypes = <DType<AnySpec>>[
      DType.float64,
      DType.float32,
      DType.float16,
      DType.bfloat16,
      DType.int64,
      DType.int32,
      DType.int16,
      DType.int8,
      DType.uint64,
      DType.uint32,
      DType.uint16,
      DType.uint8,
      DType.complex128,
      DType.complex64,
      DType.boolean,
    ];

    test(
      'All 15 DTypes keep backing memory alive across GC and native heap churn '
      'for typed, base, dynamic, slice, reshape, ravel, and transpose .data views',
      () async {
        const count = 64;
        const modes = <String>[
          'typed',
          'base',
          'dynamic',
          'slice',
          'reshape',
          'ravel',
          'transpose',
        ];

        final retained = <(DType<AnySpec>, String, List<Object?>, Object?)>[];
        for (final dtype in allDTypes) {
          for (final mode in modes) {
            final (view, expected) = _extractTypedDataAndDropOwner(
              dtype,
              count: count,
              mode: mode,
            );
            retained.add((dtype, mode, view, expected));
          }
        }

        await _churnGcAndNativeHeap(rounds: 10);

        for (final (dtype, mode, view, expected) in retained) {
          final (expectedFirst, expectedLast) = expected as (Object?, Object?);
          expect(
            view[0],
            equals(expectedFirst),
            reason: 'Corrupted first element for $dtype ($mode)',
          );
          expect(
            view[count - 1],
            equals(expectedLast),
            reason: 'Corrupted last element for $dtype ($mode)',
          );
        }
      },
    );

    test(
      'ComplexList.backingList and BoolList.backingList also retain root NDArray owner',
      () async {
        @pragma('vm:never-inline')
        (List<double>, List<double>, List<int>) extractBackingLists() {
          final c128 = NDArray.fromList(
            [Complex(1.5, -2.5), Complex(3.5, 4.5)],
            [2],
            DType.complex128,
          );
          final c64 = NDArray.fromList(
            [Complex(-5.0, 6.0), Complex(7.0, -8.0)],
            [2],
            DType.complex64,
          );
          final b = NDArray.fromList(
            [true, false, true, true],
            [4],
            DType.boolean,
          );
          return (
            (c128.data as ComplexList).backingList,
            (c64.data as ComplexList).backingList,
            (b.data as BoolList).backingList,
          );
        }

        final (c128Backing, c64Backing, boolBacking) = extractBackingLists();
        await _churnGcAndNativeHeap(rounds: 10);

        expect(c128Backing, equals([1.5, -2.5, 3.5, 4.5]));
        expect(c64Backing, equals([-5.0, 6.0, 7.0, -8.0]));
        expect(boolBacking, equals([1, 0, 1, 1]));
      },
    );

    test(
      'Large mmap-backed (>8 MB) .data view survives GC without SIGSEGV',
      () async {
        @pragma('vm:never-inline')
        List<double> allocateLargeDataView() {
          final a = NDArray.create([2 * 1024 * 1024], DType.float64); // 16 MB
          final d = a.data;
          d[0] = 3.141592653589793;
          d[d.length - 1] = 2.718281828459045;
          return d;
        }

        final largeView = allocateLargeDataView();
        await _churnGcAndNativeHeap(rounds: 8);

        expect(largeView[0], equals(3.141592653589793));
        expect(largeView[largeView.length - 1], equals(2.718281828459045));
      },
    );
  });

  group('M8: NDArray.fromPointer custom nativeFinalizer lifecycle', () {
    test('Custom nativeFinalizer is retained statically, invoked on GC, and '
        'detached on explicit dispose()', () async {
      // Use a real C symbol from libndarray (ndarray_set_oom_flag) as an
      // observable native finalizer callback, plus malloc.nativeFree for
      // actual deallocation. Dart's NativeFinalizer runs during GC safepoints
      // where NativeCallable trampolines are not permitted.
      final setOomPtr =
          ffi.Native.addressOf<ffi.NativeFunction<ffi.Void Function()>>(
                ndarray_set_oom_flag,
              )
              .cast<
                ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>
              >();

      ndarray_consume_oom_flag();
      final rawBuffers = <ffi.Pointer<ffi.Double>>[];
      try {
        // 1. Verify explicit dispose() invokes the custom finalizer once and
        //    detaches the NativeFinalizer so GC does not invoke it again.
        @pragma('vm:never-inline')
        void createAndDispose(ffi.Pointer<ffi.Double> raw) {
          final a = NDArray<Float64>.fromPointer(
            raw.cast(),
            [64],
            DType.float64,
            nativeFinalizer: setOomPtr,
          );
          expect(a.getCell([0]), equals(123.0));
          a.dispose();
        }

        final disposedBuf = malloc<ffi.Double>(64);
        disposedBuf[0] = 123.0;
        rawBuffers.add(disposedBuf);
        createAndDispose(disposedBuf);

        expect(
          ndarray_consume_oom_flag(),
          equals(1),
          reason: 'dispose() must synchronously invoke custom nativeFinalizer',
        );

        await _churnGcAndNativeHeap(rounds: 6);
        expect(
          ndarray_consume_oom_flag(),
          equals(0),
          reason: 'dispose() must detach custom nativeFinalizer from GC',
        );

        // 2. Verify un-disposed fromPointer arrays invoke custom
        //    nativeFinalizer when collected by GC, and malloc.nativeFree
        //    arrays are finalized without crashing.
        for (var i = 0; i < 16; i++) {
          final raw = malloc<ffi.Double>(64);
          raw[0] = 123.0;
          rawBuffers.add(raw);
          _createFromPointerAndDrop(raw, setOomPtr);

          final ownedByFree = malloc<ffi.Double>(64);
          ownedByFree[0] = 123.0;
          _createFromPointerAndDrop(ownedByFree, malloc.nativeFree);
        }

        var finalizerRan = false;
        for (var attempt = 0; attempt < 25 && !finalizerRan; attempt++) {
          await _churnGcAndNativeHeap(rounds: 4);
          if (ndarray_consume_oom_flag() != 0) {
            finalizerRan = true;
          }
        }

        expect(
          finalizerRan,
          isTrue,
          reason:
              'Custom nativeFinalizer passed to NDArray.fromPointer was never invoked by GC',
        );
      } finally {
        ndarray_consume_oom_flag();
        for (final p in rawBuffers) {
          malloc.free(p);
        }
      }
    });
  });

  group('ScratchArena: OOM recovery, exception safety, and cleanup guards', () {
    test(
      'Failed huge allocation throws OutOfMemoryError and preserves arena state',
      () {
        final initialMarker = ScratchArena.marker;
        try {
          final p1 = ScratchArena.allocate<ffi.Uint8>(64);
          p1[0] = 0x5A;
          final markerAfterP1 = ScratchArena.marker;

          // 1. Failing new-page allocation (1 << 62 bytes) throws OutOfMemoryError
          expect(
            () => ScratchArena.allocate<ffi.Uint8>(1 << 62),
            throwsA(isA<OutOfMemoryError>()),
          );
          expect(ScratchArena.marker, equals(markerAfterP1));
          expect(p1[0], equals(0x5A));

          // 2. Near-int64-max allocation (overflow guard) throws OutOfMemoryError
          expect(
            () => ScratchArena.allocate<ffi.Uint8>(0x7ffffffffffffff7),
            throwsA(isA<OutOfMemoryError>()),
          );
          expect(ScratchArena.marker, equals(markerAfterP1));

          // 3. Subsequent normal allocation and reset succeed without RangeError
          final p2 = ScratchArena.allocate<ffi.Uint8>(64);
          p2[0] = 0xA5;
          expect(p1[0], equals(0x5A));
          expect(p2[0], equals(0xA5));

          // 4. Failing allocation when shifting a cached small page also preserves state
          final markerBeforePage1 = ScratchArena.marker;
          ScratchArena.allocate<ffi.Uint8>(
            400 * 1024,
          ); // Allocates Page 1 (512 KB)
          ScratchArena.reset(
            markerBeforePage1,
          ); // Rewinds to Page 0, keeping Page 1 cached

          expect(
            () => ScratchArena.allocate<ffi.Uint8>(1 << 62),
            throwsA(isA<OutOfMemoryError>()),
          );
          expect(ScratchArena.marker, equals(markerBeforePage1));

          final p3 = ScratchArena.allocate<ffi.Uint8>(400 * 1024);
          p3[0] = 0x33;
          expect(p3[0], equals(0x33));
        } finally {
          ScratchArena.reset(initialMarker);
        }
      },
    );

    test('ScratchArena.reset inside finally never masks an in-flight exception '
        'even if native OOM flag was set', () {
      final marker = ScratchArena.marker;
      ScratchArena.allocate<ffi.Uint8>(64);
      ndarray_set_oom_flag();

      try {
        expect(() {
          try {
            throw const FormatException('original exception in try block');
          } finally {
            ScratchArena.reset(marker);
          }
        }, throwsFormatException);
      } finally {
        // Consume any remaining native OOM flag so it cannot affect other tests.
        ndarray_consume_oom_flag();
      }
    });

    test('ScratchArena.cleanup throws StateError while allocations are active '
        'and succeeds when arena is at root marker', () {
      final rootMarker = ScratchArena.marker;
      expect(rootMarker.pageIndex, equals(0));
      expect(rootMarker.offset, equals(0));

      final ptr = ScratchArena.allocate<ffi.Uint8>(32);
      ptr[0] = 77;

      // Calling cleanup() while an allocation is active on the stack must throw StateError.
      expect(() => ScratchArena.cleanup(), throwsStateError);
      // And the active allocation remains valid!
      expect(ptr[0], equals(77));

      ScratchArena.reset(rootMarker);

      // Once reset to root (0, 0), cleanup() succeeds and arena re-initializes on next allocate.
      ScratchArena.cleanup();
      final nextMarker = ScratchArena.marker;
      try {
        final ptrAfterCleanup = ScratchArena.allocate<ffi.Uint8>(32);
        ptrAfterCleanup[0] = 88;
        expect(ptrAfterCleanup[0], equals(88));
      } finally {
        ScratchArena.reset(nextMarker);
      }
    });
  });

  group('Static AST Invariant: No unowned escaping .asTypedList views in lib/', () {
    test(
      'Every .asTypedList(...) invocation in pkgs/*/lib/ is either internal '
      '(immediately copied/consumed without escaping) or owner-retaining via NDArray._data',
      () {
        final inRoot = Directory('pkgs/ndarray').existsSync();
        final pkgsRoot = inRoot ? Directory('pkgs') : Directory('..');

        final violations = <String>[];
        for (final pkgDir in pkgsRoot.listSync().whereType<Directory>()) {
          final libDir = Directory('${pkgDir.path}/lib');
          if (!libDir.existsSync()) continue;

          for (final file
              in libDir
                  .listSync(recursive: true)
                  .whereType<File>()
                  .where((f) => f.path.endsWith('.dart'))) {
            final normalizedFilePath = Uri.file(
              file.absolute.path,
            ).normalizePath().toFilePath();
            final parseResult = parseFile(
              path: normalizedFilePath,
              featureSet: FeatureSet.latestLanguageVersion(),
              throwIfDiagnostics: false,
            );
            final visitor = _AsTypedListEscapeVisitor(
              normalizedFilePath,
              parseResult.lineInfo,
            );
            parseResult.unit.accept(visitor);
            violations.addAll(visitor.violations);
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Found .asTypedList(...) calls that may escape without retaining '
              'their native memory owner:\n${violations.join('\n')}',
        );
      },
    );
  });
}

class _AsTypedListEscapeVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final List<String> violations = [];

  _AsTypedListEscapeVisitor(this.filePath, this.lineInfo);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    super.visitMethodInvocation(node);
    if (node.methodName.name != 'asTypedList') return;

    final line = lineInfo.getLocation(node.offset).lineNumber;
    final normalizedPath = filePath.replaceAll('\\', '/');

    // 1. Check if immediately chained with or passed as argument to a non-escaping consumer (.toList(), .setAll(), .setRange(), .fillRange())
    const safeConsumers = {'toList', 'setAll', 'setRange', 'fillRange'};
    final parent = node.parent;
    if (parent is MethodInvocation && identical(parent.target, node)) {
      if (safeConsumers.contains(parent.methodName.name)) {
        return;
      }
    }
    if (parent is ArgumentList) {
      final call = parent.parent;
      if (call is MethodInvocation &&
          safeConsumers.contains(call.methodName.name)) {
        return;
      }
    }

    // 2. Check if inside NDArray.create, NDArray.view, or NDArray.fromPointer in ndarray.dart
    final enclosingExec = node.thisOrAncestorMatching(
      (n) =>
          n is MethodDeclaration ||
          n is FunctionDeclaration ||
          n is ConstructorDeclaration,
    );
    if (normalizedPath.endsWith('pkgs/ndarray/lib/src/ndarray.dart') ||
        normalizedPath.endsWith('lib/src/ndarray.dart')) {
      if (enclosingExec is ConstructorDeclaration) {
        final ctorName = enclosingExec.name?.lexeme;
        if (ctorName == 'create' ||
            ctorName == 'view' ||
            ctorName == 'fromPointer') {
          return;
        }
      }
    }

    // 3. Otherwise, must be assigned to a local variable inside a function/method
    //    that never returns the typed list view (e.g. returns void, Pointer,
    //    NDArray, String, TransferableTypedData, or a fresh Uint8List copy).
    final varDecl = node.thisOrAncestorOfType<VariableDeclaration>();
    if (varDecl != null && identical(varDecl.initializer, node)) {
      String? returnTypeSource;
      AstNode? bodyNode;
      if (enclosingExec is MethodDeclaration) {
        returnTypeSource = enclosingExec.returnType?.toSource();
        bodyNode = enclosingExec.body;
      } else if (enclosingExec is FunctionDeclaration) {
        returnTypeSource = enclosingExec.returnType?.toSource();
        bodyNode = enclosingExec.functionExpression.body;
      } else if (enclosingExec is ConstructorDeclaration) {
        returnTypeSource = enclosingExec.typeName?.toSource();
        bodyNode = enclosingExec.body;
      }

      if (bodyNode != null) {
        final varName = varDecl.name.lexeme;
        final returnChecker = _LocalIdentifierReturnedVisitor(varName);
        bodyNode.accept(returnChecker);
        if (!returnChecker.isReturnedDirectly) {
          return;
        }
      }
      violations.add(
        '$filePath:$line — `.asTypedList` assigned to `${varDecl.name.lexeme}` '
        'escapes via return in `${returnTypeSource ?? 'unknown'}` function.',
      );
      return;
    }

    violations.add(
      '$filePath:$line — Unrecognized `.asTypedList` usage `${node.toSource()}`; '
      'must either be consumed locally or wrapped with owner retention.',
    );
  }
}

class _LocalIdentifierReturnedVisitor extends RecursiveAstVisitor<void> {
  final String identifierName;
  bool isReturnedDirectly = false;

  _LocalIdentifierReturnedVisitor(this.identifierName);

  @override
  void visitReturnStatement(ReturnStatement node) {
    final expr = node.expression;
    if (expr is SimpleIdentifier && expr.name == identifierName) {
      isReturnedDirectly = true;
    }
    super.visitReturnStatement(node);
  }

  @override
  void visitExpressionFunctionBody(ExpressionFunctionBody node) {
    final expr = node.expression;
    if (expr is SimpleIdentifier && expr.name == identifierName) {
      isReturnedDirectly = true;
    }
    super.visitExpressionFunctionBody(node);
  }
}
