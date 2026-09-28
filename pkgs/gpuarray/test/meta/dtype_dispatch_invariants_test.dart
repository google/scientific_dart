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

import 'dart:io';

import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  final pkgRoot = Directory.current.path.endsWith('pkgs/gpuarray')
      ? Directory.current
      : Directory('pkgs/gpuarray');

  group('DType dispatch invariants', () {
    test('DType enum has all 15 expected scientific data types', () {
      expect(DType.values, hasLength(15));
      expect(
        DType.values.map((d) => d.name).toSet(),
        equals({
          'float64',
          'float32',
          'float16',
          'bfloat16',
          'int64',
          'int32',
          'int16',
          'int8',
          'uint64',
          'uint32',
          'uint16',
          'uint8',
          'boolean',
          'complex64',
          'complex128',
        }),
      );
    });

    test(
      'GpuArray._create switch in lib/src/gpu_array.dart explicitly dispatches all 15 DType values',
      () {
        final gpuArrayFile = File('${pkgRoot.path}/lib/src/gpu_array.dart');
        final result = parseString(content: gpuArrayFile.readAsStringSync());
        final visitor = _CreateSwitchVisitor();
        result.unit.accept(visitor);
        expect(visitor.foundCreateMethod, isTrue);
        for (final dtype in DType.values) {
          expect(
            visitor.coveredCases,
            contains(dtype.name),
            reason:
                'GpuArray._create must explicitly handle DType.${dtype.name}',
          );
        }
      },
    );

    test(
      'All 15 DTypes instantiate reified GpuArray<T> across factories and views',
      () {
        final device = GpuDevice.cpu();
        device.detachFromScope();
        try {
          final expectedTypes = <DType, bool Function(GpuArray<DTypeTag>)>{
            DType.float64: (a) => a is GpuArray<Float64>,
            DType.float32: (a) => a is GpuArray<Float32>,
            DType.float16: (a) => a is GpuArray<Float16>,
            DType.bfloat16: (a) => a is GpuArray<BFloat16>,
            DType.int64: (a) => a is GpuArray<Int64>,
            DType.int32: (a) => a is GpuArray<Int32>,
            DType.int16: (a) => a is GpuArray<Int16>,
            DType.int8: (a) => a is GpuArray<Int8>,
            DType.uint64: (a) => a is GpuArray<Uint64>,
            DType.uint32: (a) => a is GpuArray<Uint32>,
            DType.uint16: (a) => a is GpuArray<Uint16>,
            DType.uint8: (a) => a is GpuArray<Uint8>,
            DType.boolean: (a) => a is GpuArray<Boolean>,
            DType.complex64: (a) => a is GpuArray<Complex64>,
            DType.complex128: (a) => a is GpuArray<Complex128>,
          };

          for (final dtype in DType.values) {
            ResourceScope.scope(() {
              final check = expectedTypes[dtype]!;
              final z = GpuArray.zeros([2, 3], dtype, device: device);
              expect(check(z), isTrue, reason: 'zeros for $dtype');
              expect(
                check(z.transpose()),
                isTrue,
                reason: 'transpose for $dtype',
              );
              expect(
                check(z.reshape([6])),
                isTrue,
                reason: 'reshape for $dtype',
              );
              expect(
                check(z.slice([0, const Slice.all()])),
                isTrue,
                reason: 'slice for $dtype',
              );
              expect(check(z.copy()), isTrue, reason: 'copy for $dtype');
            });
          }
        } finally {
          device.dispose();
        }
      },
    );
  });
}

class _CreateSwitchVisitor extends RecursiveAstVisitor<void> {
  bool foundCreateMethod = false;
  final Set<String> coveredCases = {};

  @override
  void visitMethodDeclaration(MethodDeclaration node) {
    if (node.name.lexeme == '_create') {
      foundCreateMethod = true;
      node.accept(_SwitchCaseCollector(coveredCases));
    }
    super.visitMethodDeclaration(node);
  }
}

class _SwitchCaseCollector extends RecursiveAstVisitor<void> {
  final Set<String> coveredCases;

  _SwitchCaseCollector(this.coveredCases);

  @override
  void visitSwitchExpressionCase(SwitchExpressionCase node) {
    final patSource = node.guardedPattern.pattern.toSource();
    if (patSource.startsWith('DType.')) {
      coveredCases.add(patSource.substring('DType.'.length));
    }
    super.visitSwitchExpressionCase(node);
  }
}
