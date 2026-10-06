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

import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' hide diagonal, trace;
import 'package:test/test.dart';

void main() {
  group('Linear Algebra Tensor Contractions & Products (F16)', () {
    test('multiDot chains matrix multiplications with out: parameter', () {
      final first = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      final second = GpuArray.fromList(
        <double>[2.0, 0.0, 0.0, 2.0],
        [2, 2],
        DType.float64,
      );
      final third = GpuArray.fromList(
        <double>[1.0, 1.0, 0.0, 1.0],
        [2, 2],
        DType.float64,
      );
      final outProduct = GpuArray.zeros([2, 2], DType.float64);
      try {
        final product = multiDot(<GpuArray<Float64>>[
          first,
          second,
          third,
        ], out: outProduct);
        expect(identical(product, outProduct), isTrue);
        expect(product.toList(), equals(<double>[2.0, 6.0, 6.0, 14.0]));
      } finally {
        outProduct.dispose();
        third.dispose();
        second.dispose();
        first.dispose();
      }
    });

    test('vdot conjugates complex first vector (F10.1)', () {
      final vectorA = GpuArray.fromList(
        <Complex>[Complex(1.0, 2.0), Complex(3.0, -4.0)],
        [2],
        DType.complex128,
      );
      final vectorB = GpuArray.fromList(
        <Complex>[Complex(5.0, 6.0), Complex(7.0, 8.0)],
        [2],
        DType.complex128,
      );
      try {
        final result = vdot(vectorA, vectorB);
        try {
          final scalar = result.scalar as Complex;
          // conj(1 + 2i)*(5 + 6i) + conj(3 - 4i)*(7 + 8i) = 6 + 48i
          expect(scalar.real, closeTo(6.0, 1e-10));
          expect(scalar.imag, closeTo(48.0, 1e-10));
        } finally {
          result.dispose();
        }
      } finally {
        vectorB.dispose();
        vectorA.dispose();
      }
    });

    test('einsum evaluates trace, matrix multiplication, and out:', () {
      final matrix = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      final outProduct = GpuArray.zeros([2, 2], DType.float64);
      try {
        final traceResult = einsum('ii->', <GpuArray<Float64>>[matrix]);
        try {
          expect(traceResult.scalar as double, closeTo(5.0, 1e-10));
        } finally {
          traceResult.dispose();
        }

        final squared = einsum('ij,jk->ik', <GpuArray<Float64>>[
          matrix,
          matrix,
        ], out: outProduct);
        expect(identical(squared, outProduct), isTrue);
        expect(squared.toList(), equals(<double>[7.0, 10.0, 15.0, 22.0]));
      } finally {
        outProduct.dispose();
        matrix.dispose();
      }
    });

    test('tensordot supports integer and explicit axis lists', () {
      final left = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      final right = GpuArray.fromList(
        <double>[5.0, 6.0, 7.0, 8.0],
        [2, 2],
        DType.float64,
      );
      try {
        final scalarContracted = tensordot(left, right, axes: 2);
        final axisContracted = tensordot(
          left,
          right,
          axes: <List<int>>[
            <int>[1],
            <int>[0],
          ],
        );
        try {
          expect(scalarContracted.scalar as double, closeTo(70.0, 1e-10));
          expect(
            axisContracted.toList(),
            equals(<double>[19.0, 22.0, 43.0, 50.0]),
          );
        } finally {
          scalarContracted.dispose();
          axisContracted.dispose();
        }
      } finally {
        right.dispose();
        left.dispose();
      }
    });

    test('kron, inner, outer, and cross products with out: parameter', () {
      final vectorA = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0],
        [3],
        DType.float64,
      );
      final vectorB = GpuArray.fromList(
        <double>[4.0, 5.0, 6.0],
        [3],
        DType.float64,
      );
      final outCross = GpuArray.zeros([3], DType.float64);
      try {
        final innerResult = inner(vectorA, vectorB);
        final outerResult = outer(vectorA, vectorB);
        final kronResult = kron(vectorA, vectorB);
        final crossResult = cross(vectorA, vectorB, out: outCross);
        try {
          expect(innerResult.scalar as double, closeTo(32.0, 1e-10));
          expect(outerResult.shape, equals(<int>[3, 3]));
          expect(kronResult.shape, equals(<int>[9]));
          expect(identical(crossResult, outCross), isTrue);
          expect(crossResult.toList(), equals(<double>[-3.0, 6.0, -3.0]));
        } finally {
          innerResult.dispose();
          outerResult.dispose();
          kronResult.dispose();
        }
      } finally {
        outCross.dispose();
        vectorB.dispose();
        vectorA.dispose();
      }
    });

    test('Float32 tensor contractions preserve Float32 dtype (F12)', () {
      final a = GpuArray.fromList(<double>[1.0, 2.0, 3.0], [3], DType.float32);
      final b = GpuArray.fromList(<double>[4.0, 5.0, 6.0], [3], DType.float32);
      final m = GpuArray.fromList(
        <double>[1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float32,
      );
      try {
        final vdotRes = vdot(a, b);
        final innerRes = inner(a, b);
        final outerRes = outer(a, b);
        final kronRes = kron(a, b);
        final crossRes = cross(a, b);
        final tdRes = tensordot(m, m, axes: 1);
        final einRes = einsum('ij,jk->ik', <GpuArray<Float32>>[m, m]);
        final mdRes = multiDot(<GpuArray<Float32>>[m, m]);
        try {
          expect(vdotRes.dtype, equals(DType.float32));
          expect(vdotRes.scalar as double, closeTo(32.0, 1e-5));
          expect(innerRes.dtype, equals(DType.float32));
          expect(outerRes.dtype, equals(DType.float32));
          expect(kronRes.dtype, equals(DType.float32));
          expect(crossRes.dtype, equals(DType.float32));
          expect(crossRes.toList(), equals(<double>[-3.0, 6.0, -3.0]));
          expect(tdRes.dtype, equals(DType.float32));
          expect(einRes.dtype, equals(DType.float32));
          expect(mdRes.dtype, equals(DType.float32));
        } finally {
          vdotRes.dispose();
          innerRes.dispose();
          outerRes.dispose();
          kronRes.dispose();
          crossRes.dispose();
          tdRes.dispose();
          einRes.dispose();
          mdRes.dispose();
        }
      } finally {
        m.dispose();
        b.dispose();
        a.dispose();
      }
    });
  });
}
