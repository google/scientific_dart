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

import "dart:ffi" as ffi;
import "dart:math" as math;
import "package:ndarray/ndarray.dart";
import "package:ndarray/src/ndarray_bindings.dart" as bindings;
import "package:test/test.dart";

void main() {
  group("0D Scalar Elementwise Ufuncs Tests", () {
    test("0D scalar arithmetic (+, -, *, /) with out and on strided views", () {
      NDArray.scope(() {
        final a = NDArray<Float64>.scalar(6.0, dtype: DType.float64);
        final b = NDArray<Float64>.scalar(2.0, dtype: DType.float64);

        // Standard operator syntax
        final sumOp = a + b;
        final diffOp = a - b;
        final prodOp = a * b;
        final quotOp = a / b;
        expect((sumOp.scalar as num).toDouble(), closeTo(8.0, 1e-12));
        expect((diffOp.scalar as num).toDouble(), closeTo(4.0, 1e-12));
        expect((prodOp.scalar as num).toDouble(), closeTo(12.0, 1e-12));
        expect((quotOp.scalar as num).toDouble(), closeTo(3.0, 1e-12));

        // With explicit out: argument
        final outAdd = NDArray<Float64>.zeros(<int>[], DType.float64);
        final outSub = NDArray<Float64>.zeros(<int>[], DType.float64);
        final outMul = NDArray<Float64>.zeros(<int>[], DType.float64);
        final outDiv = NDArray<Float64>.zeros(<int>[], DType.float64);

        add<Float64>(a, b, out: outAdd);
        subtract<Float64>(a, b, out: outSub);
        multiply<Float64>(a, b, out: outMul);
        divide<Float64, Float64, Float64>(a, b, out: outDiv);

        expect((outAdd.scalar as num).toDouble(), closeTo(8.0, 1e-12));
        expect((outSub.scalar as num).toDouble(), closeTo(4.0, 1e-12));
        expect((outMul.scalar as num).toDouble(), closeTo(12.0, 1e-12));
        expect((outDiv.scalar as num).toDouble(), closeTo(3.0, 1e-12));

        // On strided/scalar 0D views (non-contiguous 0D view triggering strided path)
        final backingA = NDArray<Float64>.fromList(
          <double>[0.0, 6.0, 0.0],
          <int>[3],
          DType.float64,
        );
        final backingB = NDArray<Float64>.fromList(
          <double>[0.0, 0.0, 2.0],
          <int>[3],
          DType.float64,
        );
        final backingOut = NDArray<Float64>.zeros(<int>[3], DType.float64);

        final viewA = NDArray<Float64>.view(
          backingA,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final viewB = NDArray<Float64>.view(
          backingB,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 2,
        );
        final viewOut = NDArray<Float64>.view(
          backingOut,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );

        expect(viewA.isContiguous, isTrue);
        expect(viewOut.isContiguous, isTrue);

        add<Float64>(viewA, viewB, out: viewOut);
        expect((viewOut.scalar as num).toDouble(), closeTo(8.0, 1e-12));

        subtract<Float64>(viewA, viewB, out: viewOut);
        expect((viewOut.scalar as num).toDouble(), closeTo(4.0, 1e-12));

        multiply<Float64>(viewA, viewB, out: viewOut);
        expect((viewOut.scalar as num).toDouble(), closeTo(12.0, 1e-12));

        divide<Float64, Float64, Float64>(viewA, viewB, out: viewOut);
        expect((viewOut.scalar as num).toDouble(), closeTo(3.0, 1e-12));
      });
    });

    test("0D scalar sin and cos with out and on strided views", () {
      NDArray.scope(() {
        final backing = NDArray<Float64>.fromList(
          <double>[0.0, math.pi / 6.0, 0.0],
          <int>[3],
          DType.float64,
        );
        final view0D = NDArray<Float64>.view(
          backing,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final outBacking = NDArray<Float64>.zeros(<int>[2], DType.float64);
        final outView = NDArray<Float64>.view(
          outBacking,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );

        sin(view0D, out: outView);
        expect((outView.scalar as num).toDouble(), closeTo(0.5, 1e-12));

        cos(view0D, out: outView);
        expect(
          (outView.scalar as num).toDouble(),
          closeTo(math.sqrt(3.0) / 2.0, 1e-12),
        );
      });
    });

    test(
      "0D scalar abs (Float64 and Complex128) and conj with out and strided views",
      () {
        NDArray.scope(() {
          final backingF = NDArray<Float64>.fromList(
            <double>[0.0, -7.5],
            <int>[2],
            DType.float64,
          );
          final viewF = NDArray<Float64>.view(
            backingF,
            shape: <int>[],
            strides: <int>[],
            offsetElements: 1,
          );
          final outF = NDArray<Float64>.view(
            NDArray<Float64>.zeros(<int>[2], DType.float64),
            shape: <int>[],
            strides: <int>[],
            offsetElements: 1,
          );

          abs(viewF, out: outF);
          expect((outF.scalar as num).toDouble(), closeTo(7.5, 1e-12));

          final backingC = NDArray<Complex128>.fromList(
            <Complex>[Complex(0.0, 0.0), Complex(3.0, -4.0)],
            <int>[2],
            DType.complex128,
          );
          final viewC = NDArray<Complex128>.view(
            backingC,
            shape: <int>[],
            strides: <int>[],
            offsetElements: 1,
          );
          final outAbsC = NDArray<Float64>.view(
            NDArray<Float64>.zeros(<int>[2], DType.float64),
            shape: <int>[],
            strides: <int>[],
            offsetElements: 1,
          );

          abs(viewC, out: outAbsC);
          expect((outAbsC.scalar as num).toDouble(), closeTo(5.0, 1e-12));

          final outConjC = NDArray<Complex128>.view(
            NDArray<Complex128>.zeros(<int>[2], DType.complex128),
            shape: <int>[],
            strides: <int>[],
            offsetElements: 1,
          );
          conj<Complex128>(viewC, out: outConjC);
          final conjVal = outConjC.scalar;
          expect(conjVal.real, closeTo(3.0, 1e-12));
          expect(conjVal.imag, closeTo(4.0, 1e-12));
        });
      },
    );

    test("0D scalar atan2 and pow/power with out and on strided views", () {
      NDArray.scope(() {
        final backingY = NDArray<Float64>.fromList(
          <double>[0.0, 1.0],
          <int>[2],
          DType.float64,
        );
        final backingX = NDArray<Float64>.fromList(
          <double>[0.0, 1.0],
          <int>[2],
          DType.float64,
        );
        final viewY = NDArray<Float64>.view(
          backingY,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final viewX = NDArray<Float64>.view(
          backingX,
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final outAtan2 = NDArray<Float64>.view(
          NDArray<Float64>.zeros(<int>[2], DType.float64),
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );

        atan2<Float64, Float64>(viewY, viewX, out: outAtan2);
        expect(
          (outAtan2.scalar as num).toDouble(),
          closeTo(math.pi / 4.0, 1e-12),
        );

        // Power on Float64 0D strided views
        final baseF = NDArray<Float64>.view(
          NDArray<Float64>.fromList(
            <double>[0.0, 2.0],
            <int>[2],
            DType.float64,
          ),
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final expF = NDArray<Float64>.view(
          NDArray<Float64>.fromList(
            <double>[0.0, 3.0],
            <int>[2],
            DType.float64,
          ),
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final outPowF = NDArray<Float64>.view(
          NDArray<Float64>.zeros(<int>[2], DType.float64),
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        power<Float64>(baseF, expF, out: outPowF);
        expect((outPowF.scalar as num).toDouble(), closeTo(8.0, 1e-12));

        // Power on Complex128 0D strided views
        final baseC = NDArray<Complex128>.view(
          NDArray<Complex128>.fromList(
            <Complex>[Complex(0.0, 0.0), Complex(0.0, 1.0)],
            <int>[2],
            DType.complex128,
          ),
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final expC = NDArray<Complex128>.view(
          NDArray<Complex128>.fromList(
            <Complex>[Complex(0.0, 0.0), Complex(2.0, 0.0)],
            <int>[2],
            DType.complex128,
          ),
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final outPowC = NDArray<Complex128>.view(
          NDArray<Complex128>.zeros(<int>[2], DType.complex128),
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        power<Complex128>(baseC, expC, out: outPowC);
        expect(outPowC.scalar.real, closeTo(-1.0, 1e-12));
        expect(outPowC.scalar.imag, closeTo(0.0, 1e-12));
      });
    });

    test("0D scalar castNDArray and low-level strided C ufuncs on rank 0", () {
      NDArray.scope(() {
        final sU8 = NDArray<Uint8>.scalar(200, dtype: DType.uint8);
        final castedF64 = castNDArray<Float64>(sU8, DType.float64);
        expect(castedF64.rank, 0);
        expect((castedF64.scalar as num).toDouble(), closeTo(200.0, 1e-12));

        final viewU8 = NDArray<Uint8>.view(
          NDArray<Uint8>.fromList(<int>[10, 250], <int>[2], DType.uint8),
          shape: <int>[],
          strides: <int>[],
          offsetElements: 1,
        );
        final castedViewF64 = castNDArray<Float64>(viewU8, DType.float64);
        expect(castedViewF64.rank, 0);
        expect((castedViewF64.scalar as num).toDouble(), closeTo(250.0, 1e-12));

        // Direct verification of low-level rank == 0 strided C functions
        final marker = ScratchArena.marker;
        try {
          final aD = NDArray<Float64>.scalar(9.0, dtype: DType.float64);
          final bD = NDArray<Float64>.scalar(3.0, dtype: DType.float64);
          final resD = NDArray<Float64>.scalar(0.0, dtype: DType.float64);

          bindings.s_add_double(
            aD.pointer.cast(),
            ffi.nullptr,
            bD.pointer.cast(),
            ffi.nullptr,
            resD.pointer.cast(),
            ffi.nullptr,
            ffi.nullptr,
            0,
          );
          expect((resD.scalar as num).toDouble(), closeTo(12.0, 1e-12));

          bindings.s_sub_double(
            aD.pointer.cast(),
            ffi.nullptr,
            bD.pointer.cast(),
            ffi.nullptr,
            resD.pointer.cast(),
            ffi.nullptr,
            ffi.nullptr,
            0,
          );
          expect((resD.scalar as num).toDouble(), closeTo(6.0, 1e-12));

          bindings.s_mul_double(
            aD.pointer.cast(),
            ffi.nullptr,
            bD.pointer.cast(),
            ffi.nullptr,
            resD.pointer.cast(),
            ffi.nullptr,
            ffi.nullptr,
            0,
          );
          expect((resD.scalar as num).toDouble(), closeTo(27.0, 1e-12));

          bindings.s_div_double(
            aD.pointer.cast(),
            ffi.nullptr,
            bD.pointer.cast(),
            ffi.nullptr,
            resD.pointer.cast(),
            ffi.nullptr,
            ffi.nullptr,
            0,
          );
          expect((resD.scalar as num).toDouble(), closeTo(3.0, 1e-12));

          final u8Src = NDArray<Uint8>.scalar(42, dtype: DType.uint8);
          bindings.s_cast_uint8_to_double(
            u8Src.pointer.cast(),
            ffi.nullptr,
            resD.pointer.cast(),
            ffi.nullptr,
            ffi.nullptr,
            0,
          );
          expect((resD.scalar as num).toDouble(), closeTo(42.0, 1e-12));

          final i16Dst = NDArray<Int16>.scalar(0, dtype: DType.int16);
          bindings.s_cast_double_to_int16(
            resD.pointer.cast(),
            ffi.nullptr,
            i16Dst.pointer.cast(),
            ffi.nullptr,
            ffi.nullptr,
            0,
          );
          expect(i16Dst.scalar, 42);
        } finally {
          ScratchArena.reset(marker);
        }
      });
    });
  });

  group("Empty Axis Integer Mean Tests", () {
    test(
      "mean on integer array with 0-length dimension returns NaN matching var_",
      () {
        NDArray.scope(() {
          final emptyInt = NDArray<Int64>.zeros(<int>[2, 0, 3], DType.int64);
          final m = mean<Float64, Int64>(emptyInt, axis: 1);
          final v = var_<Int64>(emptyInt, axis: 1);

          expect(m.shape, <int>[2, 3]);
          expect(v.shape, <int>[2, 3]);

          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 3; j++) {
              final meanVal = (m.getCell(<int>[i, j]) as num).toDouble();
              final varVal = (v.getCell(<int>[i, j]) as num).toDouble();
              expect(meanVal.isNaN, isTrue);
              expect(varVal.isNaN, isTrue);
            }
          }

          // Verify with explicit zero-initialized out buffer
          final outBuf = NDArray<Float64>.zeros(<int>[2, 3], DType.float64);
          mean<Float64, Int64>(emptyInt, axis: 1, out: outBuf);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 3; j++) {
              expect(
                (outBuf.getCell(<int>[i, j]) as num).toDouble().isNaN,
                isTrue,
              );
            }
          }

          // Direct FFI call to s_mean_int64_to_double on empty axis
          final outFfi = NDArray<Float64>.zeros(<int>[2, 3], DType.float64);
          final marker = ScratchArena.marker;
          try {
            final cShape = ScratchArena.copyInts(<int>[2, 0, 3]);
            final cStridesSrc = ScratchArena.copyInts(emptyInt.strides);
            final cStridesDst = ScratchArena.copyInts(outFfi.strides);
            bindings.s_mean_int64_to_double(
              emptyInt.pointer.cast(),
              cStridesSrc,
              outFfi.pointer.cast(),
              cStridesDst,
              cShape,
              3,
              1,
            );
            for (var i = 0; i < 2; i++) {
              for (var j = 0; j < 3; j++) {
                expect(
                  (outFfi.getCell(<int>[i, j]) as num).toDouble().isNaN,
                  isTrue,
                );
              }
            }
          } finally {
            ScratchArena.reset(marker);
          }
        });
      },
    );
  });
}
