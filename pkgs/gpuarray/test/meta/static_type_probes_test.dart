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

/// Static-type probe tests for `package:gpuarray` (R5 & behaviour preservation).
///
/// Spot-checks exact inferred static types on `GpuArray<Float32>`,
/// `GpuArray<Int32>`, `GpuArray<Complex64>`, `GpuArray<AnySpec>`, and integer
/// LUBs via both `package:analyzer` (`getDisplayString()`) and the CFE runtime
/// `staticTypeOf` witness, plus negative overlay diagnostics on bare
/// `GpuArray<DTypeTag>`.
library;

import 'dart:io';

import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/diagnostic/diagnostic.dart';
import 'package:analyzer/file_system/overlay_file_system.dart';
import 'package:analyzer/file_system/physical_file_system.dart';
import 'package:analyzer/source/line_info.dart';
import 'package:gpuarray/fft.dart' as gfft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as glinalg;
import 'package:gpuarray/nn.dart' as gnn;
import 'package:gpuarray/random.dart' as grandom;
import 'package:test/test.dart';

/// Captures the exact static return type `T` inferred by the Dart compiler for
/// a closure `f` without executing `f`.
Type staticTypeOf<T>(T Function() f) => T;

/// Returns the runtime representation of the static type `T`.
Type typeOf<T>() => T;

late final GpuArray<Float64> _g64;
late final GpuArray<Float32> _g32;
late final GpuArray<Float16> _gf16;
late final GpuArray<BFloat16> _gbf16;
late final GpuArray<Int64> _gi64;
late final GpuArray<Int32> _gi32;
late final GpuArray<Uint8> _gu8;
late final GpuArray<Boolean> _gb;
late final GpuArray<Complex64> _gc64;
late final GpuArray<AnySpec> _ganySpec;

const String _positiveOverlaySource = '''
// ignore_for_file: non_constant_identifier_names, unused_element
import 'package:gpuarray/fft.dart' as gfft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as glinalg;
import 'package:gpuarray/nn.dart' as gnn;
import 'package:gpuarray/random.dart' as grandom;

late final GpuArray<Float64> g64;
late final GpuArray<Float32> g32;
late final GpuArray<Float16> gf16;
late final GpuArray<BFloat16> gbf16;
late final GpuArray<Int64> gi64;
late final GpuArray<Int32> gi32;
late final GpuArray<Uint8> gu8;
late final GpuArray<Boolean> gb;
late final GpuArray<Complex64> gc64;
late final GpuArray<AnySpec> ganySpec;
late final GpuArray<DTypeTag> gdyn;

// --- GpuArray<Float32>
// expect: GpuArray<Float32>
final t_divide_fn_f32 = divide(g32, g32);
// expect: GpuArray<Float32>
final t_divide_m_f32 = g32.divide(g32);
// expect: GpuArray<Float32>
final t_divide_op_f32 = g32 / g32;
// expect: GpuArray<Float32>
final t_sin_fn_f32 = sin(g32);
// expect: GpuArray<Float32>
final t_sin_m_f32 = g32.sin();
// expect: GpuArray<Float32>
final t_sum_fn_f32 = sum(g32);
// expect: GpuArray<Float32>
final t_sum_m_f32 = g32.sum();
// expect: GpuArray<Float32>
final t_abs_fn_f32 = abs(g32);
// expect: GpuArray<Float32>
final t_abs_m_f32 = g32.abs();
// expect: GpuArray<Float32>
final t_real_m_f32 = g32.real();
// expect: GpuArray<Float32>
final t_real_fn_f32 = real(g32);
// expect: GpuArray<Float32>
final t_angle_m_f32 = g32.angle();
// expect: GpuArray<Float32>
final t_angle_fn_f32 = angle(g32);
// expect: GpuArray<Float32>
final t_mean_m_f32 = g32.mean();
// expect: GpuArray<Complex64>
final t_fft_fn_f32 = gfft.fft(g32);
// expect: GpuArray<Complex64>
final t_rfft_fn_f32 = gfft.rfft(g32);
// expect: GpuArray<Float32>
final t_irfft_fn_f32 = gfft.irfft(g32);

// --- GpuArray<Float16> & GpuArray<BFloat16>
// expect: GpuArray<Float16>
final t_divide_fn_f16 = divide(gf16, gf16);
// expect: GpuArray<Float16>
final t_divide_m_f16 = gf16.divide(gf16);
// expect: GpuArray<Float16>
final t_divide_op_f16 = gf16 / gf16;
// expect: GpuArray<BFloat16>
final t_divide_op_bf16 = gbf16 / gbf16;
// expect: GpuArray<Float64>
final t_angle_m_f16 = gf16.angle();
// expect: GpuArray<Float64>
final t_angle_fn_f16 = angle(gf16);
// expect: GpuArray<Float64>
final t_angle_m_bf16 = gbf16.angle();

// --- GpuArray<Int32> & GpuArray<Boolean> (NumPy true-divide -> Float64)
// expect: GpuArray<Float64>
final t_divide_fn_i32 = divide(gi32, gi32);
// expect: GpuArray<Float64>
final t_divide_m_i32 = gi32.divide(gi32);
// expect: GpuArray<Float64>
final t_divide_op_i32 = gi32 / gi32;
// expect: GpuArray<Float64>
final t_divide_op_bool = gb / gb;
// expect: GpuArray<Int32>
final t_sin_fn_i32 = sin(gi32);
// expect: GpuArray<Int32>
final t_sin_m_i32 = gi32.sin();
// expect: GpuArray<Int32>
final t_sum_fn_i32 = sum(gi32);
// expect: GpuArray<Int32>
final t_sum_m_i32 = gi32.sum();
// expect: GpuArray<Int32>
final t_abs_fn_i32 = abs(gi32);
// expect: GpuArray<Int32>
final t_abs_m_i32 = gi32.abs();
// expect: GpuArray<Int32>
final t_real_m_i32 = gi32.real();
// expect: GpuArray<Float64>
final t_angle_m_i32 = gi32.angle();
// expect: GpuArray<DTypeTag>
final t_mean_m_i32 = gi32.mean();
// expect: GpuArray<Complex128>
final t_fft_fn_i32 = gfft.fft(gi32);
// expect: GpuArray<Complex128>
final t_rfft_fn_i32 = gfft.rfft(gi32);
// expect: GpuArray<Float64>
final t_irfft_fn_i32 = gfft.irfft(gi32);

// --- GpuArray<Complex64>
// expect: GpuArray<Complex64>
final t_divide_fn_c64 = divide(gc64, gc64);
// expect: GpuArray<Complex64>
final t_divide_m_c64 = gc64.divide(gc64);
// expect: GpuArray<Complex64>
final t_divide_op_c64 = gc64 / gc64;
// expect: GpuArray<Complex64>
final t_sin_fn_c64 = sin(gc64);
// expect: GpuArray<Complex64>
final t_sum_fn_c64 = sum(gc64);
// expect: GpuArray<Complex64>
final t_abs_fn_c64 = abs(gc64);
// expect: GpuArray<Float32>
final t_real_m_c64 = gc64.real();
// expect: GpuArray<Float32>
final t_real_fn_c64 = real(gc64);
// expect: GpuArray<Float32>
final t_imag_fn_c64 = imag(gc64);
// expect: GpuArray<Float32>
final t_angle_m_c64 = gc64.angle();
// expect: GpuArray<Float32>
final t_angle_fn_c64 = angle(gc64);
// expect: GpuArray<Complex64>
final t_mean_m_c64 = gc64.mean();
// expect: GpuArray<Complex64>
final t_fft_fn_c64 = gfft.fft(gc64);
// expect: GpuArray<Complex64>
final t_rfft_fn_c64 = gfft.rfft(gc64);
// expect: GpuArray<Float32>
final t_irfft_fn_c64 = gfft.irfft(gc64);

// --- GpuArray<AnySpec>
// expect: GpuArray<DTypeTag>
final t_divide_fn_anyspec = divide(ganySpec, ganySpec);
// expect: GpuArray<DTypeTag>
final t_divide_m_anyspec = ganySpec.divide(ganySpec);
// expect: GpuArray<DTypeTag>
final t_divide_op_anyspec = ganySpec / ganySpec;
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_sin_fn_anyspec = sin(ganySpec);
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_sum_fn_anyspec = sum(ganySpec);
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_abs_fn_anyspec = abs(ganySpec);
// expect: GpuArray<DTypeTag>
final t_real_m_anyspec = ganySpec.real();
// expect: GpuArray<DTypeTag>
final t_angle_m_anyspec = ganySpec.angle();
// expect: GpuArray<DTypeTag>
final t_mean_m_anyspec = ganySpec.mean();
// expect: GpuArray<DTypeTag>
final t_fft_fn_anyspec = gfft.fft(ganySpec);
// expect: GpuArray<DTypeTag>
final t_rfft_fn_anyspec = gfft.rfft(ganySpec);
// expect: GpuArray<DTypeTag>
final t_irfft_fn_anyspec = gfft.irfft(ganySpec);

// --- Linalg & LUBs
// expect: ({GpuArray<Float32> eigenvalues, GpuArray<Float32> eigenvectors})
final t_eigh_f32 = glinalg.eigh(g32);
// expect: ({GpuArray<Float32> s, GpuArray<Float32> u, GpuArray<Float32> vt})
final t_svd_f32 = glinalg.svd(g32);
// expect: ({int rank, GpuArray<Float32> residuals, GpuArray<Float32> singularValues, GpuArray<Float32> solution})
final t_lstsq_f32 = glinalg.lstsq(g32, g32);
// expect: ({GpuArray<Float32> logabsdet, GpuArray<Float32> sign})
final t_slogdet_f32 = glinalg.slogdet(g32);
// expect: GpuArray<Complex128>
final t_fft_lub_int = gfft.fft([gi64, gi32].first);
// expect: GpuArray<Float64>
final t_inv_lub_int = glinalg.inv([gi64, gi32].first);
// expect: GpuArray<Float64>
final t_div_lub_int = [gi64, gi32].first / [gi64, gi32].first;

// --- Same-dtype binary functions (SelfOf) infer the concrete tag
// expect: GpuArray<Float32>
final t_add_fn_f32 = add(g32, g32);
// expect: GpuArray<Float64>
final t_add_fn_f64 = add(g64, g64);
// expect: GpuArray<Float32>
final t_add_fn_out_f32 = add(g32, g32, out: g32);
// expect: GpuArray<Int32>
final t_maximum_fn_i32 = maximum(gi32, gi32);
// expect: GpuArray<Int32>
final t_atan2_fn_i32 = atan2(gi32, gi32);
// expect: GpuArray<Float32>
final t_hypot_fn_f32 = hypot(g32, g32);
// expect: GpuArray<Boolean>
final t_equal_fn_f32 = equal(g32, g32);
// expect: GpuArray<Boolean>
final t_greater_fn_i32 = greater(gi32, gi32);
// expect: GpuArray<Int32>
final t_bitwiseAnd_fn_i32 = bitwiseAnd(gi32, gi32);
// expect: GpuArray<Boolean>
final t_bitwiseOr_fn_bool = bitwiseOr(gb, gb);
// expect: GpuArray<Int32>
final t_invert_fn_i32 = invert(gi32);
// expect: GpuArray<Uint8>
final t_leftShift_fn_u8 = leftShift(gu8, gu8);
// expect: GpuArray<Int64>
final t_gcd_fn_i64 = gcd(gi64, gi64);
// expect: GpuArray<Float32>
final t_ldexp_fn_f32_i32 = ldexp(g32, gi32);
// expect: GpuArray<DTypeTag>
final t_ldexp_fn_dyn = ldexp(gdyn, gi32);
// expect: GpuArray<Float64>
final t_matmul_fn_f64 = glinalg.matmul(g64, g64);
// expect: GpuArray<Float32>
final t_tensordot_fn_f32 = glinalg.tensordot(g32, g32, axes: 1);
// expect: GpuArray<Float32>
final t_multiDot_fn_f32 = glinalg.multiDot([g32, g32]);
// expect: GpuArray<Float32>
final t_einsum_fn_f32 = glinalg.einsum('ij,jk->ik', [g32, g32]);
// expect: GpuArray<Float32>
final t_mseLoss_fn_f32 = gnn.mseLoss(g32, g32);
// expect: GpuArray<Float32>
final t_sdpa_fn_f32 = gnn.scaledDotProductAttention(g32, g32, g32);

// --- Marker-bounded operators (GpuArrayBitwise / GpuArrayShift) and LUBs
// expect: GpuArray<Int32>
final t_opand_i32 = gi32 & gi32;
// expect: GpuArray<Int32>
final t_opand_scalar_i32 = gi32 & 1;
// expect: GpuArray<Boolean>
final t_opor_bool = gb | gb;
// expect: GpuArray<Int32>
final t_opxor_i32 = gi32 ^ gi32;
// expect: GpuArray<Int32>
final t_opnot_i32 = ~gi32;
// expect: GpuArray<Boolean>
final t_opnot_bool = ~gb;
// expect: GpuArray<Int32>
final t_opshl_i32 = gi32 << gi32;
// expect: GpuArray<Uint8>
final t_opshr_u8 = gu8 >> 1;
// expect: GpuArray<IntegerDType>
final t_opnot_lub_int = ~[gi32, gu8].first;
// expect: GpuArray<IntegerDType>
final t_opand_lub_int = [gi32, gu8].first & [gi32, gu8].first;
// expect: GpuArray<IntegerDType>
final t_opshl_lub_int = [gi32, gu8].first << 1;
// expect: GpuArray<BitwiseDType>
final t_opand_lub_bitwise = [gi32, gb].first & [gi32, gb].first;
// expect: GpuArray<BitwiseDType>
final t_opnot_lub_bitwise = ~[gi32, gb].first;

// --- Escape hatches: AnySpec / AnyBitwiseSpec / AnyIntegerSpec rows
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_add_fn_anyspec = add(ganySpec, ganySpec);
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_asAnySpec_dyn = [g32, gi32].first.asAnySpec;
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_matmul_fn_anyspec = glinalg.matmul(ganySpec, ganySpec);
// expect: GpuArray<Boolean>
final t_equal_fn_anyspec = equal(ganySpec, ganySpec);
// expect: GpuArray<DTypeSpec<BitwiseDType, dynamic, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_asBitwise_i32 = gi32.asBitwiseDType;
// expect: GpuArray<DTypeSpec<BitwiseDType, dynamic, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_bitwiseAnd_fn_bitwiseSpec = bitwiseAnd(
  gi32.asBitwiseDType,
  gi32.asBitwiseDType,
);
// expect: GpuArray<DTypeSpec<BitwiseDType, dynamic, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_opand_bitwiseSpec = gi32.asBitwiseDType & gi32.asBitwiseDType;
// expect: GpuArray<DTypeSpec<BitwiseDType, dynamic, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_opnot_bitwiseSpec = ~gi32.asBitwiseDType;
// expect: GpuArray<DTypeSpec<IntegerDType, int, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_asInteger_i32 = gi32.asIntegerDType;
// expect: GpuArray<DTypeSpec<IntegerDType, int, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_gcd_fn_integerSpec = gcd(gi32.asIntegerDType, gi32.asIntegerDType);
// expect: GpuArray<DTypeSpec<IntegerDType, int, Float64, Complex128, Float64, DTypeTag, Float64, Float64>>
final t_opshl_integerSpec = gi32.asIntegerDType << gi32.asIntegerDType;

// --- Run-time-typed results default to AnySpec and infer from `out:`
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_concatenate_default = concatenate([g32, g32]);
// expect: GpuArray<Float32>
final t_concatenate_out_f32 = concatenate([g32, g32], out: g32);
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_add_concatenate_anyspec = add(concatenate([g32, g32]), g32);
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_where_default = where(gb, g32, g32);
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_permutation_default = grandom.permutation(4);
// expect: GpuArray<Int64>
final t_permutation_out_i64 = grandom.permutation(4, out: gi64);
''';

const String _negativeOverlaySource = '''
// ignore_for_file: non_constant_identifier_names, unused_element
import 'package:gpuarray/fft.dart' as gfft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as glinalg;

late final GpuArray<DTypeTag> gdyn;
late final GpuArray<Float64> gf64;
late final GpuArray<Float32> gf32;
late final GpuArray<Int64> gi64;
late final GpuArray<Int32> gi32;
late final GpuArray<Uint8> gu8;
late final GpuArray<Boolean> gb;
late final GpuArray<AnySpec> ganySpec;

final t_neg_divide_dyn = divide(gdyn, gdyn); // error: could_not_infer, argument_type_not_assignable
final t_neg_real_dyn = real(gdyn); // error: argument_type_not_assignable
final t_neg_imag_dyn = imag(gdyn); // error: argument_type_not_assignable
final t_neg_angle_dyn = angle(gdyn); // error: argument_type_not_assignable
final t_neg_fft_dyn = gfft.fft(gdyn); // error: argument_type_not_assignable
final t_neg_rfft_dyn = gfft.rfft(gdyn); // error: argument_type_not_assignable
final t_neg_irfft_dyn = gfft.irfft(gdyn); // error: argument_type_not_assignable

// Mixed-dtype binary & multi-array operations are rejected symmetrically (SelfOf):
final t_neg_add_f64_f32 = add(gf64, gf32); // error: could_not_infer
final t_neg_add_f32_f64 = add(gf32, gf64); // error: could_not_infer
final t_neg_add_i64_i32 = add(gi64, gi32); // error: could_not_infer
final t_neg_add_i32_i64 = add(gi32, gi64); // error: could_not_infer
final t_neg_add_out_f32 = add(gf64, gf64, out: gf32); // error: could_not_infer
final t_neg_add_dyn = add(gdyn, gdyn); // error: argument_type_not_assignable
final t_neg_add_scalar = add(gf64, 2.0); // error: argument_type_not_assignable
final t_neg_divide_i64_i32 = divide(gi64, gi32); // error: could_not_infer
final t_neg_atan2_i64_i32 = atan2(gi64, gi32); // error: could_not_infer
final t_neg_equal_f64_f32 = equal(gf64, gf32); // error: could_not_infer
final t_neg_matmul_f64_f32 = glinalg.matmul(gf64, gf32); // error: could_not_infer
final t_neg_multiDot_f64_f32 = glinalg.multiDot([gf64, gf32]); // error: argument_type_not_assignable
final t_neg_bitwiseAnd_i64_i32 = bitwiseAnd(gi64, gi32); // error: could_not_infer
final t_neg_gcd_i64_i32 = gcd(gi64, gi32); // error: could_not_infer

// Bitwise / shift functions reject non-integer dtypes and the bare markers:
final t_neg_bitwiseAnd_f64 = bitwiseAnd(gf64, gf64); // error: argument_type_not_assignable
final t_neg_invert_f64 = invert(gf64); // error: argument_type_not_assignable
final t_neg_gcd_bool = gcd(gb, gb); // error: argument_type_not_assignable
final t_neg_leftShift_bool = leftShift(gb, gb); // error: argument_type_not_assignable
final t_neg_bitwiseAnd_anySpec = bitwiseAnd(ganySpec, ganySpec); // error: argument_type_not_assignable
final t_neg_bitwiseAnd_lub = bitwiseAnd([gi32, gu8].first, [gi32, gu8].first); // error: argument_type_not_assignable

// The operators only exist on BitwiseDType / IntegerDType tags and the *Spec rows:
final t_neg_opand_f64 = gf64 & gf64; // error: undefined_operator
final t_neg_opnot_f32 = ~gf32; // error: undefined_operator
final t_neg_opshl_f64 = gf64 << 1; // error: undefined_operator
final t_neg_opshl_bool = gb << gb; // error: undefined_operator
final t_neg_opand_dyn = gdyn & gdyn; // error: undefined_operator
final t_neg_opand_anySpec = ganySpec & ganySpec; // error: undefined_operator
final t_neg_opshl_lub_bitwise = [gi32, gb].first << 1; // error: undefined_operator

// Generic helpers must declare the same kind of bound as the callee:
GpuArray<T> badGenericAdd<T extends DTypeTag>(GpuArray<T> a) => add(a, a); // error: argument_type_not_assignable
GpuArray<T> badGenericAnd<T extends BitwiseDType>(GpuArray<T> a) => bitwiseAnd(a, a); // error: argument_type_not_assignable

// A bare GpuArray<DTypeTag> is no longer accepted as a run-time-typed `out:`:
final t_neg_concatenate_out_dyn = concatenate([gf32, gf32], out: gdyn); // error: argument_type_not_assignable
''';

String _native(String path) => Uri.file(path).toFilePath();

void main() {
  final pkgRoot = Directory.current.path.endsWith('pkgs/gpuarray')
      ? Directory.current
      : Directory('pkgs/gpuarray');
  final resolvedPkgRoot = _native(pkgRoot.resolveSymbolicLinksSync());

  group('gpuarray static type probes', () {
    test(
      'analyzer verifies exact static types for GpuArray operations and LUBs',
      () async {
        final virtualPath = _native(
          '$resolvedPkgRoot/test/meta/_positive_gpu_probes_overlay.dart',
        );
        final lines = _positiveOverlaySource.split('\n');
        final expectedByLine = <int, String>{};
        final expectRegex = RegExp(r'//\s*expect:\s*(.+)$');
        for (var i = 0; i < lines.length; i++) {
          final match = expectRegex.firstMatch(lines[i]);
          if (match != null) {
            expectedByLine[i + 1] = match.group(1)!.trim();
          }
        }

        final overlay = OverlayResourceProvider(
          PhysicalResourceProvider.INSTANCE,
        );
        overlay.setOverlay(
          virtualPath,
          content: _positiveOverlaySource,
          modificationStamp: 0,
        );
        final collection = AnalysisContextCollection(
          includedPaths: [virtualPath],
          resourceProvider: overlay,
        );
        final session = collection.contextFor(virtualPath).currentSession;
        final result = await session.getResolvedUnit(virtualPath);
        expect(result, isA<ResolvedUnitResult>());
        final unitResult = result as ResolvedUnitResult;

        final errors = unitResult.diagnostics
            .where((d) => d.severity == Severity.error)
            .map(
              (d) =>
                  'L${unitResult.lineInfo.getLocation(d.offset).lineNumber} '
                  '${d.diagnosticCode.lowerCaseName}: ${d.message}',
            )
            .toList();
        expect(errors, isEmpty, reason: errors.join('\n'));

        final visitor = _GpuProbeVisitor(unitResult.lineInfo, expectedByLine);
        unitResult.unit.accept(visitor);
        expect(visitor.missingExpectations, isEmpty);
        expect(
          visitor.mismatches,
          isEmpty,
          reason: visitor.mismatches.join('\n'),
        );
        expect(visitor.checkedCount, equals(expectedByLine.length));
      },
    );

    test(
      'CFE runtime staticTypeOf matches exact static types for GpuArray spot checks',
      () {
        // Float32
        expect(
          staticTypeOf(() => divide(_g32, _g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => _g32.divide(_g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => _g32 / _g32),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => sin(_g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => sum(_g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => abs(_g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => _g32.real()),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => _g32.angle()),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => _g32.mean()),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => gfft.fft(_g32)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => gfft.rfft(_g32)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => gfft.irfft(_g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );

        // Float16 & BFloat16
        expect(
          staticTypeOf(() => _gf16 / _gf16),
          equals(typeOf<GpuArray<Float16>>()),
        );
        expect(
          staticTypeOf(() => _gbf16 / _gbf16),
          equals(typeOf<GpuArray<BFloat16>>()),
        );
        expect(
          staticTypeOf(() => _gf16.angle()),
          equals(typeOf<GpuArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => _gbf16.angle()),
          equals(typeOf<GpuArray<Float64>>()),
        );

        // Int32 & Boolean (NumPy true-divide -> GpuArray<Float64>)
        expect(
          staticTypeOf(() => divide(_gi32, _gi32)),
          equals(typeOf<GpuArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => _gi32.divide(_gi32)),
          equals(typeOf<GpuArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => _gi32 / _gi32),
          equals(typeOf<GpuArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => _gb / _gb),
          equals(typeOf<GpuArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => sin(_gi32)),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => sum(_gi32)),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => abs(_gi32)),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => _gi32.real()),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => _gi32.angle()),
          equals(typeOf<GpuArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => _gi32.mean()),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => gfft.fft(_gi32)),
          equals(typeOf<GpuArray<Complex128>>()),
        );
        expect(
          staticTypeOf(() => gfft.rfft(_gi32)),
          equals(typeOf<GpuArray<Complex128>>()),
        );
        expect(
          staticTypeOf(() => gfft.irfft(_gi32)),
          equals(typeOf<GpuArray<Float64>>()),
        );

        // Complex64
        expect(
          staticTypeOf(() => divide(_gc64, _gc64)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => _gc64.divide(_gc64)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => _gc64 / _gc64),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => sin(_gc64)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => sum(_gc64)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => abs(_gc64)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => _gc64.real()),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => _gc64.angle()),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => _gc64.mean()),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => gfft.fft(_gc64)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => gfft.rfft(_gc64)),
          equals(typeOf<GpuArray<Complex64>>()),
        );
        expect(
          staticTypeOf(() => gfft.irfft(_gc64)),
          equals(typeOf<GpuArray<Float32>>()),
        );

        // AnySpec
        expect(
          staticTypeOf(() => divide(_ganySpec, _ganySpec)),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => _ganySpec.divide(_ganySpec)),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => _ganySpec / _ganySpec),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => sin(_ganySpec)),
          equals(typeOf<GpuArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(() => sum(_ganySpec)),
          equals(typeOf<GpuArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(() => abs(_ganySpec)),
          equals(typeOf<GpuArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(() => _ganySpec.real()),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => _ganySpec.angle()),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => _ganySpec.mean()),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => gfft.fft(_ganySpec)),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => gfft.rfft(_ganySpec)),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );
        expect(
          staticTypeOf(() => gfft.irfft(_ganySpec)),
          equals(typeOf<GpuArray<DTypeTag>>()),
        );

        // Linalg & LUBs
        expect(
          staticTypeOf(() => glinalg.eigh(_g32)),
          equals(
            typeOf<
              ({GpuArray<Float32> eigenvalues, GpuArray<Float32> eigenvectors})
            >(),
          ),
        );
        expect(
          staticTypeOf(() => glinalg.svd(_g32)),
          equals(
            typeOf<
              ({GpuArray<Float32> u, GpuArray<Float32> s, GpuArray<Float32> vt})
            >(),
          ),
        );
        expect(
          staticTypeOf(() => glinalg.lstsq(_g32, _g32)),
          equals(
            typeOf<
              ({
                GpuArray<Float32> solution,
                GpuArray<Float32> residuals,
                int rank,
                GpuArray<Float32> singularValues,
              })
            >(),
          ),
        );
        expect(
          staticTypeOf(() => glinalg.slogdet(_g32)),
          equals(
            typeOf<({GpuArray<Float32> sign, GpuArray<Float32> logabsdet})>(),
          ),
        );
        expect(
          staticTypeOf(() => gfft.fft([_gi64, _gi32].first)),
          equals(typeOf<GpuArray<Complex128>>()),
        );
        expect(
          staticTypeOf(() => glinalg.inv([_gi64, _gi32].first)),
          equals(typeOf<GpuArray<Float64>>()),
        );

        // Same-dtype binary functions (SelfOf)
        expect(
          staticTypeOf(() => add(_g32, _g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => add(_g32, _g32, out: _g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => atan2(_gi32, _gi32)),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => equal(_g32, _g32)),
          equals(typeOf<GpuArray<Boolean>>()),
        );
        expect(
          staticTypeOf(() => bitwiseAnd(_gi32, _gi32)),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => leftShift(_gu8, _gu8)),
          equals(typeOf<GpuArray<Uint8>>()),
        );
        expect(
          staticTypeOf(() => ldexp(_g32, _gi32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => glinalg.matmul(_g64, _g64)),
          equals(typeOf<GpuArray<Float64>>()),
        );
        expect(
          staticTypeOf(() => glinalg.multiDot([_g32, _g32])),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => gnn.mseLoss(_g32, _g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => add(_ganySpec, _ganySpec)),
          equals(typeOf<GpuArray<AnySpec>>()),
        );

        // Marker-bounded operators and their least upper bounds
        expect(
          staticTypeOf(() => _gi32 & _gi32),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => _gb | _gb),
          equals(typeOf<GpuArray<Boolean>>()),
        );
        expect(
          staticTypeOf(() => _gi32 << 1),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => ~[_gi32, _gu8].first),
          equals(typeOf<GpuArray<IntegerDType>>()),
        );
        expect(
          staticTypeOf(() => [_gi32, _gu8].first >> [_gi32, _gu8].first),
          equals(typeOf<GpuArray<IntegerDType>>()),
        );
        expect(
          staticTypeOf(() => [_gi32, _gb].first ^ [_gi32, _gb].first),
          equals(typeOf<GpuArray<BitwiseDType>>()),
        );

        // Escape hatches and the *Spec operator siblings
        expect(
          staticTypeOf(() => _gi32.asAnySpec),
          equals(typeOf<GpuArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(() => _gi32.asBitwiseDType & _gi32.asBitwiseDType),
          equals(typeOf<GpuArray<AnyBitwiseSpec>>()),
        );
        expect(
          staticTypeOf(() => ~_gi32.asBitwiseDType),
          equals(typeOf<GpuArray<AnyBitwiseSpec>>()),
        );
        expect(
          staticTypeOf(() => _gi32.asIntegerDType << _gi32.asIntegerDType),
          equals(typeOf<GpuArray<AnyIntegerSpec>>()),
        );
        expect(
          staticTypeOf(() => gcd(_gi32.asIntegerDType, _gi32.asIntegerDType)),
          equals(typeOf<GpuArray<AnyIntegerSpec>>()),
        );

        // Run-time-typed results default to AnySpec and infer from `out:`
        expect(
          staticTypeOf(() => concatenate([_g32, _g32])),
          equals(typeOf<GpuArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(() => concatenate([_g32, _g32], out: _g32)),
          equals(typeOf<GpuArray<Float32>>()),
        );
        expect(
          staticTypeOf(() => where(_gb, _g32, _g32)),
          equals(typeOf<GpuArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(() => grandom.permutation(4)),
          equals(typeOf<GpuArray<AnySpec>>()),
        );
      },
    );

    test(
      'negative overlay rejects projecting functions on bare GpuArray<DTypeTag>, mixed-dtype SelfOf calls and bitwise operators on non-integer tags',
      () async {
        final virtualPath = _native(
          '$resolvedPkgRoot/test/meta/_negative_gpu_probes_overlay.dart',
        );
        final lines = _negativeOverlaySource.split('\n');
        final expectedCodesByLine = <int, Set<String>>{};
        final errorCommentRegex = RegExp(r'//\s*error:\s*(.+)$');
        for (var i = 0; i < lines.length; i++) {
          final match = errorCommentRegex.firstMatch(lines[i]);
          if (match != null) {
            expectedCodesByLine[i + 1] = match
                .group(1)!
                .split(',')
                .map((s) => s.trim().toLowerCase())
                .where((s) => s.isNotEmpty)
                .toSet();
          }
        }

        final overlay = OverlayResourceProvider(
          PhysicalResourceProvider.INSTANCE,
        );
        overlay.setOverlay(
          virtualPath,
          content: _negativeOverlaySource,
          modificationStamp: 0,
        );
        final collection = AnalysisContextCollection(
          includedPaths: [virtualPath],
          resourceProvider: overlay,
        );
        final session = collection.contextFor(virtualPath).currentSession;
        final result = await session.getResolvedUnit(virtualPath);
        expect(result, isA<ResolvedUnitResult>());
        final unitResult = result as ResolvedUnitResult;

        final actualCodesByLine = <int, Set<String>>{};
        for (final d in unitResult.diagnostics) {
          if (d.severity != Severity.error) continue;
          final line = unitResult.lineInfo.getLocation(d.offset).lineNumber;
          actualCodesByLine
              .putIfAbsent(line, () => <String>{})
              .add(d.diagnosticCode.lowerCaseName);
        }

        expect(actualCodesByLine, equals(expectedCodesByLine));
      },
    );
  });
}

final class _GpuProbeVisitor extends RecursiveAstVisitor<void> {
  _GpuProbeVisitor(this.lineInfo, this.expectedByLine);

  final LineInfo lineInfo;
  final Map<int, String> expectedByLine;
  final List<String> mismatches = [];
  final List<String> missingExpectations = [];
  int checkedCount = 0;

  @override
  void visitVariableDeclaration(VariableDeclaration node) {
    final name = node.name.lexeme;
    if (name.startsWith('t_')) {
      final declLine = lineInfo.getLocation(node.offset).lineNumber;
      final expected = expectedByLine[declLine] ?? expectedByLine[declLine - 1];
      if (expected == null) {
        missingExpectations.add('L$declLine $name: missing // expect: comment');
      } else {
        checkedCount++;
        final actual = node.declaredFragment?.element.type.getDisplayString();
        if (actual != expected) {
          mismatches.add(
            'L$declLine $name: expected `$expected`, got `$actual`',
          );
        }
      }
    }
    super.visitVariableDeclaration(node);
  }
}
