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
import 'package:test/test.dart';

/// Captures the exact static return type `T` inferred by the Dart compiler for
/// a closure `f` without executing `f`.
Type staticTypeOf<T>(T Function() f) => T;

/// Returns the runtime representation of the static type `T`.
Type typeOf<T>() => T;

late final GpuArray<Float32> _g32;
late final GpuArray<Int64> _gi64;
late final GpuArray<Int32> _gi32;
late final GpuArray<Complex64> _gc64;
late final GpuArray<AnySpec> _ganySpec;

const String _positiveOverlaySource = '''
// ignore_for_file: non_constant_identifier_names, unused_element
import 'package:gpuarray/fft.dart' as gfft;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as glinalg;

late final GpuArray<Float32> g32;
late final GpuArray<Int64> gi64;
late final GpuArray<Int32> gi32;
late final GpuArray<Complex64> gc64;
late final GpuArray<AnySpec> ganySpec;

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
final t_angle_m_f32 = g32.angle();
// expect: GpuArray<Float32>
final t_mean_m_f32 = g32.mean();
// expect: GpuArray<Complex64>
final t_fft_fn_f32 = gfft.fft(g32);
// expect: GpuArray<Complex64>
final t_rfft_fn_f32 = gfft.rfft(g32);
// expect: GpuArray<Float32>
final t_irfft_fn_f32 = gfft.irfft(g32);

// --- GpuArray<Int32> (R5: no true-divide promotion on GPU)
// expect: GpuArray<Int32>
final t_divide_fn_i32 = divide(gi32, gi32);
// expect: GpuArray<Int32>
final t_divide_m_i32 = gi32.divide(gi32);
// expect: GpuArray<Int32>
final t_divide_op_i32 = gi32 / gi32;
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
final t_angle_m_c64 = gc64.angle();
// expect: GpuArray<Complex64>
final t_mean_m_c64 = gc64.mean();
// expect: GpuArray<Complex64>
final t_fft_fn_c64 = gfft.fft(gc64);
// expect: GpuArray<Complex64>
final t_rfft_fn_c64 = gfft.rfft(gc64);
// expect: GpuArray<Float32>
final t_irfft_fn_c64 = gfft.irfft(gc64);

// --- GpuArray<AnySpec>
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_divide_fn_anyspec = divide(ganySpec, ganySpec);
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
final t_divide_m_anyspec = ganySpec.divide(ganySpec);
// expect: GpuArray<DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>>
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
''';

const String _negativeOverlaySource = '''
// ignore_for_file: non_constant_identifier_names, unused_element
import 'package:gpuarray/fft.dart' as gfft;
import 'package:gpuarray/gpuarray.dart';

late final GpuArray<DTypeTag> gdyn;

final t_neg_fft_dyn = gfft.fft(gdyn); // error: argument_type_not_assignable
final t_neg_rfft_dyn = gfft.rfft(gdyn); // error: argument_type_not_assignable
final t_neg_irfft_dyn = gfft.irfft(gdyn); // error: argument_type_not_assignable
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

        // Int32 (R5: divide stays GpuArray<Int32>)
        expect(
          staticTypeOf(() => divide(_gi32, _gi32)),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => _gi32.divide(_gi32)),
          equals(typeOf<GpuArray<Int32>>()),
        );
        expect(
          staticTypeOf(() => _gi32 / _gi32),
          equals(typeOf<GpuArray<Int32>>()),
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
          equals(typeOf<GpuArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(() => _ganySpec.divide(_ganySpec)),
          equals(typeOf<GpuArray<AnySpec>>()),
        );
        expect(
          staticTypeOf(() => _ganySpec / _ganySpec),
          equals(typeOf<GpuArray<AnySpec>>()),
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
      },
    );

    test(
      'negative overlay rejects projecting functions on bare GpuArray<DTypeTag>',
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
