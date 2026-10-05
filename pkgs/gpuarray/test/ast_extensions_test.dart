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

import 'dart:math' as math;
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/jit.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:test/test.dart';

void main() {
  group('Feature 1: Intrinsic Grid & Coordinate Expressions', () {
    test('CoordExpr generates valid 1D and 2D WGSL coordinate expressions', () {
      final xCoord = Expr.coord(1, shape: [256, 512], normalized: true);
      final yCoord = Expr.coord(0, shape: [256, 512], normalized: true);
      final unnormX = Expr.coord(1, shape: [256, 512], normalized: false);
      final index = Expr.index();

      expect(xCoord.toWgsl(), contains('idx % 512u'));
      expect(xCoord.toWgsl(), contains('512.0f'));
      expect(yCoord.toWgsl(), contains('(idx / 512u) % 256u'));
      expect(unnormX.toWgsl(), equals('f32(idx % 512u)'));
      expect(index.toWgsl(), equals('f32(idx)'));
    });

    test('Zero-input compute shader compiles without input buffers', () {
      final width = 256;
      final height = 256;
      final x = Expr.coord(1, shape: [height, width], normalized: true);
      final y = Expr.coord(0, shape: [height, width], normalized: true);
      final radius = (x * x + y * y).sqrt();

      final desc = FusedKernelDescriptor(
        name: 'radial_gradient',
        outputExpr: radius,
      );

      expect(desc.inputs, isEmpty);
      final wgsl = desc.generateWgslSource();
      WgslSyntaxValidator.validate(wgsl);

      expect(wgsl, contains('fn main('));
      expect(wgsl, contains('dst[idx] = result;'));
      expect(wgsl, isNot(contains('@binding(0) var<storage, read> input_')));
      expect(
        wgsl,
        contains('@binding(0) var<storage, read_write> dst: array<f32>;'),
      );

      // Also verify createBrowserWidget works with no inputArrays passed
      final widget = desc.createBrowserWidget(
        outputShape: [height, width],
        title: 'Radial Gradient Demo',
      );
      expect(widget.pipelinePackage.inputs, isEmpty);
      expect(widget.pipelinePackage.output.shape, equals([height, width]));
    });
  });

  group('Feature 2: Graphics & Shader Math Intrinsics', () {
    test(
      'mix, smoothstep, step, mod, fract, atan2, hypot, sign emit valid WGSL',
      () {
        final x = Expr.variable('x');
        final y = Expr.variable('y');
        final t = Expr.variable('t');

        final mixed = x.mix(y, t);
        final sstep = x.smoothstep(0.0, 1.0);
        final stepped = x.step(0.5);
        final modded = x % y;
        final fractured = x.fract();
        final angle = y.atan2(x);
        final hyp = x.hypot(y);
        final sgn = x.sign();
        final hsinh = x.sinh();
        final hcosh = x.cosh();

        expect(mixed.toWgsl(), equals('mix(x_val, y_val, t_val)'));
        expect(sstep.toWgsl(), equals('smoothstep(0.0f, 1.0f, x_val)'));
        expect(stepped.toWgsl(), contains('step(0.5f, x_val)'));
        expect(modded.toWgsl(), contains('x_val % y_val'));
        expect(fractured.toWgsl(), equals('fract((x_val))'));
        expect(angle.toWgsl(), contains('atan2(y_val, x_val)'));
        expect(
          hyp.toWgsl(),
          contains('sqrt((x_val * x_val) + (y_val * y_val))'),
        );
        expect(sgn.toWgsl(), equals('sign((x_val))'));
        expect(hsinh.toWgsl(), equals('sinh((x_val))'));
        expect(hcosh.toWgsl(), equals('cosh((x_val))'));

        final desc = FusedKernelDescriptor(
          name: 'shader_intrinsics_kernel',
          outputExpr:
              mixed +
              sstep +
              stepped +
              modded +
              fractured +
              angle +
              hyp +
              sgn +
              hsinh +
              hcosh,
        );
        final wgsl = desc.generateWgslSource();
        WgslSyntaxValidator.validate(wgsl);
      },
    );
  });

  group('Feature 3: Logical Operators & Boolean Combinators', () {
    test(
      'Logical and (&), or (|), not (~) emit valid WGSL select predicates',
      () {
        final a = Expr.variable('a');
        final b = Expr.variable('b');

        final andExpr = a & b;
        final orExpr = a | b;
        final notExpr = ~a;
        final fluentAnd = a.and(b);
        final fluentOr = a.or(b);
        final fluentNot = a.not();

        expect(andExpr.toWgsl(), contains('(a_val > 0.0) && (b_val > 0.0)'));
        expect(orExpr.toWgsl(), contains('(a_val > 0.0) || (b_val > 0.0)'));
        expect(notExpr.toWgsl(), contains('select(0.0, 1.0, (a_val) <= 0.0)'));
        expect(fluentAnd.toWgsl(), equals(andExpr.toWgsl()));
        expect(fluentOr.toWgsl(), equals(orExpr.toWgsl()));
        expect(fluentNot.toWgsl(), equals(notExpr.toWgsl()));

        // Composite condition: (a > 0 && b > 0) || !(a < 5)
        final cond =
            (a.greaterThan(0.0) & b.greaterThan(0.0)) | ~(a.lessThan(5.0));
        final desc = FusedKernelDescriptor(
          name: 'logical_kernel',
          outputExpr: cond.where(10.0, -10.0),
        );
        final wgsl = desc.generateWgslSource();
        WgslSyntaxValidator.validate(wgsl);
        expect(wgsl, contains('select('));
      },
    );
  });

  group(
    'Feature 4: Local Variables & Common Subexpression Elimination (CSE)',
    () {
      test('Expr.let binds and evaluates local variables in WGSL', () {
        final x = Expr.variable('x');
        final letExpr = Expr.let(x * x + 2.0, (val) => val * val + val);

        final desc = FusedKernelDescriptor(
          name: 'let_test_kernel',
          outputExpr: letExpr,
        );
        final wgsl = desc.generateWgslSource();
        WgslSyntaxValidator.validate(wgsl);

        expect(wgsl, contains('let _let_0 = ((x_val * x_val) + 2.0f);'));
        expect(wgsl, contains('(_let_0 * _let_0) + _let_0'));
      });

      test(
        'eliminateCommonSubexpressions automatically hoists repeated subexpressions',
        () {
          final x = Expr.variable('x');
          final y = Expr.variable('y');

          // (x * y + 3.0) repeated three times
          final term = x * y + 3.0;
          final expr = term * term + term.sqrt();

          final optimized = expr.eliminateCommonSubexpressions();
          expect(optimized, isA<LetExpr>());

          final desc = FusedKernelDescriptor(
            name: 'cse_kernel',
            outputExpr: expr,
          );
          final wgsl = desc.generateWgslSource(enableCse: true);
          WgslSyntaxValidator.validate(wgsl);

          expect(wgsl, contains('let _cse_'));
          expect(wgsl, contains('(x_val * y_val)'));
          expect(wgsl, contains('_cse_0'));
        },
      );
    },
  );

  group('Feature 5: Stencil & Neighborhood Sampling', () {
    test(
      'OffsetVarExpr generates clamp, wrap, and zero boundary helper functions',
      () {
        final grid = Expr.variable('grid');
        final leftClamp = grid.offset(
          [0, -1],
          shape: [64, 64],
          boundary: BoundaryMode.clamp,
        );
        final topWrap = grid.offset(
          [-1, 0],
          shape: [64, 64],
          boundary: BoundaryMode.wrap,
        );
        final cornerZero = grid.offset(
          [1, 1],
          shape: [64, 64],
          boundary: BoundaryMode.zero,
        );

        final laplacian = leftClamp + topWrap + cornerZero - grid * 3.0;
        final desc = FusedKernelDescriptor(
          name: 'stencil_kernel',
          outputExpr: laplacian,
        );

        final wgsl = desc.generateWgslSource();
        WgslSyntaxValidator.validate(wgsl);

        expect(wgsl, contains('fn stencil_grid_p0_m1_clamp('));
        expect(wgsl, contains('fn stencil_grid_m1_p0_wrap('));
        expect(wgsl, contains('fn stencil_grid_p1_p1_zero('));
        expect(wgsl, contains('clamp(c + (-1), 0, i32(W - 1u))'));
        expect(wgsl, contains('((r + (-1)) % i32(H) + i32(H)) % i32(H)'));
        expect(wgsl, contains('return 0.0f;'));
      },
    );

    test(
      'Conway Game of Life 9-point neighborhood stencil compiles cleanly',
      () {
        final board = Expr.variable('board');
        const shape = [128, 128];
        const wrap = BoundaryMode.wrap;

        final neighbors =
            board.offset([-1, -1], shape: shape, boundary: wrap) +
            board.offset([-1, 0], shape: shape, boundary: wrap) +
            board.offset([-1, 1], shape: shape, boundary: wrap) +
            board.offset([0, -1], shape: shape, boundary: wrap) +
            board.offset([0, 1], shape: shape, boundary: wrap) +
            board.offset([1, -1], shape: shape, boundary: wrap) +
            board.offset([1, 0], shape: shape, boundary: wrap) +
            board.offset([1, 1], shape: shape, boundary: wrap);

        // Alive if neighbors == 3 or (board == 1 and neighbors == 2)
        final willLive =
            neighbors.equal(3.0) | (board.equal(1.0) & neighbors.equal(2.0));
        final nextState = willLive.where(1.0, 0.0);

        final desc = FusedKernelDescriptor(
          name: 'game_of_life',
          outputExpr: nextState,
        );
        final wgsl = desc.generateWgslSource();
        WgslSyntaxValidator.validate(wgsl);

        expect(wgsl, contains('stencil_board_m1_m1_wrap'));
        expect(wgsl, contains('stencil_board_p1_p1_wrap'));
      },
    );
  });

  group('Feature 6: Symbolic Auto-Diff on Expr AST', () {
    test(
      'Polynomial differentiation matches calculus: d/dx(x^3 + 2x) = 3x^2 + 2',
      () {
        final x = Expr.variable('x');
        final poly = x.pow(3.0) + x * 2.0;
        final dpoly = poly.grad(x);

        expect(dpoly.toFingerprint(), contains('pow'));
        expect(dpoly.toFingerprint(), contains('3.0'));

        final desc = FusedKernelDescriptor(
          name: 'poly_grad',
          outputExpr: dpoly,
        );
        final wgsl = desc.generateWgslSource();
        WgslSyntaxValidator.validate(wgsl);
      },
    );

    test('Quotient rule: d/dx(x / (x + 1)) = 1 / (x + 1)^2', () {
      final x = Expr.variable('x');
      final f = x / (x + 1.0);
      final df = f.grad(x);

      final desc = FusedKernelDescriptor(name: 'quotient_grad', outputExpr: df);
      final wgsl = desc.generateWgslSource();
      WgslSyntaxValidator.validate(wgsl);
    });

    test(
      'Activation derivatives (sin, cos, exp, log, silu, sigmoid, tanh, relu)',
      () {
        final x = Expr.variable('x');

        final funcs = <String, Expr>{
          'sin': x.sin().grad(x),
          'cos': x.cos().grad(x),
          'exp': x.exp().grad(x),
          'log': x.log().grad(x),
          'sigmoid': x.sigmoid().grad(x),
          'silu': x.silu().grad(x),
          'tanh': x.tanh().grad(x),
          'relu': x.relu().grad(x),
        };

        for (final entry in funcs.entries) {
          final desc = FusedKernelDescriptor(
            name: '${entry.key}_grad',
            outputExpr: entry.value,
          );
          final wgsl = desc.generateWgslSource();
          WgslSyntaxValidator.validate(wgsl);
        }
      },
    );

    test('Multi-variable partial derivatives df/dx and df/dy', () {
      final x = Expr.variable('x', bindingIndex: 0);
      final y = Expr.variable('y', bindingIndex: 1);

      // f(x, y) = x^2 + 3*x*y + y^2
      final f = x * x + x * y * 3.0 + y * y;
      final dfDx = f.grad(x);
      final dfDy = f.grad(y);

      expect(dfDx.variables, contains(x));
      expect(dfDy.variables, contains(y));

      final descX = FusedKernelDescriptor(name: 'partial_x', outputExpr: dfDx);
      final descY = FusedKernelDescriptor(name: 'partial_y', outputExpr: dfDy);
      WgslSyntaxValidator.validate(descX.generateWgslSource());
      WgslSyntaxValidator.validate(descY.generateWgslSource());
    });

    test(
      'Numerical finite difference verification of analytical gradients',
      () {
        final xVal = 1.25;
        final eps = 1e-5;

        double evaluate(double val) {
          return math.exp(math.sin(val)) + val * val;
        }

        final numericalGrad =
            (evaluate(xVal + eps) - evaluate(xVal - eps)) / (2.0 * eps);
        final analyticalGrad =
            math.cos(xVal) * math.exp(math.sin(xVal)) + 2.0 * xVal;

        expect(analyticalGrad, closeTo(numericalGrad, 1e-4));
      },
    );
  });

  group('Feature 18: WgslJitCompiler & CompiledWgslKernel GpuArray Execution', () {
    test(
      'compileKernel and execute evaluate fused AST on GpuArray<Float32> and GpuArray<Float64>',
      () {
        ResourceScope.scope(() {
          final x = Expr.variable('x', bindingIndex: 0);
          final y = Expr.variable('y', bindingIndex: 1);
          final scale = Expr.scalar('scale', defaultValue: 2.0);
          final expr = (x * scale + y).relu();

          final kernel = WgslJitCompiler.instance.compileKernel(
            expr,
            name: 'fused_axpy_relu',
          );
          expect(kernel.descriptor.name, equals('fused_axpy_relu'));

          final a32 = GpuArray<Float32>.fromList(
            [-2.0, 1.0, 3.0, -0.5],
            [4],
            DType.float32,
          );
          final b32 = GpuArray<Float32>.fromList(
            [1.0, 2.0, -1.0, 2.0],
            [4],
            DType.float32,
          );

          // Execute via named inputs with default scalar (scale = 2.0)
          final out32 = kernel<Float32>({'x': a32, 'y': b32});
          expect(out32.dtype, equals(DType.float32));
          expect(out32.shape, equals([4]));
          expect(out32.toList().cast<double>(), equals([0.0, 4.0, 5.0, 1.0]));

          // Execute with scalar override (scale = 3.0) and preallocated out:
          final preallocated = GpuArray<Float32>.empty([4], DType.float32);
          final outOverride = kernel.executePositional<Float32>(
            [a32, b32],
            scalars: {'scale': 3.0},
            out: preallocated,
          );
          expect(identical(outOverride, preallocated), isTrue);
          expect(
            outOverride.toList().cast<double>(),
            equals([0.0, 5.0, 8.0, 0.5]),
          );

          // Execute via WgslJitCompiler.instance.execute on Float64 inputs
          final a64 = GpuArray<Float64>.fromList(
            [1.0, 2.0],
            [2],
            DType.float64,
          );
          final b64 = GpuArray<Float64>.fromList(
            [0.5, 1.5],
            [2],
            DType.float64,
          );
          final out64 = WgslJitCompiler.instance.execute<Float64>(
            expr,
            {'x': a64, 'y': b64},
            scalars: {'scale': 4.0},
          );
          expect(out64.dtype, equals(DType.float64));
          expect(out64.toList().cast<double>(), equals([4.5, 9.5]));
        });
      },
    );

    test(
      'Zero-input procedural CoordExpr kernel executes on GPU with explicit outputShape and dtype',
      () {
        ResourceScope.scope(() {
          final col = Expr.coord(1, shape: [2, 3], normalized: false);
          final row = Expr.coord(0, shape: [2, 3], normalized: false);
          final expr = row * 10.0 + col;

          final out = WgslJitCompiler.instance.execute<Float32>(
            expr,
            const {},
            outputShape: [2, 3],
            dtype: DType.float32,
          );
          expect(out.shape, equals([2, 3]));
          expect(out.dtype, equals(DType.float32));
          expect(
            out.toList().cast<double>(),
            equals([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
          );
        });
      },
    );
  });
}
