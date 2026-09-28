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

import 'package:test/test.dart';
import 'package:gpuarray/wgsl.dart';
import 'package:gpuarray/jit.dart';

void main() {
  group('WGSL Types and Binding Layouts', () {
    test('WgslDType converts from DType correctly', () {
      expect(WgslDType.float32.wgslType, equals('f32'));
      expect(WgslDType.float32.byteSize, equals(4));
      expect(WgslDType.float16.wgslType, equals('f16'));
      expect(WgslDType.int32.wgslType, equals('i32'));
      expect(WgslDType.uint32.wgslType, equals('u32'));
    });

    test('WgslBinding generates valid WGSL variable declarations', () {
      const storageRead = WgslBinding(
        group: 0,
        binding: 0,
        name: 'src_a',
        dtype: WgslDType.float32,
        access: WgslBufferAccess.read,
      );
      expect(
        storageRead.toWgslDeclaration(),
        equals('@group(0) @binding(0) var<storage, read> src_a: array<f32>;'),
      );

      const storageWrite = WgslBinding(
        group: 0,
        binding: 1,
        name: 'dst',
        dtype: WgslDType.float32,
        access: WgslBufferAccess.readWrite,
      );
      expect(
        storageWrite.toWgslDeclaration(),
        equals(
          '@group(0) @binding(1) var<storage, read_write> dst: array<f32>;',
        ),
      );

      const uniformBinding = WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'Uniforms',
      );
      expect(
        uniformBinding.toWgslDeclaration(),
        equals('@group(0) @binding(2) var<uniform> uniforms: Uniforms;'),
      );
    });

    test('WgslWorkgroupSize generates correct attributes', () {
      expect(
        WgslWorkgroupSize.linear1D.toAttribute(),
        equals('@workgroup_size(256)'),
      );
      expect(
        WgslWorkgroupSize.tiled2D.toAttribute(),
        equals('@workgroup_size(16, 16)'),
      );
      expect(
        const WgslWorkgroupSize(8, 8, 4).toAttribute(),
        equals('@workgroup_size(8, 8, 4)'),
      );
    });

    test(
      'WgslShaderModule calculates 1D, 2D, and 3D dispatches accurately',
      () {
        final module1D = WgslShaderModule(
          name: 'test_1d',
          code: '@compute @workgroup_size(256) fn main() {}',
          workgroupSize: WgslWorkgroupSize.linear1D,
        );
        final d1 = module1D.calculateDispatch1D(1000);
        expect(d1.workgroupsX, equals(4)); // ceil(1000 / 256) = 4
        expect(d1.workgroupsY, equals(1));
        expect(d1.workgroupsZ, equals(1));

        final module2D = WgslShaderModule(
          name: 'test_2d',
          code: '@compute @workgroup_size(16, 16) fn main() {}',
          workgroupSize: WgslWorkgroupSize.tiled2D,
        );
        final d2 = module2D.calculateDispatch2D(100, 50);
        expect(d2.workgroupsX, equals(7)); // ceil(100 / 16) = 7
        expect(d2.workgroupsY, equals(4)); // ceil(50 / 16) = 4

        final d3 = module2D.calculateDispatch3D(32, 32, 8);
        expect(d3.workgroupsX, equals(2));
        expect(d3.workgroupsY, equals(2));
        expect(d3.workgroupsZ, equals(8));
      },
    );
  });

  group('WGSL Elementwise Shader Templates', () {
    test('Contiguous binary compute shaders generate valid WGSL', () {
      final ops = ['add', 'sub', 'mul', 'div', 'pow', 'max', 'min', 'eq', 'gt'];
      for (final op in ops) {
        final shader = WgslTemplates.elementwiseBinary(op: op, strided: false);
        expect(shader.name, contains('elementwise_binary_$op'));
        expect(shader.bindings.length, equals(4));
        expect(shader.code, contains('@compute'));
        expect(shader.code, contains('@workgroup_size(256)'));
        expect(shader.code, contains('dst[idx] = '));

        final validation = WgslSyntaxValidator.validate(shader.code);
        expect(
          validation.isValid,
          isTrue,
          reason: 'Failed validation for op $op: ${validation.errors}',
        );
      }
    });

    test('Strided binary compute shaders include multi-index routines', () {
      final shader = WgslTemplates.elementwiseBinary(op: 'add', strided: true);
      expect(shader.code, contains('struct StridedMetadata'));
      expect(shader.code, contains('flat_to_strided_offsets'));
      expect(shader.code, contains('src_a[off_a]'));
      expect(shader.code, contains('src_b[off_b]'));
      expect(shader.code, contains('dst[off_dst] = '));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('Contiguous and strided unary shaders generate valid WGSL', () {
      final unaryOps = [
        'relu',
        'silu',
        'gelu',
        'sigmoid',
        'tanh',
        'exp',
        'log',
        'sqrt',
        'rsqrt',
        'abs',
        'negate',
        'sin',
        'cos',
        'hardswish',
        'mish',
      ];

      for (final op in unaryOps) {
        final shader = WgslTemplates.elementwiseUnary(op: op, strided: false);
        expect(shader.bindings.length, equals(3));
        expect(shader.code, contains('@compute'));
        final val = WgslSyntaxValidator.validate(shader.code);
        expect(
          val.isValid,
          isTrue,
          reason: 'Op $op validation error: ${val.errors}',
        );

        final stridedShader = WgslTemplates.elementwiseUnary(
          op: op,
          strided: true,
        );
        final stridedVal = WgslSyntaxValidator.validate(stridedShader.code);
        expect(
          stridedVal.isValid,
          isTrue,
          reason: 'Strided op $op error: ${stridedVal.errors}',
        );
      }
    });
  });

  group('WGSL Reduction Shaders', () {
    test(
      'getWgslReductionInit handles float16, int32, and float32 data types',
      () {
        // float16
        expect(
          WgslTemplates.getWgslReductionInit('sum', WgslDType.float16),
          equals('0.0h'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('mean', WgslDType.float16),
          equals('0.0h'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('sum_sq', WgslDType.float16),
          equals('0.0h'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('prod', WgslDType.float16),
          equals('1.0h'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('product', WgslDType.float16),
          equals('1.0h'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('min', WgslDType.float16),
          equals('65504.0h'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('max', WgslDType.float16),
          equals('-65504.0h'),
        );

        // int32
        expect(
          WgslTemplates.getWgslReductionInit('sum', WgslDType.int32),
          equals('0'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('mean', WgslDType.int32),
          equals('0'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('sum_sq', WgslDType.int32),
          equals('0'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('prod', WgslDType.int32),
          equals('1'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('product', WgslDType.int32),
          equals('1'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('min', WgslDType.int32),
          equals('2147483647'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('max', WgslDType.int32),
          equals('-2147483648'),
        );

        // float32
        expect(
          WgslTemplates.getWgslReductionInit('sum', WgslDType.float32),
          equals('0.0'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('prod', WgslDType.float32),
          equals('1.0'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('min', WgslDType.float32),
          equals('3.402823e+38'),
        );
        expect(
          WgslTemplates.getWgslReductionInit('max', WgslDType.float32),
          equals('-3.402823e+38'),
        );
      },
    );

    test('Tree reduction compute shader allocates workgroup shared memory', () {
      final reductionOps = ['sum', 'prod', 'min', 'max', 'sum_sq'];
      for (final op in reductionOps) {
        final shader = WgslTemplates.treeReduction(op: op, workgroupSize: 256);
        expect(shader.code, contains('var<workgroup> sdata: array<f32, 256>;'));
        expect(shader.code, contains('workgroupBarrier()'));
        expect(shader.code, contains('dst[workgroup_id.x] = sdata[0];'));

        final validation = WgslSyntaxValidator.validate(shader.code);
        expect(
          validation.isValid,
          isTrue,
          reason: 'Reduction $op failed: ${validation.errors}',
        );
      }
    });
  });

  group('WGSL Matrix Multiplication & Convolution Shaders', () {
    test('Tiled GEMM shader allocates 16x16 shared memory tiles', () {
      final shader = WgslTemplates.tiledMatmul(tileSize: 16, hasBias: false);
      expect(
        shader.code,
        contains('var<workgroup> tile_a: array<array<f32, 16>, 16>;'),
      );
      expect(
        shader.code,
        contains('var<workgroup> tile_b: array<array<f32, 16>, 16>;'),
      );
      expect(shader.code, contains('workgroupBarrier()'));
      expect(shader.code, contains('uniforms.alpha * acc'));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('Tiled GEMM with bias incorporates bias addition', () {
      final shader = WgslTemplates.tiledMatmul(tileSize: 16, hasBias: true);
      expect(shader.bindings.length, equals(5));
      expect(shader.code, contains('bias[col]'));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('Batched GEMM computes batched matrix products', () {
      final shader = WgslTemplates.batchedMatmul(tileSize: 16);
      expect(shader.code, contains('batch_idx = global_id.z'));
      expect(shader.code, contains('batch_stride_a'));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test(
      '2D Convolution shader handles NCHW tensor layout with padding and stride',
      () {
        final shader = WgslTemplates.conv2d(hasBias: true);
        expect(shader.code, contains('struct Conv2dUniforms'));
        expect(shader.code, contains('dilation_h'));
        expect(shader.code, contains('pad_h'));
        expect(shader.code, contains('bias[oc]'));

        final validation = WgslSyntaxValidator.validate(shader.code);
        expect(
          validation.isValid,
          isTrue,
          reason: validation.errors.toString(),
        );
      },
    );

    test(
      '2D Transpose uses padded shared memory tile to avoid bank conflicts',
      () {
        final shader = WgslTemplates.transpose(tileSize: 16);
        expect(shader.code, contains('array<array<f32, 17>, 16>'));
        expect(shader.code, contains('workgroupBarrier()'));

        final validation = WgslSyntaxValidator.validate(shader.code);
        expect(
          validation.isValid,
          isTrue,
          reason: validation.errors.toString(),
        );
      },
    );

    test('Softmax and RMSNorm shaders generate valid WGSL', () {
      final softmaxShader = WgslTemplates.softmax(workgroupSize: 256);
      expect(softmaxShader.code, contains('s_max'));
      expect(softmaxShader.code, contains('s_sum'));
      final vSoftmax = WgslSyntaxValidator.validate(softmaxShader.code);
      expect(vSoftmax.isValid, isTrue, reason: vSoftmax.errors.toString());

      final rmsNormShader = WgslTemplates.rmsNorm(workgroupSize: 256);
      expect(rmsNormShader.code, contains('s_sq'));
      expect(rmsNormShader.code, contains('inverseSqrt'));
      final vRms = WgslSyntaxValidator.validate(rmsNormShader.code);
      expect(vRms.isValid, isTrue, reason: vRms.errors.toString());
    });
  });

  group('Dynamic JIT Kernel Fusion & AST Generator', () {
    test('Builds simple fused expression AST', () {
      final x = Expr.variable('x', bindingIndex: 0);
      final y = Expr.variable('y', bindingIndex: 1);
      final expr = (x * 2.0 + y).silu();

      expect(expr.depth, equals(4));
      expect(expr.variables.map((v) => v.name).toSet(), equals({'x', 'y'}));
      expect(expr.toFingerprint(), contains('silu(add(mul('));
    });

    test('Compiles fused elementwise shader y = silu(a * x + b)', () {
      final a = Expr.variable('a', bindingIndex: 0);
      final x = Expr.variable('x', bindingIndex: 1);
      final b = Expr.variable('b', bindingIndex: 2);
      final expr = (a * x + b).silu();

      final compiler = WgslJitCompiler();
      final module = compiler.compile(expr, kernelName: 'fused_linear_silu');

      expect(module.name, equals('fused_linear_silu'));
      expect(module.bindings.length, equals(5)); // a, x, b, dst, uniforms
      expect(module.code, contains('let a_val = a[idx];'));
      expect(module.code, contains('let x_val = x[idx];'));
      expect(module.code, contains('let b_val = b[idx];'));
      expect(module.code, contains('silu('));

      final val = WgslSyntaxValidator.validate(module.code);
      expect(val.isValid, isTrue, reason: val.errors.toString());
    });

    test('JIT Compiler caches compiled shader modules by fingerprint', () {
      final compiler = WgslJitCompiler();
      compiler.clearCache();
      expect(compiler.cachedCount, equals(0));

      final x = Expr.variable('x', bindingIndex: 0);
      final expr1 = (x * 3.0 + 1.0).relu();
      final expr2 = (x * 3.0 + 1.0).relu(); // Identical structure

      final m1 = compiler.compile(expr1);
      expect(compiler.cacheMisses, equals(1));
      expect(compiler.cacheHits, equals(0));
      expect(compiler.cachedCount, equals(1));

      final m2 = compiler.compile(expr2);
      expect(compiler.cacheHits, equals(1));
      expect(compiler.cacheMisses, equals(1));
      expect(identical(m1, m2), isTrue);

      final diffExpr = (x * 5.0).gelu();
      compiler.compile(diffExpr);
      expect(compiler.cacheMisses, equals(2));
      expect(compiler.cachedCount, equals(2));
    });

    test('Fused kernel with clamp and where / select constructs', () {
      final x = Expr.variable('x', bindingIndex: 0);
      final expr = x.where(x.exp(), x.log()).clamp(0.0, 10.0);

      final compiler = WgslJitCompiler();
      final module = compiler.compile(expr);
      expect(module.code, contains('select('));
      expect(module.code, contains('clamp('));

      final val = WgslSyntaxValidator.validate(module.code);
      expect(val.isValid, isTrue, reason: val.errors.toString());
    });

    test(
      'ConstExpr formats float literals correctly including scientific notation',
      () {
        expect(const ConstExpr(1e-7).toWgsl(), equals('1e-7f'));
        expect(const ConstExpr(2.5).toWgsl(), equals('2.5f'));
        expect(const ConstExpr(42.0).toWgsl(), equals('42.0f'));
        expect(const ConstExpr(0.0).toWgsl(), equals('0.0f'));
        expect(const ConstExpr(-1.23e4).toWgsl(), equals('-12300.0f'));
        expect(const ConstExpr(1e-8).toWgsl(), equals('1e-8f'));
        expect(const ConstExpr(1e21).toWgsl(), equals('1e+21f'));
      },
    );

    test('Multi-input strided fused kernel maps inputs to off_a and off_b', () {
      final a = Expr.variable('a', bindingIndex: 0);
      final b = Expr.variable('b', bindingIndex: 1);
      final expr = a * b + 2.0;

      final compiler = WgslJitCompiler();
      final module = compiler.compile(expr, strided: true);

      expect(module.code, contains('let a_val = a[off_a];'));
      expect(module.code, contains('let b_val = b[off_b];'));
      expect(
        module.code,
        contains(
          'flat_to_strided_offsets(idx, meta, &off_a, &off_b, &off_dst);',
        ),
      );

      final val = WgslSyntaxValidator.validate(module.code);
      expect(val.isValid, isTrue, reason: val.errors.toString());
    });

    test(
      '16-byte aligned uniform buffer padding for 0, 1, 2, and 3 scalar parameters',
      () {
        final x = Expr.variable('x', bindingIndex: 0);
        final compiler = WgslJitCompiler();

        // 0 scalar parameters: 1 field (total_elements) -> needs 3 pad fields (pad0, pad1, pad2)
        final expr0 = x + 1.0;
        final m0 = compiler.compile(expr0);
        expect(m0.code, contains('pad0: u32,'));
        expect(m0.code, contains('pad1: u32,'));
        expect(m0.code, contains('pad2: u32,'));
        expect(m0.code, isNot(contains('pad3: u32,')));
        expect(WgslSyntaxValidator.validate(m0.code).isValid, isTrue);

        // 1 scalar parameter: 2 fields -> needs 2 pad fields (pad0, pad1)
        final s1 = Expr.scalar('scale');
        final expr1 = x * s1;
        final m1 = compiler.compile(expr1);
        expect(m1.code, contains('pad0: u32,'));
        expect(m1.code, contains('pad1: u32,'));
        expect(m1.code, isNot(contains('pad2: u32,')));
        expect(WgslSyntaxValidator.validate(m1.code).isValid, isTrue);

        // 2 scalar parameters: 3 fields -> needs 1 pad field (pad0)
        final s2 = Expr.scalar('bias');
        final expr2 = x * s1 + s2;
        final m2 = compiler.compile(expr2);
        expect(m2.code, contains('pad0: u32,'));
        expect(m2.code, isNot(contains('pad1: u32,')));
        expect(WgslSyntaxValidator.validate(m2.code).isValid, isTrue);

        // 3 scalar parameters: 4 fields -> needs 0 pad fields (already 16-byte aligned)
        final s3 = Expr.scalar('gamma');
        final expr3 = x * s1 + s2 + s3;
        final m3 = compiler.compile(expr3);
        expect(m3.code, isNot(contains('pad0: u32,')));
        expect(WgslSyntaxValidator.validate(m3.code).isValid, isTrue);
      },
    );

    test(
      'WgslJitCompiler enforces LRU cache eviction when capacity is exceeded',
      () {
        final compiler = WgslJitCompiler(maxCacheSize: 2);
        final x = Expr.variable('x', bindingIndex: 0);

        final exprA = (x + 1.0).relu();
        final exprB = (x + 2.0).relu();
        final exprC = (x + 3.0).relu();

        final mA = compiler.compile(exprA);
        expect(compiler.cachedCount, equals(1));

        final mB = compiler.compile(exprB);
        expect(compiler.cachedCount, equals(2));

        // Access A to make it most recently used (MRU), B becomes least recently used (LRU)
        final mA2 = compiler.compile(exprA);
        expect(identical(mA, mA2), isTrue);
        expect(compiler.cacheHits, equals(1));

        // Compiling C should evict B (the LRU entry)
        final mC = compiler.compile(exprC);
        expect(compiler.cachedCount, equals(2));
        expect(compiler.cacheMisses, equals(3));

        // A and C should be cached, B should not
        expect(compiler.isCached(mA.metadata['cacheKey'] as String), isTrue);
        expect(compiler.isCached(mB.metadata['cacheKey'] as String), isFalse);
        expect(compiler.isCached(mC.metadata['cacheKey'] as String), isTrue);
      },
    );

    test(
      'Strided fused kernel generates multidimensional offset translation',
      () {
        final x = Expr.variable('x', bindingIndex: 0);
        final y = Expr.variable('y', bindingIndex: 1);
        final expr = (x + y).gelu();

        final compiler = WgslJitCompiler();
        final module = compiler.compile(expr, strided: true);

        expect(module.code, contains('flat_to_strided_offsets'));
        expect(module.code, contains('struct StridedMetadata'));

        final val = WgslSyntaxValidator.validate(module.code);
        expect(val.isValid, isTrue, reason: val.errors.toString());
      },
    );
  });

  group('WGSL Syntax Validator Error Detection', () {
    test('Catches unmatched braces and parentheses', () {
      const badBrace =
          '@compute @workgroup_size(256) fn main() { dst[0] = 1.0;';
      final v1 = WgslSyntaxValidator.validate(badBrace);
      expect(v1.isValid, isFalse);
      expect(v1.errors.any((e) => e.contains('Unclosed brace')), isTrue);

      const badParen = '@compute @workgroup_size(256 fn main() {}';
      final v2 = WgslSyntaxValidator.validate(badParen);
      expect(v2.isValid, isFalse);
    });

    test('Catches duplicate resource bindings', () {
      const dupBinding = '''
@group(0) @binding(0) var<storage, read> src_a: array<f32>;
@group(0) @binding(0) var<storage, read> src_b: array<f32>;
@compute @workgroup_size(256) fn main() {}
''';
      final v = WgslSyntaxValidator.validate(dupBinding);
      expect(v.isValid, isFalse);
      expect(
        v.errors.any((e) => e.contains('Duplicate resource binding')),
        isTrue,
      );
    });

    test('Catches invalid storage access qualifier', () {
      const badAccess = '''
@group(0) @binding(0) var<storage, write_only> src_a: array<f32>;
@compute @workgroup_size(256) fn main() {}
''';
      final v = WgslSyntaxValidator.validate(badAccess);
      expect(v.isValid, isFalse);
      expect(
        v.errors.any(
          (e) => e.contains('Invalid storage buffer access qualifier'),
        ),
        isTrue,
      );
    });

    test(
      'Syntax validator ignores comments containing unmatched parentheses or braces',
      () {
        const codeWithComments = '''
// Compute shader (with unmatched parenthesis in comment
/* Another comment with [unmatched bracket and {brace */
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // Nested (unmatched ) paren) in comment
  /* multi-line
     comment with ( unmatched parenthesis */
  let idx = global_id.x;
  dst[idx] = src[idx] + 1.0f;
}
''';
        final result = WgslSyntaxValidator.validate(codeWithComments);
        expect(result.isValid, isTrue, reason: result.errors.toString());
      },
    );
  });

  group('WGSL Loop Expressions & AST Fusion', () {
    test('Compiles functional loop AST into valid WGSL shader', () {
      final xIn = GpuExpr.variable('x_grid');
      final yIn = GpuExpr.variable('y_grid');
      final zoom = GpuExpr.scalar('zoom', defaultValue: 2.5);
      final centerX = GpuExpr.scalar('center_x', defaultValue: -0.7);
      final centerY = GpuExpr.scalar('center_y', defaultValue: 0.0);
      final maxIter = GpuExpr.scalar('max_iter', defaultValue: 64.0);

      final cr = xIn * zoom + centerX;
      final ci = yIn * zoom + centerY;

      final mandelbrotExpr = GpuExpr.loop(
        initialValues: [
          GpuExpr.constant(0.0),
          GpuExpr.constant(0.0),
          GpuExpr.constant(0.0),
        ],
        maxIterations: maxIter,
        condition: (s, i) => (s[0] * s[0] + s[1] * s[1]).lessThan(4.0),
        step: (s, i) => [
          s[0] * s[0] - s[1] * s[1] + cr,
          s[0] * s[1] * 2.0 + ci,
          s[2] + 1.0,
        ],
        result: (s) => s[2] / maxIter,
        name: 'mandelbrot_core',
      );

      final descriptor = FusedKernelDescriptor(
        name: 'mandelbrot_loop_test',
        outputExpr: mandelbrotExpr,
      );

      final wgsl = descriptor.generateWgslSource();
      expect(wgsl, contains('fn mandelbrot_core('));
      expect(wgsl, contains('while ('));
      expect(wgsl, contains('mandelbrot_core(x_grid_val, y_grid_val)'));

      final val = WgslSyntaxValidator.validate(wgsl);
      expect(val.isValid, isTrue, reason: val.errors.toString());
    });
  });
}
