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

import "dart:math" as math;
import "package:gpuarray/nn.dart" as nn;
import "package:gpuarray/src/dtype.dart";
import "package:gpuarray/src/gpu_array.dart" hide ResourceScope, ScopedResource;
import "package:resource_scope/resource_scope.dart";
import "package:test/test.dart";

void main() {
  group("Modern Transformer & LLM Primitives (Stream 3)", () {
    test("Scaled Dot-Product Attention (SDPA) forward and causal masking", () {
      ResourceScope.scope(() {
        // Query, Key, Value: shape [1, 1, 3, 2] (B=1, H=1, seqLen=3, headDim=2)
        final q = GpuArray.fromList(
          [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [1, 1, 3, 2],
          DType.float64,
          requiresGrad: true,
        );

        final k = GpuArray.fromList(
          [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [1, 1, 3, 2],
          DType.float64,
          requiresGrad: true,
        );

        final v = GpuArray.fromList(
          [10.0, 0.0, 0.0, 20.0, 30.0, 30.0],
          [1, 1, 3, 2],
          DType.float64,
          requiresGrad: true,
        );

        // 1. Standard unmasked SDPA
        final out = nn.scaled_dot_product_attention(q, k, v);
        expect(out.shape, equals([1, 1, 3, 2]));
        expect(out.requiresGrad, isTrue);

        // 2. Causal masked SDPA
        final causalOut = nn.scaled_dot_product_attention(
          q,
          k,
          v,
          isCausal: true,
        );
        expect(causalOut.shape, equals([1, 1, 3, 2]));

        final causalList = causalOut.toList().cast<double>();
        // First token can only attend to position 0 -> output must be exactly v[0] = [10.0, 0.0]
        expect(causalList[0], closeTo(10.0, 1e-4));
        expect(causalList[1], closeTo(0.0, 1e-4));

        // 3. Backward differentiation
        final loss = causalOut.sum();
        loss.backward();

        expect(q.grad, isNotNull);
        expect(q.grad!.shape, equals([1, 1, 3, 2]));
        expect(k.grad, isNotNull);
        expect(k.grad!.shape, equals([1, 1, 3, 2]));
        expect(v.grad, isNotNull);
        expect(v.grad!.shape, equals([1, 1, 3, 2]));
      });
    });

    test("SDPA with boolean and float custom attention masks", () {
      ResourceScope.scope(() {
        final q = GpuArray.ones([2, 4], DType.float64, requiresGrad: true);
        final k = GpuArray.ones([2, 4], DType.float64, requiresGrad: true);
        final v = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
          [2, 4],
          DType.float64,
          requiresGrad: true,
        );

        // Boolean mask where position 0 only attends to position 0
        final boolMask = GpuArray.fromList(
          [true, false, true, true],
          [2, 2],
          DType.boolean,
        );

        final out = nn.scaled_dot_product_attention(
          q,
          k,
          v,
          attnMask: boolMask,
        );
        expect(out.shape, equals([2, 4]));

        final outList = out.toList().cast<double>();
        // First token attends ONLY to pos 0, so out[0] = v[0] = [1.0, 2.0, 3.0, 4.0]
        expect(outList.sublist(0, 4), equals([1.0, 2.0, 3.0, 4.0]));

        out.sum().backward();
        expect(q.grad, isNotNull);
        expect(k.grad, isNotNull);
        expect(v.grad, isNotNull);
      });
    });

    test(
      "MultiheadAttention forward pass, head splitting, and autograd backward",
      () {
        ResourceScope.scope(() {
          const embedDim = 8;
          const numHeads = 2;
          const seqLen = 4;
          const batchSize = 2;

          final mha = nn.MultiheadAttention(embedDim, numHeads);
          expect(mha.parameters.length, equals(8)); // 4 weights + 4 biases

          final input = GpuArray.ones(
            [batchSize, seqLen, embedDim],
            DType.float64,
            requiresGrad: true,
          );

          final out = mha(input, isCausal: true);
          expect(out.shape, equals([batchSize, seqLen, embedDim]));
          expect(out.requiresGrad, isTrue);

          final loss = out.sum();
          loss.backward();

          expect(mha.qProj.weight.grad, isNotNull);
          expect(mha.qProj.weight.grad!.shape, equals([embedDim, embedDim]));
          expect(mha.kProj.weight.grad, isNotNull);
          expect(mha.vProj.weight.grad, isNotNull);
          expect(mha.outProj.weight.grad, isNotNull);
          expect(input.grad, isNotNull);
          expect(input.grad!.shape, equals([batchSize, seqLen, embedDim]));

          final mhaAlias = nn.MultiHeadAttention(embedDim, numHeads);
          expect(mhaAlias, isA<nn.MultiheadAttention>());
        });
      },
    );

    test("MultiheadAttention with 2D inputs and cross-attention key/value", () {
      ResourceScope.scope(() {
        final mha = nn.MultiheadAttention(4, 2, kdim: 6, vdim: 6);

        final query = GpuArray.ones([3, 4], DType.float64, requiresGrad: true);
        final key = GpuArray.ones([5, 6], DType.float64, requiresGrad: true);
        final value = GpuArray.ones([5, 6], DType.float64, requiresGrad: true);

        final out = mha(query, key: key, value: value);
        expect(out.shape, equals([3, 4]));

        out.sum().backward();
        expect(query.grad, isNotNull);
        expect(key.grad, isNotNull);
        expect(value.grad, isNotNull);
        expect(mha.kProj.weight.grad!.shape, equals([4, 6]));
      });
    });

    test("RMSNorm forward accuracy and autograd backward", () {
      ResourceScope.scope(() {
        final norm = nn.RMSNorm([4]);
        expect(norm.weight.shape, equals([4]));
        expect(norm.weight.toList(), equals([1.0, 1.0, 1.0, 1.0]));

        // Vector: [2.0, 2.0, 2.0, 2.0] -> RMS is 2.0 -> Normalized should be [1.0, 1.0, 1.0, 1.0]
        final x = GpuArray.fromList(
          [2.0, 2.0, 2.0, 2.0],
          [1, 4],
          DType.float64,
          requiresGrad: true,
        );

        final out = norm(x);
        expect(out.shape, equals([1, 4]));
        final outList = out.toList().cast<double>();
        for (final val in outList) {
          expect(val, closeTo(1.0, 1e-4));
        }

        final loss = out.sum();
        loss.backward();

        expect(norm.weight.grad, isNotNull);
        expect(norm.weight.grad!.shape, equals([4]));
        expect(x.grad, isNotNull);
        expect(x.grad!.shape, equals([1, 4]));
      });
    });

    test(
      "RotaryEmbedding (RoPE) forward rotation, norm preservation, and backward",
      () {
        ResourceScope.scope(() {
          const dim = 4;
          const seqLen = 3;
          final rope = nn.RotaryEmbedding(dim, maxSequenceLength: 16);

          final x = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, -1.0, 0.0, 1.0, 2.0],
            [seqLen, dim],
            DType.float64,
            requiresGrad: true,
          );

          final out = rope(x);
          expect(out.shape, equals([seqLen, dim]));

          // Position 0 should have theta = 0 -> cos = 1, sin = 0 -> out[0] == x[0]
          final xList = x.toList().cast<double>();
          final outList = out.toList().cast<double>();
          for (var i = 0; i < dim; i++) {
            expect(outList[i], closeTo(xList[i], 1e-5));
          }

          // L2 Norm preservation: for any token position p, ||rope(x[p])||_2 == ||x[p]||_2
          for (var p = 0; p < seqLen; p++) {
            var normX = 0.0;
            var normOut = 0.0;
            for (var d = 0; d < dim; d++) {
              final xv = xList[p * dim + d];
              final ov = outList[p * dim + d];
              normX += xv * xv;
              normOut += ov * ov;
            }
            expect(math.sqrt(normOut), closeTo(math.sqrt(normX), 1e-4));
          }

          final loss = out.sum();
          loss.backward();
          expect(x.grad, isNotNull);
          expect(x.grad!.shape, equals([seqLen, dim]));
        });
      },
    );

    test("SwiGLU & GeGLU gated linear units forward and backward", () {
      ResourceScope.scope(() {
        final swiglu = nn.SwiGLU(4, 8, outFeatures: 4);
        expect(swiglu.parameters.length, equals(3)); // w1, w2, w3

        final geglu = nn.GeGLU(4, 8, outFeatures: 4);
        expect(geglu.parameters.length, equals(3));

        final x = GpuArray.ones([2, 4], DType.float64, requiresGrad: true);

        final swiOut = swiglu(x);
        expect(swiOut.shape, equals([2, 4]));
        swiOut.sum().backward();

        expect(swiglu.w1.weight.grad, isNotNull);
        expect(swiglu.w2.weight.grad, isNotNull);
        expect(swiglu.w3.weight.grad, isNotNull);
        expect(x.grad, isNotNull);

        final x2 = GpuArray.ones([2, 4], DType.float64, requiresGrad: true);
        final geOut = geglu(x2);
        expect(geOut.shape, equals([2, 4]));
        geOut.sum().backward();
        expect(x2.grad, isNotNull);
      });
    });

    test(
      "TransformerEncoderLayer forward and backward (Post-LN and Pre-LN)",
      () {
        ResourceScope.scope(() {
          // Post-LN
          final postLayer = nn.TransformerEncoderLayer(
            8,
            2,
            dimFeedforward: 16,
            normFirst: false,
          );
          final x1 = GpuArray.ones(
            [2, 3, 8],
            DType.float64,
            requiresGrad: true,
          );
          final outPost = postLayer(x1);
          expect(outPost.shape, equals([2, 3, 8]));
          outPost.sum().backward();
          expect(x1.grad, isNotNull);
          expect(postLayer.selfAttn.qProj.weight.grad, isNotNull);
          expect(postLayer.linear1.weight.grad, isNotNull);

          // Pre-LN
          final preLayer = nn.TransformerEncoderLayer(
            8,
            2,
            dimFeedforward: 16,
            normFirst: true,
          );
          final x2 = GpuArray.ones(
            [2, 3, 8],
            DType.float64,
            requiresGrad: true,
          );
          final outPre = preLayer(x2, isCausal: true);
          expect(outPre.shape, equals([2, 3, 8]));
          outPre.sum().backward();
          expect(x2.grad, isNotNull);
          expect(preLayer.selfAttn.qProj.weight.grad, isNotNull);
        });
      },
    );

    test(
      "TransformerDecoderLayer forward and backward with cross-attention",
      () {
        ResourceScope.scope(() {
          final decLayer = nn.TransformerDecoderLayer(8, 2, dimFeedforward: 16);
          final tgt = GpuArray.ones(
            [2, 4, 8],
            DType.float64,
            requiresGrad: true,
          );
          final memory = GpuArray.ones(
            [2, 6, 8],
            DType.float64,
            requiresGrad: true,
          );

          final out = decLayer(tgt, memory: memory);
          expect(out.shape, equals([2, 4, 8]));

          final loss = out.sum();
          loss.backward();

          expect(tgt.grad, isNotNull);
          expect(memory.grad, isNotNull);
          expect(decLayer.selfAttn.qProj.weight.grad, isNotNull);
          expect(decLayer.multiheadAttn.kProj.weight.grad, isNotNull);
        });
      },
    );
  });
}
