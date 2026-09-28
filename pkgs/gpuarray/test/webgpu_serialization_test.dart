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
import 'package:gpuarray/gpuarray.dart';

void main() {
  group('WebGPU Compute Pipeline Serialization & Browser Widgets', () {
    test(
      'GpuComputePipelinePackage serializes to JSON and deserializes cleanly',
      () {
        final pkg = GpuComputePipelinePackage(
          name: 'test_vector_add',
          wgslCode: '@compute @workgroup_size(256, 1, 1) fn main() {}',
          entryPoint: 'main',
          workgroupSize: [256, 1, 1],
          inputs: [
            GpuBufferPayload(
              bindingIndex: 0,
              name: 'a',
              dtype: DType.float32,
              shape: [1024],
              base64Data: 'AAAA',
              sizeInBytes: 4096,
            ),
            GpuBufferPayload(
              bindingIndex: 1,
              name: 'b',
              dtype: DType.float32,
              shape: [1024],
              base64Data: 'BBBB',
              sizeInBytes: 4096,
            ),
          ],
          output: GpuBufferPayload(
            bindingIndex: 2,
            name: 'c',
            dtype: DType.float32,
            shape: [1024],
            isOutput: true,
            sizeInBytes: 4096,
          ),
          uniforms: [1024, 0, 0, 0],
          sliders: [
            WebGpuSlider(
              name: 'scale',
              label: 'Multiplier',
              min: 0.1,
              max: 5.0,
              initialValue: 1.0,
              uniformWordIndex: 1,
            ),
          ],
          renderToCanvas: true,
          canvasWidth: 256,
          canvasHeight: 256,
          colorMap: 'plasma',
          metadata: {'author': 'scientific_dart'},
        );

        final json = pkg.toJson();
        final roundtrip = GpuComputePipelinePackage.fromJson(json);

        expect(roundtrip.name, equals('test_vector_add'));
        expect(roundtrip.wgslCode, contains('@compute'));
        expect(roundtrip.inputs.length, equals(2));
        expect(roundtrip.inputs[0].name, equals('a'));
        expect(roundtrip.inputs[0].dtype, equals(DType.float32));
        expect(roundtrip.output.isOutput, isTrue);
        expect(roundtrip.sliders.length, equals(1));
        expect(roundtrip.sliders[0].label, equals('Multiplier'));
        expect(roundtrip.renderToCanvas, isTrue);
        expect(roundtrip.canvasWidth, equals(256));
        expect(roundtrip.colorMap, equals('plasma'));
        expect(roundtrip.metadata['author'], equals('scientific_dart'));
      },
    );

    test(
      'toHtml() produces valid self-contained HTML with WebGPU client bootstrap script',
      () {
        final pkg = GpuComputePipelinePackage(
          name: 'fractal_shader',
          wgslCode: '@compute @workgroup_size(16, 16, 1) fn main() {}',
          inputs: [],
          output: GpuBufferPayload(
            bindingIndex: 0,
            name: 'canvas_out',
            dtype: DType.float32,
            shape: [512, 512, 4],
            isOutput: true,
            sizeInBytes: 512 * 512 * 4 * 4,
          ),
          sliders: [
            WebGpuSlider(
              name: 'zoom',
              label: 'Fractal Zoom',
              min: 0.1,
              max: 100.0,
              initialValue: 1.0,
              uniformWordIndex: 0,
            ),
          ],
          renderToCanvas: true,
        );

        final widget = WebGpuWidget(pkg, title: 'Real-Time Fractal Explorer');
        final html = widget.toHtml();

        expect(widget.mimeType, equals('text/html'));
        expect(html, contains('Real-Time Fractal Explorer'));
        expect(html, contains('navigator.gpu'));
        expect(html, contains('createShaderModule'));
        expect(html, contains('createComputePipeline'));
        expect(html, contains('Fractal Zoom'));
        expect(html, contains('<canvas'));
        expect(html, contains('runCompute'));
      },
    );

    test('GpuArray.toWebGpuWidget exports input tensor payload cleanly', () {
      final dev = GpuDevice.cpu(name: 'Test Device');
      final arr = GpuArray.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [4],
        DType.float32,
        device: dev,
      );

      final widget = arr.toWebGpuWidget(title: 'Vector Inspector');
      final html = widget.toHtml();

      expect(html, contains('Vector Inspector'));
      expect(widget.pipeline.name, equals('Vector Inspector'));
      expect(widget.pipeline.inputs.length, equals(1));
      expect(widget.pipeline.inputs[0].shape, equals([4]));
      expect(widget.pipeline.inputs[0].base64Data, isNotNull);

      arr.dispose();
      dev.dispose();
    });

    test('FusedKernelDescriptor creates interactive WebGPU browser widget', () {
      final dev = GpuDevice.cpu(name: 'Test Device');
      final x = GpuArray.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [4],
        DType.float32,
        device: dev,
      );

      final xExpr = Expr.variable('x', bindingIndex: 0);
      final fusedAst = (xExpr * 2.5 + 1.2).silu();

      final desc = FusedKernelDescriptor(
        name: 'silu_interactive',
        expression: fusedAst,
        isStrided: false,
      );

      final widget = desc.createBrowserWidget(
        inputArrays: [x],
        outputShape: [4],
        title: 'Interactive SiLU Browser Widget',
        sliders: [
          WebGpuSlider(
            name: 'multiplier',
            label: 'Input Multiplier',
            min: 0.1,
            max: 10.0,
            initialValue: 2.5,
            uniformWordIndex: 1,
          ),
        ],
      );

      final html = widget.toHtml();
      expect(html, contains('Interactive SiLU Browser Widget'));
      expect(html, contains('Input Multiplier'));
      expect(html, contains('silu_interactive'));
      expect(widget.pipeline.inputs[0].name, equals('x'));

      x.dispose();
      dev.dispose();
    });
  });
}
