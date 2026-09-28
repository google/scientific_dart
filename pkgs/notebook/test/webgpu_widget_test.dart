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
import 'package:notebook/notebook.dart';

void main() {
  group('Notebook WebGPU Widget Integration', () {
    test(
      'WebGpuShaderWidget formats into Displayable HTML with WebGPU client bootstrap',
      () {
        final widget = WebGpuShaderWidget(
          title: 'Mandelbrot Fractal Compute',
          wgsl: '@compute @workgroup_size(16, 16, 1) fn main() {}',
          entryPoint: 'main',
          workgroups: [32, 32, 1],
          uniforms: [512, 512, 100, 0],
        );

        expect(widget.mimeType, equals('text/html'));
        final html = widget.toHtml();
        expect(html, contains('Mandelbrot Fractal Compute'));
        expect(html, contains('navigator.gpu'));
        expect(html, contains('createShaderModule'));
        expect(html, contains('createComputePipeline'));
      },
    );

    test('display() and prettyFormat() capture objects providing toHtml()', () {
      clearCapturedOutput();
      final widget = WebGpuShaderWidget(
        title: 'Neural Activation Visualizer',
        wgsl: '@compute @workgroup_size(256, 1, 1) fn main() {}',
      );

      display(widget);
      final jsonOutput = getCapturedOutputsJson();
      expect(jsonOutput, contains('text/html'));
      expect(jsonOutput, contains('Neural Activation Visualizer'));

      final formatted = prettyFormat(widget);
      expect(formatted, contains('Neural Activation Visualizer'));
      expect(formatted, contains('navigator.gpu'));
    });
  });
}
