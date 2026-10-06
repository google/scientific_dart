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
import 'package:ndarray/ndarray.dart';
import 'package:notebook/src/kernel_helper.dart';

void main() {
  test('Plot renders valid SVG markup', () {
    final x = linspace<Float64>((0.0), (10.0), 10, dtype: DType.float64);
    final y = sin(x);
    final plot = Plot(x: x, y: y, title: 'Test Plot');
    final html = plot.toHtml();

    expect(html, contains('class="plot-container"'));
    expect(html, contains('Test Plot'));
    expect(html, contains('<polyline'));
  });

  test('Heatmap renders valid grid HTML', () {
    final mat = NDArray<Float64>.zeros([4, 4], DType.float64);
    final heatmap = Heatmap(mat, title: 'Matrix Heatmap');
    final html = heatmap.toHtml();

    expect(html, contains('class="heatmap-container"'));
    expect(html, contains('Matrix Heatmap'));
  });

  test('Animation renders frame controls', () {
    final f1 = NDArray<Float64>.zeros([10, 10, 3], DType.float64);
    final f2 = NDArray<Float64>.ones([10, 10, 3], DType.float64);
    final anim = Animation([f1, f2], fps: 10);
    final html = anim.toHtml();

    expect(html, contains('class="animation-container"'));
    expect(html, contains('<input type="range"'));
    expect(html, contains('1/2'));
  });

  test('Table renders 2D matrix HTML table', () {
    final mat = NDArray<Float64>.zeros([3, 2], DType.float64);
    final table = Table(mat, headers: ['A', 'B'], title: 'Data Table');
    final html = table.toHtml();

    expect(html, contains('class="table-container"'));
    expect(html, contains('Data Table'));
    expect(html, contains('>A</th>'));
    expect(html, contains('>B</th>'));
  });

  test('Histogram renders distribution SVG bar chart', () {
    final data = linspace<Float64>((0.0), (100.0), 50, dtype: DType.float64);
    final hist = Histogram(data, bins: 10, title: 'Dist Hist');
    final html = hist.toHtml();

    expect(html, contains('class="histogram-container"'));
    expect(html, contains('Dist Hist'));
    expect(html, contains('<rect'));
  });

  test('Image scales float pixel values slightly above 1.0 correctly', () {
    final arr = NDArray<Float64>.fromList(
      [1.05, 0.5, 0.0],
      [1, 1, 3],
      DType.float64,
    );
    final img = Image(arr);
    final url = img.toDataUrl();
    expect(url, startsWith('data:image/bmp;base64,'));
  });

  test('Markdown renders headers and inline formatting', () {
    final md = Md('# Title\n- Item 1\n- **Bold** item');
    final html = md.toHtml();

    expect(html, contains('class="notebook-markdown"'));
    expect(html, contains('Title</h1>'));
    expect(html, contains('<strong>Bold</strong>'));
  });

  test('Markdown renders fenced code blocks with language class', () {
    final md = Md('```dart\nfinal a = NDArray<Float64>.zeros([2, 2]);\n```');
    final html = md.toHtml();

    expect(html, contains('<pre'));
    expect(html, contains('class="language-dart"'));
    expect(html, contains('NDArray&lt;Float64&gt;.zeros([2, 2]);'));
  });

  test('display function captures multiple outputs', () {
    clearCapturedOutput();
    display(LaTeX(r'E=mc^2'));
    display(Md('# Section'));

    final cap = getCapturedOutput();
    expect(cap, contains('math-latex'));
    expect(cap, contains('notebook-markdown'));
    clearCapturedOutput();
  });

  test('Spectrogram renders audio signal time-frequency heatmap', () {
    final t = linspace<Float64>((0.0), (0.1), 1000, dtype: DType.float64);
    final signal = sin(t * (2 * 3.14159 * 440));
    final spec = Spectrogram(signal, title: 'Tone Spectrogram');
    final html = spec.toHtml();

    expect(html, contains('class="spectrogram-container"'));
    expect(html, contains('Tone Spectrogram'));
    expect(html, contains('Frequency Spectrogram'));
  });
}
