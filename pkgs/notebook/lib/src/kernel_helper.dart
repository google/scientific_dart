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

import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import 'package:symbolic_dart/symbolic_dart.dart'
    hide sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, abs;

/// Represents a single typed output item produced during cell execution.
final class CellOutputItem {
  /// The MIME type of the output content (e.g. `text/html`, `text/plain`).
  final String mimeType;

  /// The string payload matching [mimeType].
  final String data;

  /// Constructs a [CellOutputItem] with [mimeType] and [data].
  const CellOutputItem(this.mimeType, this.data);

  /// Converts this item into a JSON-encodable map.
  Map<String, String> toJson() => {'mimeType': mimeType, 'data': data};

  /// Constructs a [CellOutputItem] from a JSON map.
  factory CellOutputItem.fromJson(Map<String, dynamic> json) =>
      CellOutputItem(json['mimeType'] as String, json['data'] as String);
}

final List<CellOutputItem> capturedOutputs = [];
final StringBuffer capturedStdout = StringBuffer();
final StringBuffer capturedStderr = StringBuffer();

/// Displays a widget or object immediately in the cell output.
void display(dynamic object) {
  if (object is Displayable) {
    capturedOutputs.add(CellOutputItem(object.mimeType, object.toHtml()));
  } else if (object is Image) {
    capturedOutputs.add(CellOutputItem('text/html', object.toHtml()));
  } else if (object is Expr) {
    capturedOutputs.add(
      CellOutputItem('text/html', LaTeX(object.toLatex()).toHtml()),
    );
  } else if (object is SymbolicMatrix) {
    capturedOutputs.add(
      CellOutputItem('text/html', LaTeX(object.toLatex()).toHtml()),
    );
  } else if (object is FlintRationalPoly) {
    capturedOutputs.add(
      CellOutputItem('text/html', LaTeX(object.toLatex()).toHtml()),
    );
  } else {
    try {
      final html = (object as dynamic).toHtml();
      if (html is String) {
        capturedOutputs.add(CellOutputItem('text/html', html));
        return;
      }
    } catch (_) {}
    capturedOutputs.add(CellOutputItem('text/plain', object.toString()));
  }
}

void clearCapturedOutput() {
  capturedOutputs.clear();
  capturedStdout.clear();
  capturedStderr.clear();
}

String getCapturedOutput() {
  return capturedOutputs.map((e) => e.data).join('\n');
}

String getCapturedOutputsJson() {
  return jsonEncode(capturedOutputs.map((e) => e.toJson()).toList());
}

/// Abstract base class for objects that provide rich HTML representations for display in the notebook interface.
abstract class Displayable {
  /// The MIME type representation of this displayable object (defaults to 'text/html').
  String get mimeType => 'text/html';

  /// Renders the object as an HTML string or Data URL.
  String toHtml();
}

/// Formats evaluation results for display in the notebook.
///
/// If [x] implements [Displayable], returns its [Displayable.toHtml] representation.
/// If [x] is an [Image], returns a Base64-encoded BMP data URL string.
/// If [x] is a symbolic AST node ([Expr], [SymbolicMatrix], [FlintRationalPoly]),
/// automatically formats it into a rendered LaTeX KaTeX widget.
/// Otherwise, falls back to [Object.toString].
String prettyFormat(dynamic x) {
  if (x is Displayable) {
    return x.toHtml();
  }
  if (x is Expr) {
    return LaTeX(x.toLatex()).toHtml();
  }
  if (x is SymbolicMatrix) {
    return LaTeX(x.toLatex()).toHtml();
  }
  if (x is FlintRationalPoly) {
    return LaTeX(x.toLatex()).toHtml();
  }
  try {
    final html = (x as dynamic).toHtml();
    if (html is String) return html;
  } catch (_) {}
  return x.toString();
}

/// Represents a raw HTML widget for display in the notebook interface.
final class Html extends Displayable {
  /// The raw HTML string.
  final String html;

  /// Constructs an [Html] widget wrapper around [html].
  Html(this.html);

  @override
  String toHtml() => html;
}

/// Represents a LaTeX mathematical equation for display in the notebook interface.
final class LaTeX extends Displayable {
  /// The LaTeX equation string.
  final String latex;

  /// Constructs a [LaTeX] widget wrapper around [latex].
  LaTeX(this.latex);

  @override
  String toHtml() =>
      '<div class="math-latex" style="font-size: 1.2em; padding: 10px 16px; color: #cdd6f4; background: #181825; border-radius: 6px; border-left: 4px solid #89b4fa; margin: 4px 0; display: inline-block;">\\($latex\\)</div>';
}

/// Alias for [LaTeX].
typedef Latex = LaTeX;

/// Represents formatted Markdown text for display in the notebook interface.
final class Markdown extends Displayable {
  /// The raw markdown string content.
  final String markdown;

  /// Constructs a [Markdown] widget wrapper around [markdown].
  Markdown(this.markdown);

  @override
  String toHtml() {
    final encoded = base64Encode(utf8.encode(markdown));
    return '<div class="notebook-markdown" data-markdown="$encoded" style="padding: 8px 12px; color: #cdd6f4; font-family: system-ui, sans-serif; line-height: 1.6;">${_simpleMarkdownToHtml(markdown)}</div>';
  }

  static String _simpleMarkdownToHtml(String text) {
    final lines = text.split('\n');
    final sb = StringBuffer();
    for (final line in lines) {
      final l = line.trimRight();
      if (l.startsWith('# ')) {
        sb.writeln(
          '<h1 style="font-size: 1.6em; margin: 12px 0 6px 0; color: #89b4fa;">${_formatInlineMd(l.substring(2))}</h1>',
        );
      } else if (l.startsWith('## ')) {
        sb.writeln(
          '<h2 style="font-size: 1.3em; margin: 10px 0 4px 0; color: #b4befe;">${_formatInlineMd(l.substring(3))}</h2>',
        );
      } else if (l.startsWith('### ')) {
        sb.writeln(
          '<h3 style="font-size: 1.1em; margin: 8px 0 4px 0; color: #cdd6f4;">${_formatInlineMd(l.substring(4))}</h3>',
        );
      } else if (l.startsWith('- ') || l.startsWith('* ')) {
        sb.writeln(
          '<li style="margin-left: 16px; color: #cdd6f4;">${_formatInlineMd(l.substring(2))}</li>',
        );
      } else if (l.isNotEmpty) {
        sb.writeln(
          '<p style="margin: 4px 0; color: #cdd6f4;">${_formatInlineMd(l)}</p>',
        );
      }
    }
    return sb.toString();
  }

  static String _formatInlineMd(String s) {
    return s
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAllMapped(
          RegExp(r'\*\*(.*?)\*\*'),
          (m) => '<strong>${m[1]}</strong>',
        )
        .replaceAllMapped(RegExp(r'\*(.*?)\*'), (m) => '<em>${m[1]}</em>')
        .replaceAllMapped(
          RegExp(r'`(.*?)`'),
          (m) =>
              '<code style="background: #313244; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">${m[1]}</code>',
        );
  }
}

/// Alias for [Markdown].
typedef Md = Markdown;

/// Represents a playable sound clip wrapping an [NDArray].
///
/// Converts audio waveform samples into a 16-bit PCM WAV audio clip.
final class Audio extends Displayable {
  /// The underlying audio waveform data.
  final NDArray data;

  /// The audio sampling rate in Hz (defaults to 44100 Hz).
  final int sampleRate;

  /// Constructs a playable [Audio] wrapper around sound samples in an [NDArray].
  ///
  /// **Preconditions:**
  /// - [data] must not be disposed.
  /// - [data] must be 1D `[samples]` or 2D `[samples, channels]`.
  ///
  /// **Throws:**
  /// - [ArgumentError] if [data] is disposed or does not match valid audio dimensions.
  Audio(this.data, {this.sampleRate = 44100}) {
    if (data.isDisposed) {
      throw ArgumentError('Cannot construct Audio from a disposed NDArray.');
    }
    if (data.shape.length != 1 && data.shape.length != 2) {
      throw ArgumentError(
        'Audio NDArray must be 1D [samples] or 2D [samples, channels]. Got shape ${data.shape}',
      );
    }
  }

  /// Converts the audio waveform samples into a Base64-encoded WAV Data URL string.
  String toDataUrl() {
    final flatList = data.toList();
    final numSamples = flatList.length;
    final dataSize = numSamples * 2;
    final fileSize = 44 + dataSize;

    final bytes = Uint8List(fileSize);
    final bd = ByteData.sublistView(bytes);

    bytes.setRange(0, 4, ascii.encode('RIFF'));
    bd.setUint32(4, fileSize - 8, Endian.little);
    bytes.setRange(8, 12, ascii.encode('WAVE'));

    bytes.setRange(12, 16, ascii.encode('fmt '));
    bd.setUint32(16, 16, Endian.little);
    bd.setUint16(20, 1, Endian.little); // 1 = PCM
    bd.setUint16(22, 1, Endian.little); // 1 = Mono
    bd.setUint32(24, sampleRate, Endian.little);
    bd.setUint32(28, sampleRate * 2, Endian.little);
    bd.setUint16(32, 2, Endian.little);
    bd.setUint16(34, 16, Endian.little);

    bytes.setRange(36, 40, ascii.encode('data'));
    bd.setUint32(40, dataSize, Endian.little);

    int offset = 44;
    for (var i = 0; i < numSamples; i++) {
      final double val = (flatList[i] as num).toDouble();
      final int pcm16 = (val.clamp(-1.0, 1.0) * 32767.0).toInt();
      bd.setInt16(offset, pcm16, Endian.little);
      offset += 2;
    }

    return 'data:audio/wav;base64,${base64.encode(bytes)}';
  }

  @override
  String toHtml() {
    return '<audio controls style="width: 100%; max-width: 500px;" src="${toDataUrl()}"></audio>';
  }
}

/// Represents a renderable graphic image wrapping an [NDArray].
///
/// Supports 2D arrays `[height, width]` for grayscale imagery, and 3D arrays
/// `[height, width, channels]` where `channels` is 1 (grayscale), 3 (RGB), or 4 (RGBA).
final class Image extends Displayable {
  /// The underlying multi-dimensional array containing image pixel values.
  final NDArray data;

  /// Constructs a renderable [Image] wrapper around the provided multi-dimensional [data] array.
  ///
  /// **Preconditions:**
  /// - [data] must not be disposed.
  /// - [data] must be 2D `[height, width]` or 3D `[height, width, channels]`.
  ///
  /// **Throws:**
  /// - [ArgumentError] if [data] is disposed or does not match valid image dimensions.
  ///
  /// **Example:**
  /// ```dart
  /// final imgArray = NDArray.zeros([100, 100, 3], DType.float64);
  /// final img = Image(imgArray);
  /// ```
  Image(this.data) {
    if (data.isDisposed) {
      throw ArgumentError('Cannot construct an Image from a disposed NDArray.');
    }
    if (data.shape.length != 2 && data.shape.length != 3) {
      throw ArgumentError(
        'Image NDArray must be 2D [height, width] or 3D [height, width, channels]. Got shape ${data.shape}',
      );
    }
    if (data.shape.length == 3) {
      final channels = data.shape[2];
      if (channels != 1 && channels != 3 && channels != 4) {
        throw ArgumentError(
          '3D Image NDArray must have 1, 3, or 4 channels. Got $channels channels.',
        );
      }
    }
  }

  @override
  String toHtml() {
    return '<img src="${toDataUrl()}" style="max-width: 100%; max-height: 500px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); display: block;" />';
  }

  /// Converts the image data into a Base64-encoded BMP Data URL string.
  ///
  /// Automatically handles floating-point scaling `[0.0, 1.0]` as well as integer pixel intensity ranges `[0, 255]`.
  ///
  /// **Performance Considerations:**
  /// - Performs $O(N)$ memory flattening and Base64 string encoding.
  String toDataUrl() {
    final shape = data.shape;
    final height = shape[0];
    final width = shape[1];
    final channels = shape.length == 3 ? shape[2] : 1;

    final flatList = data.toList();
    final pixelCount = width * height;
    final rgbaBytes = Uint8List(pixelCount * 4);

    var maxVal = 0.0;
    var isFloat = false;
    for (var i = 0; i < flatList.length; i++) {
      final v = (flatList[i] as num).toDouble();
      if (v > maxVal) maxVal = v;
      if (flatList[i] is double) isFloat = true;
    }

    final bool scaleFromFloat = isFloat && maxVal <= 2.0;

    int srcIdx = 0;
    int dstIdx = 0;

    for (var i = 0; i < pixelCount; i++) {
      double r = 0.0, g = 0.0, b = 0.0, a = 255.0;

      if (channels == 1) {
        final v = (flatList[srcIdx] as num).toDouble();
        r = g = b = v;
        srcIdx += 1;
      } else if (channels == 3) {
        r = (flatList[srcIdx] as num).toDouble();
        g = (flatList[srcIdx + 1] as num).toDouble();
        b = (flatList[srcIdx + 2] as num).toDouble();
        srcIdx += 3;
      } else if (channels == 4) {
        r = (flatList[srcIdx] as num).toDouble();
        g = (flatList[srcIdx + 1] as num).toDouble();
        b = (flatList[srcIdx + 2] as num).toDouble();
        a = (flatList[srcIdx + 3] as num).toDouble();
        srcIdx += 4;
      }

      if (scaleFromFloat) {
        r *= 255.0;
        g *= 255.0;
        b *= 255.0;
        if (channels == 4) a *= 255.0;
      }

      rgbaBytes[dstIdx] = r.clamp(0.0, 255.0).toInt();
      rgbaBytes[dstIdx + 1] = g.clamp(0.0, 255.0).toInt();
      rgbaBytes[dstIdx + 2] = b.clamp(0.0, 255.0).toInt();
      rgbaBytes[dstIdx + 3] = a.clamp(0.0, 255.0).toInt();
      dstIdx += 4;
    }

    final pixelDataSize = pixelCount * 4;
    final fileSize = 54 + pixelDataSize;
    final bmpBytes = Uint8List(fileSize);
    final bd = ByteData.sublistView(bmpBytes);

    bmpBytes[0] = 0x42; // 'B'
    bmpBytes[1] = 0x4D; // 'M'
    bd.setUint32(2, fileSize, Endian.little);
    bd.setUint32(6, 0, Endian.little);
    bd.setUint32(10, 54, Endian.little);

    bd.setUint32(14, 40, Endian.little);
    bd.setInt32(18, width, Endian.little);
    bd.setInt32(22, -height, Endian.little); // Top-down
    bd.setUint16(26, 1, Endian.little);
    bd.setUint16(28, 32, Endian.little); // 32 bpp
    bd.setUint32(30, 0, Endian.little); // BI_RGB
    bd.setUint32(34, pixelDataSize, Endian.little);
    bd.setInt32(38, 2835, Endian.little);
    bd.setInt32(42, 2835, Endian.little);
    bd.setUint32(46, 0, Endian.little);
    bd.setUint32(50, 0, Endian.little);

    int pSrc = 0;
    int pDst = 54;
    for (int i = 0; i < pixelCount; i++) {
      bmpBytes[pDst] = rgbaBytes[pSrc + 2]; // Blue
      bmpBytes[pDst + 1] = rgbaBytes[pSrc + 1]; // Green
      bmpBytes[pDst + 2] = rgbaBytes[pSrc]; // Red
      bmpBytes[pDst + 3] = rgbaBytes[pSrc + 3]; // Alpha
      pSrc += 4;
      pDst += 4;
    }

    return 'data:image/bmp;base64,${base64Encode(bmpBytes)}';
  }

  @override
  String toString() => toDataUrl();
}

/// Represents a 2D line plot widget wrapping [NDArray] coordinate data.
final class Plot extends Displayable {
  /// Optional x-axis coordinate array.
  final NDArray? x;

  /// Y-axis coordinate array.
  final NDArray y;

  /// Optional chart title.
  final String title;

  /// Optional stroke color hex string (e.g. `#89b4fa`).
  final String color;

  /// Constructs a [Plot] widget wrapping [y] (and optional [x]) coordinate arrays.
  Plot({this.x, required this.y, this.title = '', this.color = '#89b4fa'}) {
    if (y.isDisposed) {
      throw ArgumentError('Cannot construct Plot from a disposed NDArray.');
    }
  }

  @override
  String toHtml() {
    final yValues = y.toList().map((e) => (e as num).toDouble()).toList();
    final n = yValues.length;
    final xValues = x != null
        ? x!.toList().map((e) => (e as num).toDouble()).toList()
        : List<double>.generate(n, (i) => i.toDouble());

    if (n == 0) return '<div>(Empty Plot)</div>';

    final double minX = xValues.reduce(math.min);
    double maxX = xValues.reduce(math.max);
    double minY = yValues.reduce(math.min);
    double maxY = yValues.reduce(math.max);

    if (minX == maxX) maxX += 1.0;
    if (minY == maxY) {
      minY -= 1.0;
      maxY += 1.0;
    }

    const width = 500.0;
    const height = 280.0;
    const padding = 40.0;

    final points = <String>[];
    for (var i = 0; i < n; i++) {
      final px =
          padding +
          ((xValues[i] - minX) / (maxX - minX)) * (width - 2 * padding);
      final py =
          (height - padding) -
          ((yValues[i] - minY) / (maxY - minY)) * (height - 2 * padding);
      points.add('${px.toStringAsFixed(1)},${py.toStringAsFixed(1)}');
    }

    final pathData = points.join(' ');

    return '''
<div class="plot-container" style="background: #181825; border: 1px solid #45475a; border-radius: 8px; padding: 16px; display: inline-block; font-family: system-ui, sans-serif;">
  ${title.isNotEmpty ? '<div style="font-weight: bold; margin-bottom: 8px; color: #cdd6f4; font-size: 1.1em;">$title</div>' : ''}
  <svg width="$width" height="$height" viewBox="0 0 $width $height" style="overflow: visible;">
    <line x1="$padding" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#45475a" stroke-width="1"/>
    <line x1="$padding" y1="$padding" x2="$padding" y2="${height - padding}" stroke="#45475a" stroke-width="1"/>
    <polyline fill="none" stroke="$color" stroke-width="2.5" points="$pathData"/>
    <text x="$padding" y="${height - 10}" fill="#a6adc8" font-size="12">${minX.toStringAsPrecision(3)}</text>
    <text x="${width - padding}" y="${height - 10}" fill="#a6adc8" font-size="12" text-anchor="end">${maxX.toStringAsPrecision(3)}</text>
    <text x="${padding - 10}" y="${height - padding}" fill="#a6adc8" font-size="12" text-anchor="end">${minY.toStringAsPrecision(3)}</text>
    <text x="${padding - 10}" y="${padding + 10}" fill="#a6adc8" font-size="12" text-anchor="end">${maxY.toStringAsPrecision(3)}</text>
  </svg>
</div>
''';
  }
}

/// Represents a 2D matrix heatmap visualizer widget wrapping an [NDArray].
final class Heatmap extends Displayable {
  /// The 2D input array.
  final NDArray data;

  /// Optional chart title.
  final String title;

  /// Constructs a [Heatmap] widget wrapping a 2D [data] matrix array.
  Heatmap(this.data, {this.title = ''}) {
    if (data.isDisposed) {
      throw ArgumentError('Cannot construct Heatmap from a disposed NDArray.');
    }
    if (data.shape.length != 2) {
      throw ArgumentError(
        'Heatmap NDArray must be 2D [rows, cols]. Got shape ${data.shape}',
      );
    }
  }

  @override
  String toHtml() {
    final rows = data.shape[0];
    final cols = data.shape[1];
    final flatList = data.toList().map((e) => (e as num).toDouble()).toList();

    if (flatList.isEmpty) return '<div>(Empty Heatmap)</div>';

    final minV = flatList.reduce(math.min);
    final maxV = flatList.reduce(math.max);
    final range = maxV == minV ? 1.0 : (maxV - minV);

    final cellW = math.min(36.0, 400.0 / cols);
    final cellH = math.min(36.0, 400.0 / rows);

    final sb = StringBuffer();
    sb.writeln(
      '<div class="heatmap-container" style="background: #181825; border: 1px solid #45475a; border-radius: 8px; padding: 16px; display: inline-block; font-family: system-ui, sans-serif;">',
    );
    if (title.isNotEmpty) {
      sb.writeln(
        '<div style="font-weight: bold; margin-bottom: 12px; color: #cdd6f4; font-size: 1.1em;">$title</div>',
      );
    }
    sb.writeln(
      '<div style="display: grid; grid-template-columns: repeat($cols, ${cellW}px); gap: 1px; background: #313244; padding: 1px; border-radius: 4px;">',
    );

    for (var i = 0; i < flatList.length; i++) {
      final val = flatList[i];
      final norm = ((val - minV) / range).clamp(0.0, 1.0);
      final r = (255 * (0.2 + 0.8 * norm)).toInt();
      final g = (255 * (0.1 + 0.9 * (1 - (norm - 0.5).abs() * 2))).toInt();
      final b = (255 * (0.5 + 0.5 * (1 - norm))).toInt();
      final color = 'rgb($r, $g, $b)';

      final row = i ~/ cols;
      final col = i % cols;
      final tooltip = '[$row, $col] = ${val.toStringAsFixed(4)}';
      sb.writeln(
        '<div title="$tooltip" style="width: ${cellW}px; height: ${cellH}px; background: $color; border-radius: 1px;"></div>',
      );
    }

    sb.writeln('</div></div>');
    return sb.toString();
  }
}

/// Represents an interactive multi-frame animation player widget wrapping a sequence of [NDArray] frames.
final class Animation extends Displayable {
  /// The list of image frame arrays.
  final List<NDArray> frames;

  /// Frames per second playback rate.
  final int fps;

  /// Constructs an [Animation] player widget wrapping a sequence of [frames].
  Animation(this.frames, {this.fps = 20}) {
    if (frames.isEmpty) {
      throw ArgumentError('Animation frames list must not be empty.');
    }
  }

  @override
  String toHtml() {
    final id = 'anim_${DateTime.now().microsecondsSinceEpoch}';
    final imageUrls = frames.map((f) => Image(f).toDataUrl()).toList();
    final urlsJson = jsonEncode(imageUrls);
    final intervalMs = (1000 / math.max(1, fps)).round();

    return '''
<div id="$id" class="animation-container" style="background: #181825; border: 1px solid #45475a; border-radius: 8px; padding: 16px; display: inline-block; font-family: system-ui, sans-serif;">
  <img id="${id}_img" src="${imageUrls.first}" style="max-width: 100%; max-height: 350px; border-radius: 6px; display: block; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);" />
  <div style="display: flex; align-items: center; gap: 12px;">
    <button id="${id}_btn" onclick="window['${id}_toggle']()" style="background: #89b4fa; color: #11111b; border: none; padding: 6px 16px; border-radius: 4px; font-weight: bold; cursor: pointer;">Pause</button>
    <input type="range" id="${id}_slider" min="0" max="${frames.length - 1}" value="0" style="flex: 1;" oninput="window['${id}_seek'](this.value)"/>
    <span id="${id}_txt" style="font-family: monospace; color: #a6adc8; min-width: 50px; text-anchor: end;">1/${frames.length}</span>
  </div>
  <script>
    (function() {
      const urls = $urlsJson;
      const total = urls.length;
      let idx = 0;
      let playing = true;
      let timer = setInterval(tick, $intervalMs);

      function tick() {
        if (!playing) return;
        idx = (idx + 1) % total;
        update();
      }

      function update() {
        const img = document.getElementById('${id}_img');
        const slider = document.getElementById('${id}_slider');
        const txt = document.getElementById('${id}_txt');
        if (img) img.src = urls[idx];
        if (slider) slider.value = idx;
        if (txt) txt.innerText = (idx + 1) + '/' + total;
      }

      window['${id}_toggle'] = function() {
        playing = !playing;
        const btn = document.getElementById('${id}_btn');
        if (btn) btn.innerText = playing ? 'Pause' : 'Play';
      };

      window['${id}_seek'] = function(val) {
        idx = parseInt(val, 10);
        update();
      };
    })();
  </script>
</div>
''';
  }
}

/// Represents an interactive data table widget wrapping a 2D [NDArray].
final class Table extends Displayable {
  /// The 2D input array.
  final NDArray data;

  /// Optional column headers.
  final List<String>? headers;

  /// Optional title.
  final String title;

  /// Constructs a [Table] widget wrapping a 2D [data] array.
  Table(this.data, {this.headers, this.title = ''}) {
    if (data.isDisposed) {
      throw ArgumentError('Cannot construct Table from a disposed NDArray.');
    }
    if (data.shape.length != 2) {
      throw ArgumentError(
        'Table NDArray must be 2D [rows, cols]. Got shape ${data.shape}',
      );
    }
  }

  @override
  String toHtml() {
    final rows = data.shape[0];
    final cols = data.shape[1];
    final flatList = data.toList();

    final sb = StringBuffer();
    sb.writeln(
      '<div class="table-container" style="background: #181825; border: 1px solid #45475a; border-radius: 8px; padding: 16px; display: inline-block; font-family: system-ui, sans-serif; max-height: 400px; overflow: auto;">',
    );
    if (title.isNotEmpty) {
      sb.writeln(
        '<div style="font-weight: bold; margin-bottom: 12px; color: #cdd6f4; font-size: 1.1em;">$title</div>',
      );
    }
    sb.writeln(
      '<table style="border-collapse: collapse; width: 100%; color: #cdd6f4; font-size: 13px;">',
    );

    sb.writeln(
      '<tr style="background: #313244; border-bottom: 2px solid #45475a;">',
    );
    sb.writeln(
      '<th style="padding: 8px 12px; text-align: right; color: #a6adc8;">#</th>',
    );
    for (var c = 0; c < cols; c++) {
      final name = (headers != null && c < headers!.length)
          ? headers![c]
          : 'Col $c';
      sb.writeln(
        '<th style="padding: 8px 12px; text-align: right; font-weight: bold;">$name</th>',
      );
    }
    sb.writeln('</tr>');

    for (var r = 0; r < rows; r++) {
      final bg = r.isEven ? '#181825' : '#1e1e2e';
      sb.writeln(
        '<tr style="background: $bg; border-bottom: 1px solid #313244;">',
      );
      sb.writeln(
        '<td style="padding: 6px 12px; text-align: right; color: #6c7086; font-family: monospace;">$r</td>',
      );
      for (var c = 0; c < cols; c++) {
        final val = flatList[r * cols + c];
        final valStr = val is double ? val.toStringAsFixed(4) : '$val';
        sb.writeln(
          '<td style="padding: 6px 12px; text-align: right; font-family: monospace;">$valStr</td>',
        );
      }
      sb.writeln('</tr>');
    }

    sb.writeln('</table></div>');
    return sb.toString();
  }
}

/// Represents a statistical distribution histogram bar chart widget wrapping an [NDArray].
final class Histogram extends Displayable {
  /// The input data array.
  final NDArray data;

  /// Total bin count.
  final int bins;

  /// Optional chart title.
  final String title;

  /// Optional bar fill color.
  final String color;

  /// Constructs a [Histogram] bar chart widget wrapping [data].
  Histogram(
    this.data, {
    this.bins = 20,
    this.title = '',
    this.color = '#a6e3a1',
  }) {
    if (data.isDisposed) {
      throw ArgumentError(
        'Cannot construct Histogram from a disposed NDArray.',
      );
    }
  }

  @override
  String toHtml() {
    final flatList = data.toList().map((e) => (e as num).toDouble()).toList();
    if (flatList.isEmpty) return '<div>(Empty Histogram)</div>';

    final minV = flatList.reduce(math.min);
    final maxV = flatList.reduce(math.max);
    final range = maxV == minV ? 1.0 : (maxV - minV);
    final binWidth = range / bins;

    final counts = List<int>.filled(bins, 0);
    for (final val in flatList) {
      int idx = ((val - minV) / binWidth).floor();
      if (idx >= bins) idx = bins - 1;
      counts[idx]++;
    }

    final maxCount = counts.reduce(math.max);
    const svgWidth = 500.0;
    const svgHeight = 220.0;
    const padding = 40.0;
    final barW = (svgWidth - 2 * padding) / bins;

    final bars = <String>[];
    for (var i = 0; i < bins; i++) {
      final h = maxCount == 0
          ? 0.0
          : (counts[i] / maxCount) * (svgHeight - 2 * padding);
      final x = padding + i * barW;
      final y = (svgHeight - padding) - h;
      bars.add(
        '<rect x="${x.toStringAsFixed(1)}" y="${y.toStringAsFixed(1)}" width="${(barW - 1).toStringAsFixed(1)}" height="${h.toStringAsFixed(1)}" fill="$color" rx="1"><title>Bin ${i + 1}: ${counts[i]}</title></rect>',
      );
    }

    return '''
<div class="histogram-container" style="background: #181825; border: 1px solid #45475a; border-radius: 8px; padding: 16px; display: inline-block; font-family: system-ui, sans-serif;">
  ${title.isNotEmpty ? '<div style="font-weight: bold; margin-bottom: 8px; color: #cdd6f4; font-size: 1.1em;">$title</div>' : ''}
  <svg width="$svgWidth" height="$svgHeight" viewBox="0 0 $svgWidth $svgHeight" style="overflow: visible;">
    <line x1="$padding" y1="${svgHeight - padding}" x2="${svgWidth - padding}" y2="${svgHeight - padding}" stroke="#45475a" stroke-width="1"/>
    ${bars.join('\n')}
    <text x="$padding" y="${svgHeight - 10}" fill="#a6adc8" font-size="12">${minV.toStringAsPrecision(3)}</text>
    <text x="${svgWidth - padding}" y="${svgHeight - 10}" fill="#a6adc8" font-size="12" text-anchor="end">${maxV.toStringAsPrecision(3)}</text>
    <text x="${padding - 10}" y="${padding + 10}" fill="#a6adc8" font-size="12" text-anchor="end">$maxCount</text>
  </svg>
</div>
''';
  }
}

/// Represents a time-frequency spectrogram visualizer widget wrapping an [NDArray].
final class Spectrogram extends Displayable {
  /// 2D STFT magnitude matrix `[frequencies, timeFrames]` or 1D audio waveform signal.
  final NDArray data;

  /// Audio sampling rate in Hz (defaults to 44100 Hz).
  final int sampleRate;

  /// Optional chart title.
  final String title;

  /// Constructs a [Spectrogram] widget wrapping [data].
  Spectrogram(this.data, {this.sampleRate = 44100, this.title = ''}) {
    if (data.isDisposed) {
      throw ArgumentError(
        'Cannot construct Spectrogram from a disposed NDArray.',
      );
    }
  }

  @override
  String toHtml() {
    List<List<double>> spec;
    int freqs;
    int frames;

    if (data.shape.length == 1) {
      final signal = data.toList().map((e) => (e as num).toDouble()).toList();
      spec = _computeStft(signal, 256, 128);
      freqs = spec.length;
      frames = spec.isEmpty ? 0 : spec[0].length;
    } else if (data.shape.length == 2) {
      freqs = data.shape[0];
      frames = data.shape[1];
      final flat = data.toList().map((e) => (e as num).toDouble()).toList();
      spec = List.generate(
        freqs,
        (r) => List.generate(frames, (c) => flat[r * frames + c]),
      );
    } else {
      throw ArgumentError(
        'Spectrogram NDArray must be 1D audio signal or 2D matrix [freqs, time].',
      );
    }

    if (freqs == 0 || frames == 0) return '<div>(Empty Spectrogram)</div>';

    var minDb = double.infinity;
    var maxDb = -double.infinity;
    for (var r = 0; r < freqs; r++) {
      for (var c = 0; c < frames; c++) {
        final val = spec[r][c];
        if (val < minDb) minDb = val;
        if (val > maxDb) maxDb = val;
      }
    }
    final nyquist = sampleRate / 2.0;
    final totalDuration = (data.shape.length == 1)
        ? data.shape[0] / sampleRate
        : 1.0;
    final bmpUrl = _generateSpectrogramBmpUrl(spec, minDb, maxDb);

    final sb = StringBuffer();
    sb.writeln(
      '<div class="spectrogram-container" style="background: #181825; border: 1px solid #45475a; border-radius: 8px; padding: 16px; display: inline-block; font-family: system-ui, sans-serif;">',
    );
    if (title.isNotEmpty) {
      sb.writeln(
        '<div style="font-weight: bold; margin-bottom: 12px; color: #cdd6f4; font-size: 1.1em;">$title</div>',
      );
    }

    sb.writeln(
      '<img src="$bmpUrl" style="width: 480px; height: 200px; image-rendering: pixelated; border-radius: 4px; border: 1px solid #313244; display: block;" />',
    );
    sb.writeln(
      '<div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: #a6adc8; width: 480px;">',
    );
    sb.writeln(
      '<span>0.0s</span><span>Frequency Spectrogram (${nyquist.round()}Hz)</span><span>${totalDuration.toStringAsFixed(2)}s</span>',
    );
    sb.writeln('</div></div>');

    return sb.toString();
  }

  static String _generateSpectrogramBmpUrl(
    List<List<double>> spec,
    double minDb,
    double maxDb,
  ) {
    final freqs = spec.length;
    final frames = spec.isEmpty ? 0 : spec[0].length;
    if (freqs == 0 || frames == 0) return '';

    final range = maxDb == minDb ? 1.0 : (maxDb - minDb);
    final pixelCount = freqs * frames;
    const bmpHeaderSize = 54;
    final pixelDataSize = pixelCount * 4;
    final totalFileSize = bmpHeaderSize + pixelDataSize;

    final bmpBytes = Uint8List(totalFileSize);
    final bd = ByteData.sublistView(bmpBytes);

    bmpBytes[0] = 0x42; // 'B'
    bmpBytes[1] = 0x4D; // 'M'
    bd.setUint32(2, totalFileSize, Endian.little);
    bd.setUint32(10, bmpHeaderSize, Endian.little);
    bd.setUint32(14, 40, Endian.little);
    bd.setInt32(18, frames, Endian.little);
    bd.setInt32(22, freqs, Endian.little);
    bd.setUint16(26, 1, Endian.little);
    bd.setUint16(28, 32, Endian.little);
    bd.setUint32(34, pixelDataSize, Endian.little);

    int pDst = 54;
    for (var r = 0; r < freqs; r++) {
      for (var c = 0; c < frames; c++) {
        final val = spec[r][c];
        final norm = ((val - minDb) / range).clamp(0.0, 1.0);

        final cr = (255 * math.pow(norm, 0.8)).toInt().clamp(0, 255);
        final cg = (255 * math.pow(norm, 2.0)).toInt().clamp(0, 255);
        final cb = (255 * (1.0 - norm)).toInt().clamp(0, 255);

        bmpBytes[pDst] = cb;
        bmpBytes[pDst + 1] = cg;
        bmpBytes[pDst + 2] = cr;
        bmpBytes[pDst + 3] = 255;
        pDst += 4;
      }
    }

    return 'data:image/bmp;base64,${base64Encode(bmpBytes)}';
  }

  static List<List<double>> _computeStft(
    List<double> signal,
    int fftSize,
    int hopSize,
  ) {
    final numFrames = ((signal.length - fftSize) / hopSize).floor();
    if (numFrames <= 0) return [];

    final numFreqs = fftSize ~/ 2 + 1;
    final spec = List.generate(
      numFreqs,
      (_) => List<double>.filled(numFrames, 0.0),
    );
    final window = List<double>.generate(
      fftSize,
      (i) => 0.5 * (1.0 - math.cos(2 * math.pi * i / (fftSize - 1))),
    );

    for (var f = 0; f < numFrames; f++) {
      final offset = f * hopSize;
      for (var k = 0; k < numFreqs; k++) {
        double r = 0.0, img = 0.0;
        final angleStep = -2 * math.pi * k / fftSize;
        for (var n = 0; n < fftSize; n++) {
          final s = signal[offset + n] * window[n];
          final angle = angleStep * n;
          r += s * math.cos(angle);
          img += s * math.sin(angle);
        }
        final mag = math.sqrt(r * r + img * img);
        final db = 20.0 * math.log(mag + 1e-6) / math.ln10;
        spec[k][f] = db;
      }
    }
    return spec;
  }
}

/// Evaluates and plots a 1D symbolic expression [f] with respect to variable [varName]
/// over the range [[from], [to]] using [points] samples.
Plot plotSymbolic(
  Expr f,
  Expr varName, {
  num from = -10,
  num to = 10,
  int points = 200,
  String? title,
  String color = '#89b4fa',
}) {
  if (points <= 1) {
    throw ArgumentError('points must be greater than 1');
  }
  final lambda = f.lambdify([varName]);
  final xArr = linspace<Float64>(
    from.toDouble(),
    to.toDouble(),
    points,
    dtype: DType.float64,
  );
  final yArr = lambda.callArray([xArr]);
  return Plot(
    x: xArr,
    y: yArr,
    title: title ?? 'f($varName) = $f',
    color: color,
  );
}

/// Evaluates and plots a 2D symbolic expression [f] with respect to variables [xVar] and [yVar]
/// as a 2D heatmap over the ranges [[xFrom], [xTo]] and [[yFrom], [yTo]].
Heatmap plotSymbolic2D(
  Expr f,
  Expr xVar,
  Expr yVar, {
  num xFrom = -5,
  num xTo = 5,
  num yFrom = -5,
  num yTo = 5,
  int points = 50,
  String? title,
}) {
  if (points <= 1) {
    throw ArgumentError('points must be greater than 1');
  }
  final lambda = f.lambdify([xVar, yVar]);
  final grids = ogrid([
    GridRange(xFrom.toDouble(), xTo.toDouble(), numPoints: points),
    GridRange(yFrom.toDouble(), yTo.toDouble(), numPoints: points),
  ]);
  final zArr = lambda.callArray([grids[0], grids[1]]);
  return Heatmap(zArr, title: title ?? 'f($xVar, $yVar) = $f');
}

/// Represents an interactive client-side WebGPU compute widget for browser notebooks.
final class WebGpuShaderWidget extends Displayable {
  /// Custom title displayed at the top of the widget.
  final String title;

  /// Raw WGSL compute shader source code.
  final String wgsl;

  /// Compute shader entry point function name.
  final String entryPoint;

  /// Workgroup invocation layout.
  final List<int> workgroups;

  /// Uniform scalar values.
  final List<int> uniforms;

  /// Arbitrary configuration metadata.
  final Map<String, dynamic> metadata;

  WebGpuShaderWidget({
    this.title = 'WebGPU Compute Pipeline',
    required this.wgsl,
    this.entryPoint = 'main',
    this.workgroups = const [1, 1, 1],
    this.uniforms = const [],
    this.metadata = const {},
  });

  @override
  String toHtml() {
    final uid = 'webgpu_widget_${DateTime.now().microsecondsSinceEpoch}';
    final payload = jsonEncode({
      'wgsl': wgsl,
      'entryPoint': entryPoint,
      'workgroups': workgroups,
      'uniforms': uniforms,
      'metadata': metadata,
    });

    return '''
<div id="$uid" style="background:#181825; border:1px solid #45475a; border-radius:8px; padding:16px; margin:8px 0; font-family:system-ui, sans-serif; color:#cdd6f4;">
  <div style="font-weight:bold; color:#89b4fa; margin-bottom:8px;">⚡ $title (Browser WebGPU)</div>
  <div id="\${uid}_status" style="color:#a6adc8; font-size:13px;">Initializing client WebGPU...</div>
  <pre id="\${uid}_output" style="display:none; background:#11111b; color:#a6e3a1; padding:10px; border-radius:4px; margin-top:8px; font-size:12px; font-family:monospace;"></pre>
  <script>
    (async () => {
      const statusEl = document.getElementById('\${uid}_status');
      const config = $payload;
      if (!navigator.gpu) {
        statusEl.innerHTML = '<span style="color:#f38ba8;">WebGPU is not supported in this browser.</span>';
        return;
      }
      try {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter.requestDevice();
        const shader = device.createShaderModule({ code: config.wgsl });
        const pipeline = device.createComputePipeline({
          layout: 'auto',
          compute: { module: shader, entryPoint: config.entryPoint }
        });
        const arch = adapter.info?.architecture || adapter.info?.vendor || 'Physical GPU';
        statusEl.innerHTML = '<span style="color:#a6e3a1;">Shader module active on client GPU (' + arch + ').</span>';
      } catch (e) {
        statusEl.innerHTML = '<span style="color:#f38ba8;">Shader Compilation Error: ' + e.message + '</span>';
      }
    })();
  </script>
</div>
''';
  }
}

dynamic evalInNotebookZone(dynamic Function() body) {
  return IOOverrides.runZoned(
    () {
      return runZoned(
        body,
        zoneSpecification: ZoneSpecification(
          print: (self, parent, zone, line) {
            capturedOutputs.add(CellOutputItem('text/plain', line));
          },
        ),
      );
    },
    stdout: () => _NotebookStdout(stdout, capturedStdout),
    stderr: () => _NotebookStdout(stderr, capturedStderr),
  );
}

class _NotebookStdout implements Stdout {
  final Stdout _delegate;
  final StringBuffer _buffer;
  _NotebookStdout(this._delegate, this._buffer);

  @override
  String get lineTerminator => _delegate.lineTerminator;
  @override
  set lineTerminator(String value) {
    _delegate.lineTerminator = value;
  }

  @override
  Encoding get encoding => _delegate.encoding;
  @override
  set encoding(Encoding encoding) {
    _delegate.encoding = encoding;
  }

  @override
  void write(Object? object) {
    _buffer.write(object);
  }

  @override
  void writeln([Object? object = ""]) {
    _buffer.writeln(object);
  }

  @override
  void writeAll(Iterable<dynamic> objects, [String separator = ""]) {
    _buffer.writeAll(objects, separator);
  }

  @override
  void writeCharCode(int charCode) {
    _buffer.writeCharCode(charCode);
  }

  @override
  void add(List<int> data) {
    _buffer.write(utf8.decode(data, allowMalformed: true));
  }

  @override
  void addError(Object error, [StackTrace? stackTrace]) {}
  @override
  Future<void> addStream(Stream<List<int>> stream) async {}
  Future<void> clearLine([int length = 0]) => Future.value();
  @override
  Future<void> close() => Future.value();
  @override
  Future<void> get done => _delegate.done;
  @override
  Future<void> flush() => Future.value();
  @override
  bool get hasTerminal => false;
  @override
  IOSink get nonBlocking => _delegate.nonBlocking;
  @override
  bool get supportsAnsiEscapes => false;
  @override
  int get terminalColumns => 80;
  @override
  int get terminalLines => 24;
}
