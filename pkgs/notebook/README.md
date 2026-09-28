# Scientific Dart Notebook (`package:notebook`) — Experimental

> [!WARNING]
> **Experimental**: This package is an experimental prototype and under active development. Its architecture, protocol, and UI may change significantly.

A web-based interactive notebook and REPL interface for **Dart** and **`package:ndarray`**, powered by the Dart VM Service, LSP autocompletions, and high-performance SIMD/FFI C bindings.

![Notebook Overview](doc/images/media__notebook_full.png)

## Features

- **Interactive Dart REPL Kernel:** Execute Dart expressions, loop structures, and top-level functions with live hot-reload state persistence across cells.
- **LSP Autocompletion & Hover Info:** Real-time symbol completion and hover type popups powered by the Dart Language Server Protocol.
- **Structured MIME Output Protocol:** Zero string-escaping hacks. Rich HTML widgets render directly into the DOM while plain text stdout/prints output cleanly.
- **Rich Scientific Visualizers (`Displayable` Widget Suite):**
  - **`Plot`**: Responsive 2D line charts with auto-scaled axes and grid lines.
  - **`Heatmap`**: Matrix visualizer with hover value tooltips.
  - **`Animation`**: Multi-frame image sequence player with interactive Play/Pause controls.
  - **`Audio`**: Converts 1D `NDArray` samples into 16-bit PCM WAV audio clips.
  - **`Spectrogram`**: Real-time STFT time-frequency heatmap visualizer.
  - **`Table`**: Formatted dark-themed data matrix tables.
  - **`Histogram`**: Frequency distribution bar charts.
  - **`LaTeX` / `Latex`**: Mathematical equation renderer powered by KaTeX.
  - **`Markdown` / `Md`**: Formatted markdown documentation widgets.
- **Multiple Cell Outputs:** Call `display(widget)` to output multiple visualizers from a single notebook cell.

---

## Getting Started

### 1. Launching the Notebook Server

Run the server executable specifying the port (default `8080`):

```bash
dart bin/notebook_server.dart 8080
```

Open **`http://localhost:8080`** in your browser.

---

## Widget Documentation & Examples

### 1. `Plot` — 2D Line Charts
Renders 2D line plots wrapping 1D `NDArray` coordinates.

```dart
var x = linspace<Float64>(Float64(0.0), Float64(6.28), 100);
var y = sin(x);
Plot(y, x: x, title: 'Sine Wave Plot')
```

![Plot Widget](doc/images/media__plot_widget.png)

---

### 2. `Heatmap` — 2D Matrix Heatmaps
Renders 2D matrix arrays with color mapping and cell value hover tooltips.

```dart
var grid = NDArray.fromList([
  1.0, 2.0, 3.0,
  4.0, 5.0, 6.0,
  7.0, 8.0, 9.0
], [3, 3], DType.float64);
Heatmap(grid, title: '3x3 Matrix Heatmap')
```

![Heatmap Widget](doc/images/media__heatmap_widget.png)

---

### 3. `Animation` — Frame Sequence Player
Plays multi-frame `NDArray` image sequences with interactive Play/Pause and scrubber controls.

```dart
var frames = <NDArray>[];
for (var i = 0; i < 20; i++) {
  var h = linspace<Float64>(Float64(0.0), Float64(1.0), 50);
  var phase = (h + Float64(i / 20.0)).reshape([1, 50, 1]);
  var r = broadcastTo(phase, [50, 50, 1]);
  var g = broadcastTo(h.reshape([50, 1, 1]), [50, 50, 1]);
  var b = NDArray.full([50, 50, 1], Float64(0.5));
  frames.add(concatenate([r, g, b], axis: 2));
}
Animation(frames, fps: 15)
```

![Animation Widget](doc/images/media__animation_widget.png)

---

### 4. `Table` — Formatted Matrix Tables
Renders 2D matrix arrays into clean dark-themed HTML tables with custom column headers.

```dart
var matrix = NDArray.fromList([
  1.0, 2.5, 3.8,
  4.2, 5.1, 6.9,
  7.0, 8.4, 9.2
], [3, 3], DType.float64);
Table(matrix, headers: ['Alpha', 'Beta', 'Gamma'], title: 'Experimental Data Matrix')
```

![Table Widget](doc/images/media__table_widget.png)

---

### 5. `Histogram` — Frequency Distribution Charts
Computes statistical distribution bins for 1D arrays and displays SVG bar charts.

```dart
var data = NDArray.fromList([
  1.2, 1.5, 1.8, 2.1, 2.1, 2.2, 2.3, 2.5, 2.8, 3.0, 3.1, 3.2, 3.5, 4.1
], [14], DType.float64);
Histogram(data, bins: 5, title: 'Sample Distribution')
```

![Histogram Widget](doc/images/media__histogram_widget.png)

---

### 6. `Audio` & `Spectrogram` — Audio Waveforms & Frequency Heatmaps
Converts 1D sample arrays into playable 16-bit PCM WAV audio clips and real-time STFT frequency heatmaps.

```dart
var tone1 = linspace<Float64>(Float64(440.0), Float64(880.0), 44100 ~/ 2);
var tone2 = linspace<Float64>(Float64(880.0), Float64(440.0), 44100 ~/ 2);
var tone = concatenate(<NDArray>[tone1, tone2]);
var phase = cumsum(tone) * (2 * math.pi / 44100.0);
var samples = sin(phase) * 0.5;

display(Audio(samples, sampleRate: 44100));
display(Spectrogram(samples, sampleRate: 44100, title: 'Chirp Spectrogram'))
```

![Spectrogram Widget](doc/images/media__spectrogram_widget.png)

---

### 7. `LaTeX` / `Latex` — Math Equation Rendering
Renders LaTeX math formulas using KaTeX.

```dart
LaTeX(r'\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}')
```

![LaTeX Widget](doc/images/media__latex_widget.png)

---

### 8. `Markdown` / `Md` — Rich Text Formatting
Renders Markdown documentation inside notebook cells.

```dart
Markdown('# Scientific Dart Notebook\n\nInteractive **NDArray** computation kernel powered by **Dart FFI & SIMD**.')
```

![Markdown Widget](doc/images/media__markdown_widget.png)
