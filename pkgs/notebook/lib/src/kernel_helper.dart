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
import 'dart:convert';
import 'dart:developer' as developer;
import 'dart:io';
import 'package:ndarray/ndarray.dart';
import 'package:symbolic_dart/symbolic_dart.dart'
    hide sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, abs;

import 'notebook_widgets.dart' as widgets;
export 'notebook_widgets.dart' hide evalInNotebookZone;

/// Evaluates and plots a 1D symbolic expression [f] with respect to variable [varName]
/// over the range [[from], [to]] using [points] samples.
widgets.Plot plotSymbolic(
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
  return widgets.Plot(
    x: xArr,
    y: yArr,
    title: title ?? 'f($varName) = $f',
    color: color,
  );
}

/// Evaluates and plots a 2D symbolic expression [f] with respect to variables [xVar] and [yVar]
/// as a 2D heatmap over the ranges [[xFrom], [xTo]] and [[yFrom], [yTo]].
widgets.Heatmap plotSymbolic2D(
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
  return widgets.Heatmap(zArr, title: title ?? 'f($xVar, $yVar) = $f');
}

/// Executes [body] in a notebook zone that captures `print` and standard I/O output.
///
/// Uncaught asynchronous errors raised inside the zone (for example from a
/// `Future` or `Timer` callback that nobody awaits) are recorded as captured
/// output instead of propagating to the root zone, where they would terminate
/// the kernel isolate.
dynamic evalInNotebookZone(dynamic Function() body) {
  if (const bool.fromEnvironment('dart.tool.dart2wasm')) {
    return widgets.evalInNotebookZone(body);
  }
  return IOOverrides.runZoned(
    () {
      return runZoned(
        body,
        zoneSpecification: ZoneSpecification(
          print: (self, parent, zone, line) {
            widgets.capturedOutputs.add(
              widgets.CellOutputItem('text/plain', line),
            );
          },
          handleUncaughtError: (self, parent, zone, error, stackTrace) {
            widgets.capturedOutputs.add(
              widgets.CellOutputItem(
                'text/plain',
                'Unhandled async error: $error\n$stackTrace',
              ),
            );
          },
        ),
      );
    },
    stdout: () => _NotebookStdout(stdout, widgets.capturedStdout),
    stderr: () => _NotebookStdout(stderr, widgets.capturedStderr),
  );
}

/// Kind of the `dart:developer` extension event posted when a cell started by
/// [runNotebookCell] has completed. Its data is `{'runId': <int>}`.
const String notebookCellDoneEvent = 'notebook.cellDone';

bool _cellDone = true;
dynamic _cellValue;
Object? _cellError;
StackTrace? _cellStackTrace;

/// Starts the cell body [body] for the run identified by [runId] and returns
/// [runId] as soon as the body suspends or completes.
///
/// The body runs inside [evalInNotebookZone], so `print` output and
/// [display] calls are captured. Because [body] is asynchronous, cells may use
/// `await`, and a `Future` produced by the trailing expression is awaited so
/// that the cell value is its result rather than the future itself. Taking
/// the body as a `Future<dynamic> Function()` also lets a cell end in a `void`
/// expression: the closure's `Future<void>` return type is a subtype of
/// `Future<dynamic>`.
///
/// Completion, with a value or an error, is signalled by posting a
/// [notebookCellDoneEvent] extension event carrying [runId]; afterwards
/// [notebookCellIsDone] is `true` and [notebookCellResultJson] describes the
/// outcome. Previously captured output is discarded when the run starts.
int runNotebookCell(int runId, Future<dynamic> Function() body) {
  widgets.clearCapturedOutput();
  _cellDone = false;
  _cellValue = null;
  _cellError = null;
  _cellStackTrace = null;
  evalInNotebookZone(() {
    // `VmService.evaluate` invokes Dart directly from the VM service handler
    // without draining the microtask queue on return. Hopping onto the event
    // loop with `Timer.run` ensures `_RawReceivePort._handleMessage` drains all
    // microtasks scheduled by `body()` and its completion handlers.
    Timer.run(() {
      unawaited(
        body()
            .then<void>(
              (value) {
                _cellValue = value;
              },
              onError: (Object error, StackTrace stackTrace) {
                _cellError = error;
                _cellStackTrace = stackTrace;
              },
            )
            .whenComplete(() {
              _cellDone = true;
              developer.postEvent(notebookCellDoneEvent, {'runId': runId});
            }),
      );
    });
  });
  return runId;
}

/// Whether the most recent [runNotebookCell] run has completed.
bool notebookCellIsDone() => _cellDone;

/// The outcome of the most recent [runNotebookCell] run, encoded as a JSON
/// object with `isError` and `outputs`.
///
/// `outputs` lists the captured [widgets.CellOutputItem]s followed by the
/// formatted cell value (omitted when it is `null`) or, if the body threw, an
/// error item containing the error and its stack trace.
String notebookCellResultJson() {
  final outputs = List<widgets.CellOutputItem>.of(widgets.capturedOutputs);
  if (_cellError case final error?) {
    outputs.add(
      widgets.CellOutputItem('text/plain', 'Error: $error\n$_cellStackTrace'),
    );
  } else if (widgets.formatEvaluationValue(_cellValue) case final item?) {
    outputs.add(item);
  }
  return jsonEncode({
    'isError': _cellError != null,
    'outputs': [for (final item in outputs) item.toJson()],
  });
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
