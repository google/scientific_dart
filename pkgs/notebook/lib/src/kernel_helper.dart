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
        ),
      );
    },
    stdout: () => _NotebookStdout(stdout, widgets.capturedStdout),
    stderr: () => _NotebookStdout(stderr, widgets.capturedStderr),
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
