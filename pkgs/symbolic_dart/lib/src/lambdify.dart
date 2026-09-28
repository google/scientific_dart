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

import 'dart:ffi' as ffi;
import 'package:ndarray/ndarray.dart';
import 'package:resource_scope/resource_scope.dart';
import 'expr.dart';
import 'ffi/symengine_bindings.dart' as se;

final _lambdaVisitorFinalizer = ffi.NativeFinalizer(
  ffi.Native.addressOf<
        ffi.NativeFunction<
          ffi.Void Function(ffi.Pointer<se.CLambdaRealDoubleVisitor>)
        >
      >(se.lambda_real_double_visitor_free)
      .cast<ffi.NativeFinalizerFunction>(),
);

/// A callable vectorized mathematical evaluator compiled from a symbolic [Expr].
///
/// Supports evaluating at scalar numbers via [callScalar] or vectorizing
/// over N-dimensional numerical arrays via [callArray] with optional broadcasting
/// and output destination buffer (`out`).
final class SymbolicLambda implements ffi.Finalizable, ScopedResource {
  final Expr _expr;
  final List<Expr> _variables;
  final ffi.Pointer<se.CLambdaRealDoubleVisitor> _visitor;
  bool _disposed = false;

  SymbolicLambda._(this._expr, this._variables)
    : _visitor = se.lambda_real_double_visitor_new() {
    if (_visitor == ffi.nullptr) {
      throw StateError('Cannot allocate CLambdaRealDoubleVisitor');
    }
    final argsVec = se.vecbasic_new();
    final exprsVec = se.vecbasic_new();
    try {
      for (final v in _variables) {
        se.vecbasic_push_back(argsVec, v.pointer);
      }
      se.vecbasic_push_back(exprsVec, _expr.pointer);
      se.lambda_real_double_visitor_init(_visitor, argsVec, exprsVec, 1);
    } finally {
      se.vecbasic_free(argsVec);
      se.vecbasic_free(exprsVec);
    }
    _lambdaVisitorFinalizer.attach(this, _visitor.cast(), detach: this);
    ResourceScope.track(this);
  }

  @override
  bool get isDisposed => _disposed;

  @override
  void dispose() {
    if (_disposed) return;
    _lambdaVisitorFinalizer.detach(this);
    se.lambda_real_double_visitor_free(_visitor);
    _disposed = true;
    ResourceScope.untrack(this);
  }

  @override
  SymbolicLambda detachFromScope() {
    ResourceScope.untrack(this);
    return this;
  }

  @override
  SymbolicLambda detachToParentScope() {
    ResourceScope.promoteToParent(this);
    return this;
  }

  void _checkDisposed() {
    if (_disposed) {
      throw StateError('This SymbolicLambda has already been disposed.');
    }
  }

  /// Evaluates the symbolic function at scalar points [values].
  ///
  /// It is an error if `values.length` does not match the number of parameters.
  double callScalar(List<num> values) {
    _checkDisposed();
    if (values.length != _variables.length) {
      throw ArgumentError(
        'Expected ${_variables.length} input values, got ${values.length}',
      );
    }
    final marker = ScratchArena.marker;
    try {
      final numVars = values.isEmpty ? 1 : values.length;
      final inPtr = ScratchArena.allocate<ffi.Double>(
        numVars * ffi.sizeOf<ffi.Double>(),
      );
      final outPtr = ScratchArena.allocate<ffi.Double>(
        ffi.sizeOf<ffi.Double>(),
      );
      for (var i = 0; i < values.length; i++) {
        inPtr[i] = values[i].toDouble();
      }
      se.lambda_real_double_visitor_call(_visitor, outPtr, inPtr);
      return outPtr[0];
    } finally {
      ScratchArena.reset(marker);
    }
  }

  /// Evaluates the symbolic function vectorized over N-dimensional arrays [inputs].
  ///
  /// If [out] is provided, writes results into [out] in-place. Otherwise allocates
  /// a new [NDArray<Float64>] matching the broadcast shape of [inputs].
  ///
  /// **Memory & Performance:**
  /// - Supports broadcasting multiple input arrays of compatible shapes.
  /// - Uses [ScratchArena] for transient native `double*` buffers.
  NDArray<Float64> callArray(
    List<NDArray<Float64>> inputs, {
    NDArray<Float64>? out,
  }) {
    _checkDisposed();
    if (inputs.length != _variables.length) {
      throw ArgumentError(
        'Expected ${_variables.length} array inputs, got ${inputs.length}',
      );
    }
    if (inputs.isEmpty) {
      final resScalar = callScalar(const []);
      if (out != null) {
        out.setCell([], (resScalar));
        return out;
      }
      return NDArray.scalar((resScalar), dtype: DType.float64);
    }

    // Determine broadcast shape
    final firstShape = inputs.first.shape;
    var broadcastShape = List<int>.from(firstShape);
    for (var i = 1; i < inputs.length; i++) {
      broadcastShape = _broadcastShapes(broadcastShape, inputs[i].shape);
    }

    final destination = out ?? NDArray.zeros(broadcastShape, DType.float64);
    bool shapeMatches = destination.shape.length == broadcastShape.length;
    if (shapeMatches) {
      for (var i = 0; i < broadcastShape.length; i++) {
        if (destination.shape[i] != broadcastShape[i]) {
          shapeMatches = false;
          break;
        }
      }
    }
    if (!shapeMatches) {
      throw ArgumentError(
        'out array shape ${destination.shape} does not match broadcast shape $broadcastShape',
      );
    }

    final marker = ScratchArena.marker;
    try {
      final inPtr = ScratchArena.allocate<ffi.Double>(
        _variables.length * ffi.sizeOf<ffi.Double>(),
      );
      final outPtr = ScratchArena.allocate<ffi.Double>(
        ffi.sizeOf<ffi.Double>(),
      );
      final iter = NDIter.broadcast([...inputs, destination]);
      while (iter.moveNext()) {
        final coords = iter.coords;
        for (var i = 0; i < inputs.length; i++) {
          final inShape = inputs[i].shape;
          final inCoords = List<int>.generate(inShape.length, (dimIdx) {
            final axisOffset = broadcastShape.length - inShape.length;
            final mappedDim = dimIdx + axisOffset;
            if (mappedDim < 0) return 0;
            return inShape[dimIdx] == 1 ? 0 : coords[mappedDim];
          });
          inPtr[i] = inputs[i].getCell(inCoords);
        }
        se.lambda_real_double_visitor_call(_visitor, outPtr, inPtr);
        destination.setCell(coords, (outPtr[0]));
      }
    } finally {
      ScratchArena.reset(marker);
    }

    return destination;
  }

  static List<int> _broadcastShapes(List<int> a, List<int> b) {
    final maxLen = a.length > b.length ? a.length : b.length;
    final aPadded = List<int>.filled(maxLen - a.length, 1, growable: true)
      ..addAll(a);
    final bPadded = List<int>.filled(maxLen - b.length, 1, growable: true)
      ..addAll(b);
    final result = <int>[];
    for (var i = 0; i < maxLen; i++) {
      if (aPadded[i] == bPadded[i]) {
        result.add(aPadded[i]);
      } else if (aPadded[i] == 1) {
        result.add(bPadded[i]);
      } else if (bPadded[i] == 1) {
        result.add(aPadded[i]);
      } else {
        throw ArgumentError(
          'Incompatible broadcast shapes: $a and $b at dimension $i',
        );
      }
    }
    return result;
  }
}

extension LambdifyExtension on Expr {
  /// Compiles this symbolic expression into a [SymbolicLambda] evaluator
  /// taking the specified [variables] as parameters.
  SymbolicLambda lambdify(List<Expr> variables) {
    return SymbolicLambda._(this, List<Expr>.unmodifiable(variables));
  }
}
