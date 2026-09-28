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
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:resource_scope/resource_scope.dart';
import 'expr.dart';
import 'ffi/matrix_bindings.dart' as mat;
import 'ffi/symengine_bindings.dart' as se;

final _denseMatrixFinalizer = ffi.NativeFinalizer(
  ffi.Native.addressOf<
        ffi.NativeFunction<ffi.Void Function(ffi.Pointer<mat.CDenseMatrix>)>
      >(mat.dense_matrix_free)
      .cast<ffi.NativeFinalizerFunction>(),
);

final _lambdaVisitorFinalizer = ffi.NativeFinalizer(
  ffi.Native.addressOf<
        ffi.NativeFunction<
          ffi.Void Function(ffi.Pointer<se.CLambdaRealDoubleVisitor>)
        >
      >(se.lambda_real_double_visitor_free)
      .cast<ffi.NativeFinalizerFunction>(),
);

/// A callable compiled matrix evaluator that evaluates a [SymbolicMatrix]
/// into a 2D [NDArray<Float64>] for given variable points.
final class MatrixLambda implements ffi.Finalizable, ScopedResource {
  final SymbolicMatrix _matrix;
  final List<Expr> _variables;
  final int _rows;
  final int _cols;
  final ffi.Pointer<se.CLambdaRealDoubleVisitor> _visitor;
  bool _disposed = false;

  MatrixLambda._(this._matrix, this._variables)
    : _rows = _matrix.rows,
      _cols = _matrix.cols,
      _visitor = se.lambda_real_double_visitor_new() {
    if (_visitor == ffi.nullptr) {
      throw StateError('Cannot allocate CLambdaRealDoubleVisitor');
    }
    final argsVec = se.vecbasic_new();
    final exprsVec = se.vecbasic_new();
    final cellPtr = se.basic_new_heap();
    try {
      for (final v in _variables) {
        se.vecbasic_push_back(argsVec, v.pointer);
      }
      for (var i = 0; i < _rows; i++) {
        for (var j = 0; j < _cols; j++) {
          mat.dense_matrix_get_basic(cellPtr, _matrix.pointer, i, j);
          se.vecbasic_push_back(exprsVec, cellPtr);
        }
      }
      se.lambda_real_double_visitor_init(_visitor, argsVec, exprsVec, 1);
    } finally {
      se.basic_free_heap(cellPtr);
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
  MatrixLambda detachFromScope() {
    ResourceScope.untrack(this);
    return this;
  }

  @override
  MatrixLambda detachToParentScope() {
    ResourceScope.promoteToParent(this);
    return this;
  }

  void _checkDisposed() {
    if (_disposed) {
      throw StateError('This MatrixLambda has already been disposed.');
    }
  }

  /// Evaluates the symbolic matrix into a 2D [NDArray<Float64>] at scalar points [values].
  NDArray<Float64> callScalar(List<num> values, {NDArray<Float64>? out}) {
    _checkDisposed();
    if (values.length != _variables.length) {
      throw ArgumentError(
        'Expected ${_variables.length} parameters, got ${values.length}',
      );
    }
    final destination = out ?? NDArray.zeros([_rows, _cols], DType.float64);
    if (destination.shape.length != 2 ||
        destination.shape[0] != _rows ||
        destination.shape[1] != _cols) {
      throw ArgumentError(
        'Expected 2D out array of shape [$_rows, $_cols], got ${destination.shape}',
      );
    }
    final numInputs = _variables.length;
    final totalCells = _rows * _cols;
    final marker = ScratchArena.marker;
    try {
      final inPtr = ScratchArena.allocate<ffi.Double>(
        (numInputs == 0 ? 1 : numInputs) * ffi.sizeOf<ffi.Double>(),
      );
      final outPtr = ScratchArena.allocate<ffi.Double>(
        (totalCells == 0 ? 1 : totalCells) * ffi.sizeOf<ffi.Double>(),
      );
      for (var i = 0; i < numInputs; i++) {
        inPtr[i] = values[i].toDouble();
      }
      se.lambda_real_double_visitor_call(_visitor, outPtr, inPtr);
      for (var i = 0; i < _rows; i++) {
        for (var j = 0; j < _cols; j++) {
          destination.setCell([i, j], (outPtr[i * _cols + j]));
        }
      }
    } finally {
      ScratchArena.reset(marker);
    }
    return destination;
  }
}

/// A dense symbolic matrix backed by SymEngine's native `CDenseMatrix` C++ engine.
///
/// Supports exact symbolic matrix algebra, determinant ([det]), inverse ([inv]),
/// LU linear equation solving ([solve]), elementwise differentiation ([diff]),
/// analytical Jacobian generation ([jacobian]), and seamless interoperability
/// with `package:ndarray` ([fromNDArray] and [toNDArray]).
final class SymbolicMatrix implements ffi.Finalizable, ScopedResource {
  final ffi.Pointer<mat.CDenseMatrix> _ptr;
  bool _disposed = false;

  SymbolicMatrix._(this._ptr) {
    if (_ptr == ffi.nullptr) {
      throw StateError('Cannot wrap nullptr in SymbolicMatrix');
    }
    _denseMatrixFinalizer.attach(this, _ptr.cast(), detach: this);
    ResourceScope.track(this);
  }

  @override
  bool get isDisposed => _disposed;

  @override
  void dispose() {
    if (_disposed) return;
    _denseMatrixFinalizer.detach(this);
    mat.dense_matrix_free(_ptr);
    _disposed = true;
    ResourceScope.untrack(this);
  }

  @override
  SymbolicMatrix detachFromScope() {
    ResourceScope.untrack(this);
    return this;
  }

  @override
  SymbolicMatrix detachToParentScope() {
    ResourceScope.promoteToParent(this);
    return this;
  }

  void _checkDisposed() {
    if (_disposed) {
      throw StateError('This SymbolicMatrix has already been disposed.');
    }
  }

  /// Allocates a new empty [SymbolicMatrix] with [r] rows and [c] columns.
  factory SymbolicMatrix.zeros(int r, int c) {
    if (r <= 0 || c <= 0) {
      throw ArgumentError('Matrix dimensions must be positive');
    }
    final ptr = mat.dense_matrix_new();
    mat.dense_matrix_zeros(ptr, r, c);
    return SymbolicMatrix._(ptr);
  }

  /// Allocates a new [SymbolicMatrix] of ones with [r] rows and [c] columns.
  factory SymbolicMatrix.ones(int r, int c) {
    if (r <= 0 || c <= 0) {
      throw ArgumentError('Matrix dimensions must be positive');
    }
    final ptr = mat.dense_matrix_new();
    mat.dense_matrix_ones(ptr, r, c);
    return SymbolicMatrix._(ptr);
  }

  /// Allocates an identity [SymbolicMatrix] of size [n] x [m].
  /// If [m] is omitted, creates a square [n] x [n] identity matrix.
  factory SymbolicMatrix.eye(int n, [int? m, int k = 0]) {
    if (n <= 0) {
      throw ArgumentError('Matrix dimension must be positive');
    }
    final cols = m ?? n;
    final ptr = mat.dense_matrix_new();
    mat.dense_matrix_eye(ptr, n, cols, k);
    return SymbolicMatrix._(ptr);
  }

  /// Creates a [SymbolicMatrix] from a 2D list of row values
  /// (can be [Expr], [num], [String], or [BigInt]).
  factory SymbolicMatrix.fromList(List<List<Object>> rows) {
    if (rows.isEmpty || rows.first.isEmpty) {
      throw ArgumentError('Matrix rows cannot be empty');
    }
    final r = rows.length;
    final c = rows.first.length;
    final ptr = mat.dense_matrix_new_rows_cols(r, c);
    for (var i = 0; i < r; i++) {
      if (rows[i].length != c) {
        throw ArgumentError('Inconsistent row length at index $i');
      }
      for (var j = 0; j < c; j++) {
        final e = Expr.fromObject(rows[i][j]);
        mat.dense_matrix_set_basic(ptr, i, j, e.pointer);
      }
    }
    return SymbolicMatrix._(ptr);
  }

  /// Creates a column vector [SymbolicMatrix] (`n` rows x `1` column) from [values].
  factory SymbolicMatrix.fromVector(List<Object> values) {
    return SymbolicMatrix.fromList(values.map((v) => [v]).toList());
  }

  /// Converts a numeric [NDArray] (1D or 2D) into a [SymbolicMatrix].
  ///
  /// - If [arr] is 1D (`[n]`), creates an `n x 1` column vector matrix.
  /// - If [arr] is 2D (`[r, c]`), creates an `r x c` matrix.
  factory SymbolicMatrix.fromNDArray(NDArray arr) {
    final shape = arr.shape;
    if (shape.length == 1) {
      final r = shape[0];
      final matPtr = mat.dense_matrix_new_rows_cols(r, 1);
      final ePtr = se.basic_new_heap();
      try {
        for (var i = 0; i < r; i++) {
          final val = (arr.getCell([i]) as num).toDouble();
          se.real_double_set_d(ePtr, val);
          mat.dense_matrix_set_basic(matPtr, i, 0, ePtr);
        }
      } finally {
        se.basic_free_heap(ePtr);
      }
      return SymbolicMatrix._(matPtr);
    } else if (shape.length == 2) {
      final r = shape[0];
      final c = shape[1];
      final matPtr = mat.dense_matrix_new_rows_cols(r, c);
      final ePtr = se.basic_new_heap();
      try {
        for (var i = 0; i < r; i++) {
          for (var j = 0; j < c; j++) {
            final val = (arr.getCell([i, j]) as num).toDouble();
            se.real_double_set_d(ePtr, val);
            mat.dense_matrix_set_basic(matPtr, i, j, ePtr);
          }
        }
      } finally {
        se.basic_free_heap(ePtr);
      }
      return SymbolicMatrix._(matPtr);
    } else {
      throw ArgumentError(
        'SymbolicMatrix.fromNDArray expects a 1D or 2D array, got shape $shape',
      );
    }
  }

  ffi.Pointer<mat.CDenseMatrix> get pointer {
    _checkDisposed();
    return _ptr;
  }

  /// Number of rows.
  int get rows => mat.dense_matrix_rows(pointer);

  /// Number of columns.
  int get cols => mat.dense_matrix_cols(pointer);

  /// Matrix shape tuple `(rows: r, cols: c)`.
  ({int rows, int cols}) get shape => (rows: rows, cols: cols);

  /// Retrieves the symbolic expression at row [r], column [c] (0-indexed).
  ///
  /// It is an error if [r] or [c] is out of bounds.
  Expr getCell(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
      throw RangeError.index(r, this, 'r', 'Index out of matrix bounds');
    }
    final res = se.basic_new_heap();
    mat.dense_matrix_get_basic(res, pointer, r, c);
    return Expr.fromPointer(res);
  }

  /// Sets the element at row [r], column [c] to [value].
  void setCell(int r, int c, Object value) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
      throw RangeError.index(r, this, 'r', 'Index out of matrix bounds');
    }
    final e = Expr.fromObject(value);
    mat.dense_matrix_set_basic(pointer, r, c, e.pointer);
  }

  /// Matrix addition.
  SymbolicMatrix operator +(SymbolicMatrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw ArgumentError(
        'Incompatible matrix dimensions for addition: $shape vs ${other.shape}',
      );
    }
    final res = mat.dense_matrix_new();
    mat.dense_matrix_add_matrix(res, pointer, other.pointer);
    return SymbolicMatrix._(res);
  }

  /// Matrix subtraction.
  SymbolicMatrix operator -(SymbolicMatrix other) {
    return this + (other * -1);
  }

  /// Matrix or scalar multiplication.
  ///
  /// - If [other] is a [SymbolicMatrix], performs matrix multiplication `A * B`.
  /// - Otherwise converts [other] to an [Expr] and performs scalar multiplication `c * A`.
  SymbolicMatrix operator *(Object other) {
    final res = mat.dense_matrix_new();
    if (other is SymbolicMatrix) {
      if (cols != other.rows) {
        throw ArgumentError(
          'Incompatible matrix dimensions for multiplication: $shape vs ${other.shape}',
        );
      }
      mat.dense_matrix_mul_matrix(res, pointer, other.pointer);
    } else {
      final s = Expr.fromObject(other);
      mat.dense_matrix_mul_scalar(res, pointer, s.pointer);
    }
    return SymbolicMatrix._(res);
  }

  /// Negation (`-this`).
  SymbolicMatrix operator -() => this * -1;

  /// Returns the transpose matrix `A^T`.
  SymbolicMatrix transpose() {
    final res = mat.dense_matrix_new();
    mat.dense_matrix_transpose(res, pointer);
    return SymbolicMatrix._(res);
  }

  /// Computes the exact symbolic determinant of a square matrix.
  ///
  /// Throws [StateError] if the matrix is not square.
  Expr det() {
    if (rows != cols) {
      throw StateError('Determinant is only defined for square matrices.');
    }
    final res = se.basic_new_heap();
    mat.dense_matrix_det(res, pointer);
    return Expr.fromPointer(res);
  }

  /// Computes the exact symbolic inverse matrix `A^{-1}`.
  ///
  /// Throws [StateError] if the matrix is not square or is singular.
  SymbolicMatrix inv() {
    if (rows != cols) {
      throw StateError('Inverse is only defined for square matrices.');
    }
    final res = mat.dense_matrix_new();
    final err = mat.dense_matrix_inv(res, pointer);
    if (err != se.symengine_exceptions_t.SYMENGINE_NO_EXCEPTION) {
      mat.dense_matrix_free(res);
      throw StateError('Matrix is singular or could not be inverted.');
    }
    return SymbolicMatrix._(res);
  }

  /// Solves the linear system `A * x = b` for `x` symbolically using LU decomposition.
  ///
  /// [b] must have the same number of rows as this matrix.
  SymbolicMatrix solve(SymbolicMatrix b) {
    if (rows != cols) {
      throw StateError('Solve requires a square system matrix A.');
    }
    if (rows != b.rows) {
      throw ArgumentError('RHS matrix b must have $rows rows, got ${b.rows}');
    }
    final res = mat.dense_matrix_new();
    final err = mat.dense_matrix_LU_solve(res, pointer, b.pointer);
    if (err != se.symengine_exceptions_t.SYMENGINE_NO_EXCEPTION) {
      mat.dense_matrix_free(res);
      throw StateError('Linear system A * x = b could not be solved.');
    }
    return SymbolicMatrix._(res);
  }

  /// Computes the element-wise partial derivative `dA / d(variable)`.
  SymbolicMatrix diff(Object variable) {
    final sym = Expr.fromObject(variable);
    final res = mat.dense_matrix_new();
    mat.dense_matrix_diff(res, pointer, sym.pointer);
    return SymbolicMatrix._(res);
  }

  /// Computes the analytical Jacobian matrix `J_{ij} = d(f_i) / d(x_j)`
  /// of this column vector or matrix with respect to [variables].
  ///
  /// If this is an `m x 1` vector of functions $\vec{f}$ and [variables] has length `n`,
  /// returns an `m x n` Jacobian matrix `J`.
  SymbolicMatrix jacobian(List<Object> variables) {
    if (rows != 1 && cols != 1) {
      throw ArgumentError(
        'Jacobian is only defined for a row or column vector (1xM or Mx1), got shape $shape',
      );
    }
    final m = (cols == 1) ? rows : cols;
    final n = variables.length;
    final res = mat.dense_matrix_new_rows_cols(m, n);
    for (var i = 0; i < m; i++) {
      final expr = (cols == 1) ? getCell(i, 0) : getCell(0, i);
      try {
        for (var j = 0; j < n; j++) {
          final d = expr.diff(variables[j]);
          try {
            mat.dense_matrix_set_basic(res, i, j, d.pointer);
          } finally {
            d.dispose();
          }
        }
      } finally {
        expr.dispose();
      }
    }
    return SymbolicMatrix._(res);
  }

  /// Substitutes symbols across all entries of this matrix according to [substitutions].
  SymbolicMatrix subs(Map<Object, Object> substitutions) {
    if (substitutions.isEmpty) return this;
    final r = rows;
    final c = cols;
    final res = mat.dense_matrix_new_rows_cols(r, c);
    for (var i = 0; i < r; i++) {
      for (var j = 0; j < c; j++) {
        final cell = getCell(i, j);
        try {
          final subCell = cell.subs(substitutions);
          try {
            mat.dense_matrix_set_basic(res, i, j, subCell.pointer);
          } finally {
            subCell.dispose();
          }
        } finally {
          cell.dispose();
        }
      }
    }
    return SymbolicMatrix._(res);
  }

  /// Compiles this symbolic matrix into a callable [MatrixLambda] taking [variables].
  MatrixLambda lambdify(List<Expr> variables) {
    return MatrixLambda._(this, List<Expr>.unmodifiable(variables));
  }

  /// Converts this matrix into a 2D [NDArray<Float64>] of shape `[rows, cols]`.
  ///
  /// Evaluates each matrix cell as a numeric double using [Expr.asDouble].
  /// If [out] is provided, writes results directly into [out] in-place.
  NDArray<Float64> toNDArray({NDArray<Float64>? out}) {
    final r = rows;
    final c = cols;
    final destination = out ?? NDArray.zeros([r, c], DType.float64);
    if (destination.shape.length != 2 ||
        destination.shape[0] != r ||
        destination.shape[1] != c) {
      throw ArgumentError(
        'Expected 2D out array of shape [$r, $c], got ${destination.shape}',
      );
    }
    for (var i = 0; i < r; i++) {
      for (var j = 0; j < c; j++) {
        final cell = getCell(i, j);
        try {
          final val = cell.asDouble;
          destination.setCell([i, j], (val));
        } finally {
          cell.dispose();
        }
      }
    }
    return destination;
  }

  /// Returns a nested 2D Dart list of [Expr] cells.
  List<List<Expr>> toList() {
    final r = rows;
    final c = cols;
    return List.generate(
      r,
      (i) => List.generate(c, (j) => getCell(i, j)),
      growable: false,
    );
  }

  @override
  String toString() {
    _checkDisposed();
    final cStr = mat.dense_matrix_str(pointer);
    if (cStr == ffi.nullptr) return '[]';
    try {
      return cStr.cast<Utf8>().toDartString();
    } finally {
      se.basic_str_free(cStr);
    }
  }

  /// Formats this matrix as a LaTeX `\begin{bmatrix} ... \end{bmatrix}` representation.
  String toLatex() {
    _checkDisposed();
    final r = rows;
    final c = cols;
    if (r == 0 || c == 0) return r'\begin{bmatrix}\end{bmatrix}';
    final sb = StringBuffer(r'\begin{bmatrix}');
    for (var i = 0; i < r; i++) {
      for (var j = 0; j < c; j++) {
        final cell = getCell(i, j);
        try {
          sb.write(cell.toLatex());
        } finally {
          cell.dispose();
        }
        if (j < c - 1) sb.write(' & ');
      }
      if (i < r - 1) sb.write(r' \\ ');
    }
    sb.write(r'\end{bmatrix}');
    return sb.toString();
  }

  /// Checks structural equality between two symbolic matrices.
  bool eq(SymbolicMatrix other) {
    return mat.dense_matrix_eq(pointer, other.pointer) != 0;
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! SymbolicMatrix) return false;
    return eq(other);
  }

  @override
  int get hashCode {
    var h = 0;
    final r = rows;
    final c = cols;
    for (var i = 0; i < r; i++) {
      for (var j = 0; j < c; j++) {
        final cell = getCell(i, j);
        try {
          h = h ^ cell.hashCode;
        } finally {
          cell.dispose();
        }
      }
    }
    return h;
  }
}
