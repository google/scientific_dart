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

// ignore_for_file: type=lint, unused_element, unused_field, camel_case_types, non_constant_identifier_names

import 'dart:ffi' as ffi;
import 'symengine_bindings.dart';

final class CDenseMatrix extends ffi.Opaque {}

@ffi.Native<ffi.Pointer<CDenseMatrix> Function()>(
  symbol: 'dense_matrix_new',
  assetId: 'package:symbolic_dart/symengine',
)
external ffi.Pointer<CDenseMatrix> dense_matrix_new();

@ffi.Native<ffi.Void Function(ffi.Pointer<CDenseMatrix>)>(
  symbol: 'dense_matrix_free',
  assetId: 'package:symbolic_dart/symengine',
)
external void dense_matrix_free(ffi.Pointer<CDenseMatrix> self);

@ffi.Native<
  ffi.Pointer<CDenseMatrix> Function(ffi.UnsignedInt, ffi.UnsignedInt)
>(
  symbol: 'dense_matrix_new_rows_cols',
  assetId: 'package:symbolic_dart/symengine',
)
external ffi.Pointer<CDenseMatrix> dense_matrix_new_rows_cols(int r, int c);

@ffi.Native<ffi.UnsignedInt Function(ffi.Pointer<CDenseMatrix>)>(
  symbol: 'dense_matrix_rows',
  assetId: 'package:symbolic_dart/symengine',
)
external int dense_matrix_rows(ffi.Pointer<CDenseMatrix> s);

@ffi.Native<ffi.UnsignedInt Function(ffi.Pointer<CDenseMatrix>)>(
  symbol: 'dense_matrix_cols',
  assetId: 'package:symbolic_dart/symengine',
)
external int dense_matrix_cols(ffi.Pointer<CDenseMatrix> s);

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<basic_struct>,
    ffi.Pointer<CDenseMatrix>,
    ffi.UnsignedLong,
    ffi.UnsignedLong,
  )
>(symbol: 'dense_matrix_get_basic', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_get_basic(
  ffi.Pointer<basic_struct> s,
  ffi.Pointer<CDenseMatrix> mat,
  int r,
  int c,
);
symengine_exceptions_t dense_matrix_get_basic(
  ffi.Pointer<basic_struct> s,
  ffi.Pointer<CDenseMatrix> mat,
  int r,
  int c,
) => symengine_exceptions_t.fromValue(_dense_matrix_get_basic(s, mat, r, c));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.UnsignedLong,
    ffi.UnsignedLong,
    ffi.Pointer<basic_struct>,
  )
>(symbol: 'dense_matrix_set_basic', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_set_basic(
  ffi.Pointer<CDenseMatrix> mat,
  int r,
  int c,
  ffi.Pointer<basic_struct> s,
);
symengine_exceptions_t dense_matrix_set_basic(
  ffi.Pointer<CDenseMatrix> mat,
  int r,
  int c,
  ffi.Pointer<basic_struct> s,
) => symengine_exceptions_t.fromValue(_dense_matrix_set_basic(mat, r, c, s));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
  )
>(symbol: 'dense_matrix_add_matrix', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_add_matrix(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> a,
  ffi.Pointer<CDenseMatrix> b,
);
symengine_exceptions_t dense_matrix_add_matrix(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> a,
  ffi.Pointer<CDenseMatrix> b,
) => symengine_exceptions_t.fromValue(_dense_matrix_add_matrix(s, a, b));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
  )
>(symbol: 'dense_matrix_mul_matrix', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_mul_matrix(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> a,
  ffi.Pointer<CDenseMatrix> b,
);
symengine_exceptions_t dense_matrix_mul_matrix(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> a,
  ffi.Pointer<CDenseMatrix> b,
) => symengine_exceptions_t.fromValue(_dense_matrix_mul_matrix(s, a, b));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<basic_struct>,
  )
>(symbol: 'dense_matrix_mul_scalar', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_mul_scalar(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> a,
  ffi.Pointer<basic_struct> b,
);
symengine_exceptions_t dense_matrix_mul_scalar(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> a,
  ffi.Pointer<basic_struct> b,
) => symengine_exceptions_t.fromValue(_dense_matrix_mul_scalar(s, a, b));

@ffi.Native<
  ffi.UnsignedInt Function(ffi.Pointer<CDenseMatrix>, ffi.Pointer<CDenseMatrix>)
>(symbol: 'dense_matrix_transpose', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_transpose(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> mat,
);
symengine_exceptions_t dense_matrix_transpose(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> mat,
) => symengine_exceptions_t.fromValue(_dense_matrix_transpose(s, mat));

@ffi.Native<
  ffi.UnsignedInt Function(ffi.Pointer<basic_struct>, ffi.Pointer<CDenseMatrix>)
>(symbol: 'dense_matrix_det', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_det(
  ffi.Pointer<basic_struct> s,
  ffi.Pointer<CDenseMatrix> mat,
);
symengine_exceptions_t dense_matrix_det(
  ffi.Pointer<basic_struct> s,
  ffi.Pointer<CDenseMatrix> mat,
) => symengine_exceptions_t.fromValue(_dense_matrix_det(s, mat));

@ffi.Native<
  ffi.UnsignedInt Function(ffi.Pointer<CDenseMatrix>, ffi.Pointer<CDenseMatrix>)
>(symbol: 'dense_matrix_inv', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_inv(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> mat,
);
symengine_exceptions_t dense_matrix_inv(
  ffi.Pointer<CDenseMatrix> s,
  ffi.Pointer<CDenseMatrix> mat,
) => symengine_exceptions_t.fromValue(_dense_matrix_inv(s, mat));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
  )
>(symbol: 'dense_matrix_LU_solve', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_LU_solve(
  ffi.Pointer<CDenseMatrix> x,
  ffi.Pointer<CDenseMatrix> A,
  ffi.Pointer<CDenseMatrix> b,
);
symengine_exceptions_t dense_matrix_LU_solve(
  ffi.Pointer<CDenseMatrix> x,
  ffi.Pointer<CDenseMatrix> A,
  ffi.Pointer<CDenseMatrix> b,
) => symengine_exceptions_t.fromValue(_dense_matrix_LU_solve(x, A, b));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<basic_struct>,
  )
>(symbol: 'dense_matrix_diff', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_diff(
  ffi.Pointer<CDenseMatrix> result,
  ffi.Pointer<CDenseMatrix> A,
  ffi.Pointer<basic_struct> x,
);
symengine_exceptions_t dense_matrix_diff(
  ffi.Pointer<CDenseMatrix> result,
  ffi.Pointer<CDenseMatrix> A,
  ffi.Pointer<basic_struct> x,
) => symengine_exceptions_t.fromValue(_dense_matrix_diff(result, A, x));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
    ffi.Pointer<CDenseMatrix>,
  )
>(symbol: 'dense_matrix_jacobian', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_jacobian(
  ffi.Pointer<CDenseMatrix> result,
  ffi.Pointer<CDenseMatrix> A,
  ffi.Pointer<CDenseMatrix> x,
);
symengine_exceptions_t dense_matrix_jacobian(
  ffi.Pointer<CDenseMatrix> result,
  ffi.Pointer<CDenseMatrix> A,
  ffi.Pointer<CDenseMatrix> x,
) => symengine_exceptions_t.fromValue(_dense_matrix_jacobian(result, A, x));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.UnsignedInt,
    ffi.UnsignedInt,
    ffi.Int,
  )
>(symbol: 'dense_matrix_eye', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_eye(
  ffi.Pointer<CDenseMatrix> s,
  int n,
  int m,
  int k,
);
symengine_exceptions_t dense_matrix_eye(
  ffi.Pointer<CDenseMatrix> s,
  int n,
  int m,
  int k,
) => symengine_exceptions_t.fromValue(_dense_matrix_eye(s, n, m, k));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.UnsignedInt,
    ffi.UnsignedInt,
  )
>(symbol: 'dense_matrix_zeros', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_zeros(ffi.Pointer<CDenseMatrix> s, int r, int c);
symengine_exceptions_t dense_matrix_zeros(
  ffi.Pointer<CDenseMatrix> s,
  int r,
  int c,
) => symengine_exceptions_t.fromValue(_dense_matrix_zeros(s, r, c));

@ffi.Native<
  ffi.UnsignedInt Function(
    ffi.Pointer<CDenseMatrix>,
    ffi.UnsignedInt,
    ffi.UnsignedInt,
  )
>(symbol: 'dense_matrix_ones', assetId: 'package:symbolic_dart/symengine')
external int _dense_matrix_ones(ffi.Pointer<CDenseMatrix> s, int r, int c);
symengine_exceptions_t dense_matrix_ones(
  ffi.Pointer<CDenseMatrix> s,
  int r,
  int c,
) => symengine_exceptions_t.fromValue(_dense_matrix_ones(s, r, c));

@ffi.Native<
  ffi.Int Function(ffi.Pointer<CDenseMatrix>, ffi.Pointer<CDenseMatrix>)
>(symbol: 'dense_matrix_eq', assetId: 'package:symbolic_dart/symengine')
external int dense_matrix_eq(
  ffi.Pointer<CDenseMatrix> a,
  ffi.Pointer<CDenseMatrix> b,
);

@ffi.Native<ffi.Pointer<ffi.Char> Function(ffi.Pointer<CDenseMatrix>)>(
  symbol: 'dense_matrix_str',
  assetId: 'package:symbolic_dart/symengine',
)
external ffi.Pointer<ffi.Char> dense_matrix_str(ffi.Pointer<CDenseMatrix> s);
