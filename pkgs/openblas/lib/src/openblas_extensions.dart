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

// ignore_for_file: non_constant_identifier_names
@ffi.DefaultAsset('package:openblas/openblas')
library;

import 'dart:ffi' as ffi;

import 'openblas_bindings.dart'; // For blasint!

@ffi.Native<
  ffi.Void Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    blasint,
    blasint,
    blasint,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    blasint,
    ffi.Pointer<ffi.Float>,
    blasint,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    blasint,
  )
>()
external void cblas_cgemm(
  int order,
  int transA,
  int transB,
  int m,
  int n,
  int k,
  ffi.Pointer<ffi.Float> alpha,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> b,
  int ldb,
  ffi.Pointer<ffi.Float> beta,
  ffi.Pointer<ffi.Float> c,
  int ldc,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    blasint,
    blasint,
    blasint,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    blasint,
    ffi.Pointer<ffi.Double>,
    blasint,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    blasint,
  )
>()
external void cblas_zgemm(
  int order,
  int transA,
  int transB,
  int m,
  int n,
  int k,
  ffi.Pointer<ffi.Double> alpha,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> b,
  int ldb,
  ffi.Pointer<ffi.Double> beta,
  ffi.Pointer<ffi.Double> c,
  int ldc,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Int,
    ffi.Int,
    blasint,
    blasint,
    ffi.Double,
    ffi.Pointer<ffi.Double>,
    blasint,
    ffi.Pointer<ffi.Double>,
    blasint,
    ffi.Double,
    ffi.Pointer<ffi.Double>,
    blasint,
  )
>()
external void cblas_dgemv(
  int order,
  int transA,
  int m,
  int n,
  double alpha,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> x,
  int incx,
  double beta,
  ffi.Pointer<ffi.Double> y,
  int incy,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Int,
    ffi.Int,
    blasint,
    blasint,
    ffi.Float,
    ffi.Pointer<ffi.Float>,
    blasint,
    ffi.Pointer<ffi.Float>,
    blasint,
    ffi.Float,
    ffi.Pointer<ffi.Float>,
    blasint,
  )
>()
external void cblas_sgemv(
  int order,
  int transA,
  int m,
  int n,
  double alpha,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> x,
  int incx,
  double beta,
  ffi.Pointer<ffi.Float> y,
  int incy,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Int,
    ffi.Int,
    blasint,
    blasint,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    blasint,
    ffi.Pointer<ffi.Float>,
    blasint,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    blasint,
  )
>()
external void cblas_cgemv(
  int order,
  int transA,
  int m,
  int n,
  ffi.Pointer<ffi.Float> alpha,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> x,
  int incx,
  ffi.Pointer<ffi.Float> beta,
  ffi.Pointer<ffi.Float> y,
  int incy,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Int,
    ffi.Int,
    blasint,
    blasint,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    blasint,
    ffi.Pointer<ffi.Double>,
    blasint,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    blasint,
  )
>()
external void cblas_zgemv(
  int order,
  int transA,
  int m,
  int n,
  ffi.Pointer<ffi.Double> alpha,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> x,
  int incx,
  ffi.Pointer<ffi.Double> beta,
  ffi.Pointer<ffi.Double> y,
  int incy,
);
