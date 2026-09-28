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
@ffi.DefaultAsset('package:ndarray/ndarray')
library;

import 'dart:ffi' as ffi;

@ffi.Native<ffi.Uint8 Function(ffi.Pointer<ffi.Int32>, ffi.Int)>()
external int v_any_less_than_zero_int32(ffi.Pointer<ffi.Int32> arr, int size);

@ffi.Native<ffi.Uint8 Function(ffi.Pointer<ffi.Int64>, ffi.Int)>()
external int v_any_less_than_zero_int64(ffi.Pointer<ffi.Int64> arr, int size);

@ffi.Native<ffi.Uint8 Function(ffi.Pointer<ffi.Int32>, ffi.Int)>()
external int v_any_equal_to_zero_int32(ffi.Pointer<ffi.Int32> arr, int size);

@ffi.Native<ffi.Uint8 Function(ffi.Pointer<ffi.Int64>, ffi.Int)>()
external int v_any_equal_to_zero_int64(ffi.Pointer<ffi.Int64> arr, int size);

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int>,
    ffi.Int32,
    ffi.Pointer<ffi.Void>,
    ffi.Int32,
    ffi.Pointer<ffi.Int>,
    ffi.Int32,
  )
>()
external void s_cast_generic(
  ffi.Pointer<ffi.Void> src_ptr,
  ffi.Pointer<ffi.Int> stridesSrc,
  int dtypeSrc,
  ffi.Pointer<ffi.Void> dest_ptr,
  int dtypeDst,
  ffi.Pointer<ffi.Int> shape,
  int rank,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Int,
    ffi.Int,
    ffi.Int32,
  )
>()
external void v_extract_upper_triangular(
  ffi.Pointer<ffi.Void> src_ptr,
  ffi.Pointer<ffi.Void> dest_ptr,
  int k,
  int n,
  int dtype,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Int, ffi.Int32)>()
external void v_zero_upper_triangular(
  ffi.Pointer<ffi.Void> ptr,
  int n,
  int dtype,
);
