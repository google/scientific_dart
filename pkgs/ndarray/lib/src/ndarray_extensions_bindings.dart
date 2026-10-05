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

import 'ndarray_bindings.dart' show cpx_t, cpx_f_t;

/// NPZ native zip archive serialization
@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Char>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Pointer<ffi.Uint8>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Pointer<ffi.Void>>,
    ffi.Pointer<ffi.Size>,
    ffi.Int,
  )
>()
external int npz_save(
  ffi.Pointer<ffi.Char> filepath,
  int num_arrays,
  ffi.Pointer<ffi.Pointer<ffi.Char>> entry_names,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> header_bytes,
  ffi.Pointer<ffi.Size> header_lens,
  ffi.Pointer<ffi.Pointer<ffi.Void>> data_ptrs,
  ffi.Pointer<ffi.Size> data_lens,
  int compress_level,
);

/// NPZ native zip archive reader open
@ffi.Native<
  ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Int64>)
>()
external ffi.Pointer<ffi.Void> npz_open_reader(
  ffi.Pointer<ffi.Char> filepath,
  ffi.Pointer<ffi.Int64> out_num_entries,
);

/// NPZ native zip archive entry info reader
@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Void>,
    ffi.Size,
    ffi.Pointer<ffi.Char>,
    ffi.Size,
    ffi.Pointer<ffi.Uint8>,
    ffi.Size,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int npz_reader_get_entry_info(
  ffi.Pointer<ffi.Void> handle,
  int index,
  ffi.Pointer<ffi.Char> name_buf,
  int name_buf_len,
  ffi.Pointer<ffi.Uint8> header_buf,
  int header_buf_len,
  ffi.Pointer<ffi.Size> out_header_len,
  ffi.Pointer<ffi.Size> out_data_len,
);

/// NPZ native zip archive entry data extractor (zero-copy into native buffer)
@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Void>,
    ffi.Size,
    ffi.Size,
    ffi.Pointer<ffi.Void>,
    ffi.Size,
    ffi.Size,
  )
>()
external int npz_reader_extract_data(
  ffi.Pointer<ffi.Void> handle,
  int index,
  int header_len,
  ffi.Pointer<ffi.Void> dest_ptr,
  int dest_capacity,
  int data_len,
);

/// NPZ native zip archive reader close
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void npz_close_reader(ffi.Pointer<ffi.Void> handle);

/// Custom Indexing: take_along_axis
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_take_along_axis(
  int dtype,
  int indexDtype,
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int64> arrShape,
  ffi.Pointer<ffi.Int64> arrStrides,
  ffi.Pointer<ffi.Void> indices,
  ffi.Pointer<ffi.Int64> idxShape,
  ffi.Pointer<ffi.Int64> idxStrides,
  ffi.Pointer<ffi.Void> dest,
  ffi.Pointer<ffi.Int64> outShape,
  ffi.Pointer<ffi.Int64> outStrides,
  int rank,
  int axis,
  ffi.Pointer<ffi.Int64> outErrorIdx,
);

/// Custom Indexing: put_along_axis
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_put_along_axis(
  int dtype,
  int indexDtype,
  ffi.Pointer<ffi.Void> target,
  ffi.Pointer<ffi.Int64> targetShape,
  ffi.Pointer<ffi.Int64> targetStrides,
  ffi.Pointer<ffi.Void> indices,
  ffi.Pointer<ffi.Int64> idxShape,
  ffi.Pointer<ffi.Int64> idxStrides,
  ffi.Pointer<ffi.Void> values,
  ffi.Pointer<ffi.Int64> valShape,
  ffi.Pointer<ffi.Int64> valStrides,
  int rank,
  int axis,
  ffi.Pointer<ffi.Int64> outErrorIdx,
);

/// Custom Indexing / Manipulation: tile contiguous
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
  )
>()
external int native_tile_contiguous(
  int dtype,
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int64> srcShape,
  ffi.Pointer<ffi.Int64> reps,
  ffi.Pointer<ffi.Void> dest,
  ffi.Pointer<ffi.Int64> outShape,
  int rank,
);

/// Custom Indexing / Manipulation: tile strided
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
  )
>()
external int native_tile_strided(
  int dtype,
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int64> srcShape,
  ffi.Pointer<ffi.Int64> srcStrides,
  ffi.Pointer<ffi.Int64> reps,
  ffi.Pointer<ffi.Void> dest,
  ffi.Pointer<ffi.Int64> outShape,
  ffi.Pointer<ffi.Int64> outStrides,
  int rank,
);

/// Custom Indexing / Manipulation: roll 1D
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Void>,
  )
>()
external int native_roll_1d(
  int dtype,
  ffi.Pointer<ffi.Void> src,
  int size,
  int shift,
  ffi.Pointer<ffi.Void> dest,
);

/// Custom Indexing / Manipulation: roll ND
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_roll_nd(
  int dtype,
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int64> shape,
  ffi.Pointer<ffi.Int64> srcStrides,
  int rank,
  int shift,
  int axis,
  ffi.Pointer<ffi.Void> dest,
  ffi.Pointer<ffi.Int64> destStrides,
);

/// Custom Padding: 2D
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Void>,
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Int,
  )
>()
external int native_pad_2d(
  int dtype,
  ffi.Pointer<ffi.Void> src,
  int srcRows,
  int srcCols,
  int srcStrideRows,
  int srcStrideCols,
  ffi.Pointer<ffi.Void> dest,
  int padTop,
  int padBottom,
  int padLeft,
  int padRight,
  int mode,
  ffi.Pointer<ffi.Void> constBefore,
  ffi.Pointer<ffi.Void> constAfter,
  int isUniformConstant,
);

/// Unravels flat indices into multi-dimensional coordinate arrays.
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
    ffi.Int,
    ffi.Pointer<ffi.Pointer<ffi.Int64>>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_unravel_index(
  int indexDtype,
  ffi.Pointer<ffi.Void> indicesPtr,
  ffi.Pointer<ffi.Int64> indicesShape,
  ffi.Pointer<ffi.Int64> indicesStrides,
  int indicesRank,
  int indicesSize,
  ffi.Pointer<ffi.Int64> dims,
  int ndims,
  int order,
  ffi.Pointer<ffi.Pointer<ffi.Int64>> outPtrs,
  ffi.Pointer<ffi.Int64> outStridesFlat,
  int isContiguous,
  ffi.Pointer<ffi.Int64> outErrorIdx,
);

/// Converts multi-dimensional coordinate arrays into flat indices.
@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Pointer<ffi.Int64>>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int>,
    ffi.Int64,
    ffi.Int,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_ravel_multi_index(
  ffi.Pointer<ffi.Pointer<ffi.Int64>> coordsPtrs,
  ffi.Pointer<ffi.Int64> coordsStridesFlat,
  ffi.Pointer<ffi.Int64> targetShape,
  int targetRank,
  int totalSize,
  ffi.Pointer<ffi.Int64> dims,
  ffi.Pointer<ffi.Int> modes,
  int ndims,
  int order,
  ffi.Pointer<ffi.Int64> outPtr,
  ffi.Pointer<ffi.Int64> outStrides,
  int isContiguous,
  ffi.Pointer<ffi.Int64> outErrorVal,
);

/// Fills a dense grid of 64-bit indices for the given dimensions.
@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_indices_int64(
  ffi.Pointer<ffi.Int64> dims,
  int ndims,
  int sliceSize,
  ffi.Pointer<ffi.Int64> outPtr,
);

/// Fills lower-triangle 64-bit row and column index arrays.
@ffi.Native<
  ffi.Int Function(
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_tril_indices(
  int n,
  int m,
  int k,
  ffi.Pointer<ffi.Int64> outRow,
  ffi.Pointer<ffi.Int64> outCol,
);

/// Fills upper-triangle 64-bit row and column index arrays.
@ffi.Native<
  ffi.Int Function(
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_triu_indices(
  int n,
  int m,
  int k,
  ffi.Pointer<ffi.Int64> outRow,
  ffi.Pointer<ffi.Int64> outCol,
);

/// Custom Padding: ND
@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int64,
    ffi.Int,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Int,
  )
>()
external int native_pad_nd(
  int dtype,
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int64> srcShape,
  ffi.Pointer<ffi.Int64> srcStrides,
  ffi.Pointer<ffi.Void> dest,
  ffi.Pointer<ffi.Int64> destShape,
  ffi.Pointer<ffi.Int64> destStrides,
  ffi.Pointer<ffi.Int64> padBefore,
  ffi.Pointer<ffi.Int64> padAfter,
  int rank,
  int mode,
  ffi.Pointer<ffi.Void> constBefore,
  ffi.Pointer<ffi.Void> constAfter,
  int isUniformConstant,
);

/// Nanmedian contiguous reduction for Float64.
@ffi.Native<ffi.Double Function(ffi.Pointer<ffi.Double>, ffi.Int64)>()
external double r_nanmedian_double(ffi.Pointer<ffi.Double> src, int size);

/// Nanmedian contiguous reduction for Float32.
@ffi.Native<ffi.Float Function(ffi.Pointer<ffi.Float>, ffi.Int64)>()
external double r_nanmedian_float(ffi.Pointer<ffi.Float> src, int size);

/// Nanmedian contiguous reduction for Complex128.
@ffi.Native<cpx_t Function(ffi.Pointer<cpx_t>, ffi.Int64)>()
external cpx_t r_nanmedian_complex128(ffi.Pointer<cpx_t> src, int size);

/// Nanmedian contiguous reduction for Complex64.
@ffi.Native<cpx_f_t Function(ffi.Pointer<cpx_f_t>, ffi.Int64)>()
external cpx_f_t r_nanmedian_complex64(ffi.Pointer<cpx_f_t> src, int size);

/// Nanmedian strided axis reduction for Float64.
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int,
    ffi.Int,
  )
>()
external void s_nanmedian_double(
  ffi.Pointer<ffi.Double> src,
  ffi.Pointer<ffi.Int64> stridesSrc,
  ffi.Pointer<ffi.Double> dest,
  ffi.Pointer<ffi.Int64> stridesDest,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  int axis,
);

/// Nanmedian strided axis reduction for Float32.
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int,
    ffi.Int,
  )
>()
external void s_nanmedian_float(
  ffi.Pointer<ffi.Float> src,
  ffi.Pointer<ffi.Int64> stridesSrc,
  ffi.Pointer<ffi.Float> dest,
  ffi.Pointer<ffi.Int64> stridesDest,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  int axis,
);

/// Nanmedian strided axis reduction for Complex128.
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<cpx_t>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<cpx_t>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int,
    ffi.Int,
  )
>()
external void s_nanmedian_complex128(
  ffi.Pointer<cpx_t> src,
  ffi.Pointer<ffi.Int64> stridesSrc,
  ffi.Pointer<cpx_t> dest,
  ffi.Pointer<ffi.Int64> stridesDest,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  int axis,
);

/// Nanmedian strided axis reduction for Complex64.
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<cpx_f_t>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<cpx_f_t>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int,
    ffi.Int,
  )
>()
external void s_nanmedian_complex64(
  ffi.Pointer<cpx_f_t> src,
  ffi.Pointer<ffi.Int64> stridesSrc,
  ffi.Pointer<cpx_f_t> dest,
  ffi.Pointer<ffi.Int64> stridesDest,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  int axis,
);

/// Nanquantile contiguous reduction for Float64.
@ffi.Native<
  ffi.Double Function(ffi.Pointer<ffi.Double>, ffi.Int64, ffi.Double, ffi.Int)
>()
external double r_nanquantile_double(
  ffi.Pointer<ffi.Double> src,
  int size,
  double q,
  int method,
);

/// Nanquantile contiguous reduction for Float32.
@ffi.Native<
  ffi.Double Function(ffi.Pointer<ffi.Float>, ffi.Int64, ffi.Double, ffi.Int)
>()
external double r_nanquantile_float(
  ffi.Pointer<ffi.Float> src,
  int size,
  double q,
  int method,
);

/// Nanquantile strided axis reduction for Float64.
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int,
    ffi.Int,
    ffi.Double,
    ffi.Int,
  )
>()
external void s_nanquantile_double(
  ffi.Pointer<ffi.Double> src,
  ffi.Pointer<ffi.Int64> stridesSrc,
  ffi.Pointer<ffi.Double> dest,
  ffi.Pointer<ffi.Int64> stridesDest,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  int axis,
  double q,
  int method,
);

/// Nanquantile strided axis reduction for Float32.
@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.Int,
    ffi.Int,
    ffi.Double,
    ffi.Int,
  )
>()
external void s_nanquantile_float(
  ffi.Pointer<ffi.Float> src,
  ffi.Pointer<ffi.Int64> stridesSrc,
  ffi.Pointer<ffi.Float> dest,
  ffi.Pointer<ffi.Int64> stridesDest,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  int axis,
  double q,
  int method,
);
