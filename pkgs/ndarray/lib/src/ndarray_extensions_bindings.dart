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

/// Returns the `errno` of the most recent failed native file operation on the
/// calling thread and copies its `strerror` text (NUL-terminated, at most
/// [capacity] bytes) into [out_message].
@ffi.Native<ffi.Int Function(ffi.Pointer<ffi.Uint8>, ffi.Int64)>()
external int native_file_last_error(
  ffi.Pointer<ffi.Uint8> out_message,
  int capacity,
);

/// Writes [header_len] header bytes and [data_len] payload bytes to [filepath],
/// creating missing parent directories.
///
/// Returns 0 on success, -1 if the file cannot be opened, -2 if writing fails.
@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Int64,
    ffi.Pointer<ffi.Void>,
    ffi.Int64,
  )
>()
external int native_file_write_all(
  ffi.Pointer<ffi.Char> filepath,
  ffi.Pointer<ffi.Uint8> header,
  int header_len,
  ffi.Pointer<ffi.Void> data,
  int data_len,
);

/// Opens the regular file at [filepath] for reading.
///
/// Returns a handle for [native_file_handle_size], [native_file_handle_read]
/// and [native_file_close], or `nullptr` on failure.
@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Char>)>()
external ffi.Pointer<ffi.Void> native_file_open_read(
  ffi.Pointer<ffi.Char> filepath,
);

/// Returns the size in bytes of the open file [handle], or -1 on failure.
@ffi.Native<ffi.Int64 Function(ffi.Pointer<ffi.Void>)>()
external int native_file_handle_size(ffi.Pointer<ffi.Void> handle);

/// Reads up to [len] bytes at [offset] from the open file [handle] into
/// [out_data], storing the number of bytes actually read into [out_read].
///
/// Returns 0 on success (a short read means end of file), -1 for invalid
/// arguments, -2 if seeking or reading fails.
@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Void>,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int64>,
  )
>()
external int native_file_handle_read(
  ffi.Pointer<ffi.Void> handle,
  int offset,
  int len,
  ffi.Pointer<ffi.Void> out_data,
  ffi.Pointer<ffi.Int64> out_read,
);

/// Closes a handle returned by [native_file_open_read].
@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void native_file_close(ffi.Pointer<ffi.Void> handle);

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
