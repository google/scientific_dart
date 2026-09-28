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
