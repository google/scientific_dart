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

#ifndef NDARRAY_NPZ_IO_H
#define NDARRAY_NPZ_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NDARRAY_EXPORT
#if defined(_WIN32)
#define NDARRAY_EXPORT __declspec(dllexport)
#else
#define NDARRAY_EXPORT __attribute__((visibility("default")))
#endif
#endif

/**
 * Saves multiple named arrays into a ZIP archive (.npz) file.
 *
 * @param filepath Destination file path.
 * @param num_arrays Number of array entries.
 * @param entry_names Array of entry filenames (e.g. "arr_a.npy").
 * @param header_bytes Array of pointers to .npy header byte buffers (including 10-byte prefix).
 * @param header_lens Array of header buffer lengths in bytes.
 * @param data_ptrs Array of pointers to contiguous array data.
 * @param data_lens Array of array data lengths in bytes.
 * @param compress_level Compression level: 0 for uncompressed (STORED), 1-9 for Deflate.
 * @return 0 on success, negative error code on failure.
 */
NDARRAY_EXPORT int npz_save(
    const char* filepath,
    size_t num_arrays,
    const char** entry_names,
    const uint8_t** header_bytes,
    const size_t* header_lens,
    const void** data_ptrs,
    const size_t* data_lens,
    int compress_level);

/**
 * Opens a .npz ZIP archive for reading.
 *
 * @param filepath Path to the .npz archive.
 * @param out_num_entries Output pointer receiving the total number of entries in the archive.
 * @return Opaque reader handle on success, NULL on failure.
 */
NDARRAY_EXPORT void* npz_open_reader(const char* filepath, int64_t* out_num_entries);

/**
 * Retrieves entry metadata and the .npy header bytes for a given index in the archive.
 *
 * @param handle Opaque reader handle returned by npz_open_reader.
 * @param index 0-based entry index in the archive.
 * @param name_buf Output buffer for entry filename (null-terminated).
 * @param name_buf_len Size of name_buf in bytes.
 * @param header_buf Output buffer for the .npy header bytes (including 10-byte prefix).
 * @param header_buf_len Size of header_buf in bytes.
 * @param out_header_len Output pointer receiving the exact header length in bytes.
 * @param out_data_len Output pointer receiving the uncompressed array data size in bytes.
 * @return 0 on success, negative error code if entry is invalid, a directory, or not .npy.
 */
NDARRAY_EXPORT int npz_reader_get_entry_info(
    void* handle,
    size_t index,
    char* name_buf,
    size_t name_buf_len,
    uint8_t* header_buf,
    size_t header_buf_len,
    size_t* out_header_len,
    size_t* out_data_len);

/**
 * Extracts raw array data directly into a destination pointer without intermediate allocations.
 *
 * @param handle Opaque reader handle returned by npz_open_reader.
 * @param index 0-based entry index in the archive.
 * @param header_len The header length in bytes (to skip).
 * @param dest_ptr Destination pointer on native C heap.
 * @param data_len Exact number of bytes to read into dest_ptr.
 * @return 0 on success, negative error code on failure.
 */
NDARRAY_EXPORT int npz_reader_extract_data(
    void* handle,
    size_t index,
    size_t header_len,
    void* dest_ptr,
    size_t dest_capacity,
    size_t data_len);

/**
 * Closes the .npz reader handle and releases all associated resources.
 *
 * @param handle Opaque reader handle.
 */
NDARRAY_EXPORT void npz_close_reader(void* handle);

/**
 * Returns the `errno` recorded by the most recent failed `native_file_*` or
 * `npz_*` file operation on the calling thread, and copies the corresponding
 * `strerror` text (NUL-terminated, truncated to [capacity] bytes) into
 * [out_message] when it is non-NULL and [capacity] > 0.
 *
 * All paths passed to the functions below are UTF-8; on Windows they are
 * converted to UTF-16 and opened with the wide-character CRT functions.
 */
NDARRAY_EXPORT int native_file_last_error(uint8_t* out_message, int64_t capacity);

/**
 * Writes [header_len] bytes from [header] followed by [data_len] bytes from
 * [data] into [filepath] (creating parent directories if needed).
 *
 * @return 0 on success, -1 if the arguments are invalid or the file cannot be
 *   opened, -2 if writing or closing fails. The error is recorded for
 *   `native_file_last_error`.
 */
NDARRAY_EXPORT int native_file_write_all(
    const char* filepath,
    const uint8_t* header,
    int64_t header_len,
    const void* data,
    int64_t data_len);

/**
 * Opens the regular file at [filepath] for reading and returns an opaque
 * handle to be released with `native_file_close`, or NULL (with the error
 * recorded for `native_file_last_error`) if the file does not exist, is a
 * directory, or cannot be opened.
 */
NDARRAY_EXPORT void* native_file_open_read(const char* filepath);

/**
 * Returns the size in bytes of the file behind [handle], or -1 on failure.
 * The read position of the handle is preserved.
 */
NDARRAY_EXPORT int64_t native_file_handle_size(void* handle);

/**
 * Reads up to [len] bytes starting at byte [offset] of the file behind
 * [handle] into [out_data], storing the actual number of bytes read into
 * [out_read]. A short read without an error indicates end of file.
 *
 * @return 0 on success (including short reads at EOF), -1 if the arguments
 *   are invalid, -2 if seeking or reading fails.
 */
NDARRAY_EXPORT int native_file_handle_read(
    void* handle,
    int64_t offset,
    int64_t len,
    void* out_data,
    int64_t* out_read);

/**
 * Closes a handle returned by `native_file_open_read`. NULL is ignored.
 */
NDARRAY_EXPORT void native_file_close(void* handle);

#ifdef __cplusplus
}
#endif

#endif // NDARRAY_NPZ_IO_H
