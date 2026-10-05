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

#include "npz_io.h"
#include "ndarray_common.h"
#include "third_party/miniz/miniz.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <atomic>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <direct.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

static int npz_fseek64(FILE* fp, int64_t offset, int origin) {
#if defined(_WIN32)
  return _fseeki64(fp, offset, origin);
#else
  return fseeko(fp, static_cast<off_t>(offset), origin);
#endif
}

static int64_t npz_ftell64(FILE* fp) {
#if defined(_WIN32)
  return _ftelli64(fp);
#else
  return static_cast<int64_t>(ftello(fp));
#endif
}

// `errno` of the most recent failed file operation in this thread, exposed to
// Dart via `native_file_last_error` so I/O failures carry the OS message.
static thread_local int g_native_file_errno = 0;

static void npz_record_errno(void) {
    g_native_file_errno = errno;
}

#if defined(_WIN32)
// Converts a UTF-8 path to a malloc'd UTF-16 string, or returns NULL (with
// `errno` set) on failure. Dart strings arrive as UTF-8; the narrow CRT
// functions interpret bytes in the ANSI code page and corrupt non-ASCII paths.
static wchar_t* npz_utf8_to_wide(const char* utf8) {
    int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, NULL, 0);
    if (needed <= 0) {
        errno = EINVAL;
        return NULL;
    }
    wchar_t* wide = (wchar_t*)malloc((size_t)needed * sizeof(wchar_t));
    if (!wide) {
        errno = ENOMEM;
        return NULL;
    }
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, wide, needed) <= 0) {
        free(wide);
        errno = EINVAL;
        return NULL;
    }
    return wide;
}
#endif

// `fopen` for a UTF-8 encoded path. Records `errno` on failure.
static FILE* npz_fopen_utf8(const char* filepath, const char* mode) {
    FILE* fp = NULL;
#if defined(_WIN32)
    wchar_t* wide_path = npz_utf8_to_wide(filepath);
    if (!wide_path) {
        npz_record_errno();
        return NULL;
    }
    wchar_t wide_mode[8];
    size_t mode_len = strlen(mode);
    if (mode_len >= sizeof(wide_mode) / sizeof(wide_mode[0])) mode_len = 7;
    for (size_t i = 0; i < mode_len; i++) wide_mode[i] = (wchar_t)mode[i];
    wide_mode[mode_len] = L'\0';
    fp = _wfopen(wide_path, wide_mode);
    free(wide_path);
#else
    fp = fopen(filepath, mode);
#endif
    if (!fp) npz_record_errno();
    return fp;
}

// `mkdir` for a UTF-8 encoded path; failures (including EEXIST) are ignored
// by the caller, which only needs the final `fopen` to succeed or fail.
static void npz_mkdir_utf8(const char* path) {
#if defined(_WIN32)
    wchar_t* wide_path = npz_utf8_to_wide(path);
    if (!wide_path) return;
    _wmkdir(wide_path);
    free(wide_path);
#else
    mkdir(path, 0777);
#endif
}

// Creates every missing directory on the path to `filepath` (like `mkdir -p`
// on the parent). Best effort: any failure surfaces later as an `fopen` error
// with a meaningful `errno`.
static void npz_ensure_parent_dirs(const char* filepath) {
    if (!filepath) return;
    size_t len = strlen(filepath);
    if (len == 0) return;
    char* buf = (char*)malloc(len + 1);
    if (!buf) return;
    memcpy(buf, filepath, len + 1);
    for (size_t i = 1; i < len; i++) {
        if (buf[i] == '/' || buf[i] == '\\') {
            char saved = buf[i];
            buf[i] = '\0';
            npz_mkdir_utf8(buf);
            buf[i] = saved;
        }
    }
    free(buf);
}

static inline void write_u16_le(uint8_t* p, uint16_t val) {
    p[0] = (uint8_t)(val & 0xFF);
    p[1] = (uint8_t)((val >> 8) & 0xFF);
}

static inline void write_u32_le(uint8_t* p, uint32_t val) {
    p[0] = (uint8_t)(val & 0xFF);
    p[1] = (uint8_t)((val >> 8) & 0xFF);
    p[2] = (uint8_t)((val >> 16) & 0xFF);
    p[3] = (uint8_t)((val >> 24) & 0xFF);
}

static inline void write_u64_le(uint8_t* p, uint64_t val) {
    write_u32_le(p, (uint32_t)(val & 0xFFFFFFFFULL));
    write_u32_le(p + 4, (uint32_t)(val >> 32));
}

static inline uint16_t read_u16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t read_u64_le(const uint8_t* p) {
    return (uint64_t)read_u32_le(p) | ((uint64_t)read_u32_le(p + 4) << 32);
}

// ---------------------------------------------------------------------------
// Slicing-by-16 Fast IEEE 802.3 CRC-32
// ---------------------------------------------------------------------------
static uint32_t s_crc32_table[16][256];
static std::atomic<int> s_crc32_init_state{0};

static void init_crc32_tables(void) noexcept {
    int state = s_crc32_init_state.load(std::memory_order_acquire);
    if (state == 2) return;
    int expected = 0;
    if (s_crc32_init_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++) {
                c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
            }
            s_crc32_table[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; i++) {
            for (int j = 1; j < 16; j++) {
                s_crc32_table[j][i] = s_crc32_table[0][s_crc32_table[j - 1][i] & 0xFF] ^ (s_crc32_table[j - 1][i] >> 8);
            }
        }
        s_crc32_init_state.store(2, std::memory_order_release);
    } else {
        while (s_crc32_init_state.load(std::memory_order_acquire) != 2) {}
    }
}

static uint32_t npz_fast_crc32(uint32_t initial_crc, const void* buf, size_t len) {
    init_crc32_tables();
    if (!buf || len == 0) return initial_crc;

    uint32_t crc = initial_crc ^ 0xFFFFFFFF;
    const uint8_t* p = (const uint8_t*)buf;

    while (len && ((uintptr_t)p & 15)) {
        crc = s_crc32_table[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
        len--;
    }

    while (len >= 16) {
        uint32_t one, two, three, four;
        memcpy(&one, p, sizeof(uint32_t));
        memcpy(&two, p + 4, sizeof(uint32_t));
        memcpy(&three, p + 8, sizeof(uint32_t));
        memcpy(&four, p + 12, sizeof(uint32_t));
        one ^= crc;
        crc = s_crc32_table[15][one & 0xFF] ^
              s_crc32_table[14][(one >> 8) & 0xFF] ^
              s_crc32_table[13][(one >> 16) & 0xFF] ^
              s_crc32_table[12][(one >> 24) & 0xFF] ^
              s_crc32_table[11][two & 0xFF] ^
              s_crc32_table[10][(two >> 8) & 0xFF] ^
              s_crc32_table[9][(two >> 16) & 0xFF] ^
              s_crc32_table[8][(two >> 24) & 0xFF] ^
              s_crc32_table[7][three & 0xFF] ^
              s_crc32_table[6][(three >> 8) & 0xFF] ^
              s_crc32_table[5][(three >> 16) & 0xFF] ^
              s_crc32_table[4][(three >> 24) & 0xFF] ^
              s_crc32_table[3][four & 0xFF] ^
              s_crc32_table[2][(four >> 8) & 0xFF] ^
              s_crc32_table[1][(four >> 16) & 0xFF] ^
              s_crc32_table[0][(four >> 24) & 0xFF];
        p += 16;
        len -= 16;
    }

    while (len > 0) {
        crc = s_crc32_table[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
        len--;
    }

    return crc ^ 0xFFFFFFFF;
}

// ---------------------------------------------------------------------------
// Memory Allocation Hooks for Miniz Fallback
// ---------------------------------------------------------------------------
static void* npz_alloc(void* opaque, size_t items, size_t size) {
    if (items != 0 && size > SIZE_MAX / items) {
        ndarray_set_oom_flag();
        return NULL;
    }
    void* ptr = malloc(items * size);
    if (!ptr) {
        ndarray_set_oom_flag();
    }
    return ptr;
}
static void npz_free(void* opaque, void* address) {
    free(address);
}
static void* npz_realloc(void* opaque, void* address, size_t items, size_t size) {
    if (items != 0 && size > SIZE_MAX / items) {
        ndarray_set_oom_flag();
        return NULL;
    }
    void* ptr = realloc(address, items * size);
    if (!ptr) {
        ndarray_set_oom_flag();
    }
    return ptr;
}

// ---------------------------------------------------------------------------
// npz_save STORED (Uncompressed) Implementation (ZIP32 + ZIP64)
// ---------------------------------------------------------------------------
struct ZipEntryMeta {
    uint64_t offset;
    uint64_t uncomp_size;
    uint32_t crc32;
    uint16_t name_len;
    bool is_zip64;
    const char* name;
};

static int npz_save_stored(
    const char* filepath,
    size_t num_arrays,
    const char** entry_names,
    const uint8_t** header_bytes,
    const size_t* header_lens,
    const void** data_ptrs,
    const size_t* data_lens,
    bool force_zip64) {
    FILE* fp = npz_fopen_utf8(filepath, "wb");
    if (!fp) return -2;

    NoThrowBuffer<ZipEntryMeta> meta(num_arrays);
    if (!meta.ok()) {
        fclose(fp);
        return -6;
    }

    uint64_t current_offset = 0;
    bool any_entry_zip64 = false;

    // Buffer for assembling local file header + filename + extra + npy header
    uint8_t lfh_buf[1024];

    for (size_t i = 0; i < num_arrays; i++) {
        size_t nlen = strlen(entry_names[i]);
        if (nlen > 0xFFFFULL) {
            fclose(fp);
            return -7;
        }
        size_t hlen = header_lens[i];
        size_t dlen = data_lens[i];
        uint64_t uncomp_sz = (uint64_t)hlen + (uint64_t)dlen;
        bool entry_zip64 =
            force_zip64 ||
            (uncomp_sz >= 0xFFFFFFFFULL) ||
            (current_offset >= 0xFFFFFFFFULL);
        if (entry_zip64) any_entry_zip64 = true;
        uint16_t lfh_extra_len = entry_zip64 ? 20 : 0;

        uint32_t crc = npz_fast_crc32(0, header_bytes[i], hlen);
        crc = npz_fast_crc32(crc, data_ptrs[i], dlen);

        meta[i].offset = current_offset;
        meta[i].crc32 = crc;
        meta[i].uncomp_size = uncomp_sz;
        meta[i].name_len = (uint16_t)nlen;
        meta[i].is_zip64 = entry_zip64;
        meta[i].name = entry_names[i];

        size_t prefix_len = 30 + nlen + lfh_extra_len + hlen;
        uint8_t* p_buf = lfh_buf;
        NoThrowBuffer<uint8_t> p_heap;
        if (prefix_len > sizeof(lfh_buf)) {
            if (!p_heap.resize(prefix_len)) {
                fclose(fp);
                return -6;
            }
            p_buf = p_heap.data();
        }

        write_u32_le(p_buf + 0, 0x04034b50);
        write_u16_le(p_buf + 4, entry_zip64 ? 45 : 20);
        write_u16_le(p_buf + 6, 0);
        write_u16_le(p_buf + 8, 0);
        write_u16_le(p_buf + 10, 0);
        write_u16_le(p_buf + 12, 0);
        write_u32_le(p_buf + 14, crc);
        write_u32_le(p_buf + 18, entry_zip64 ? 0xFFFFFFFFU : (uint32_t)uncomp_sz);
        write_u32_le(p_buf + 22, entry_zip64 ? 0xFFFFFFFFU : (uint32_t)uncomp_sz);
        write_u16_le(p_buf + 26, (uint16_t)nlen);
        write_u16_le(p_buf + 28, lfh_extra_len);
        memcpy(p_buf + 30, entry_names[i], nlen);
        if (entry_zip64) {
            uint8_t* extra = p_buf + 30 + nlen;
            write_u16_le(extra + 0, 0x0001);
            write_u16_le(extra + 2, 16);
            write_u64_le(extra + 4, uncomp_sz);
            write_u64_le(extra + 12, uncomp_sz);
        }
        memcpy(p_buf + 30 + nlen + lfh_extra_len, header_bytes[i], hlen);

        size_t written_prefix = fwrite(p_buf, 1, prefix_len, fp);

        if (written_prefix != prefix_len ||
            (dlen > 0 && fwrite(data_ptrs[i], 1, dlen, fp) != dlen)) {
            fclose(fp);
            return -3;
        }

        current_offset += (uint64_t)prefix_len + (uint64_t)dlen;
    }

    uint64_t cd_offset = current_offset;
    uint64_t cd_size = 0;
    for (size_t i = 0; i < num_arrays; i++) {
        uint16_t cdh_extra_len = meta[i].is_zip64 ? 28 : 0;
        cd_size += 46 + meta[i].name_len + cdh_extra_len;
    }

    bool archive_zip64 =
        force_zip64 ||
        any_entry_zip64 ||
        (num_arrays >= 0xFFFFULL) ||
        (cd_size >= 0xFFFFFFFFULL) ||
        (cd_offset >= 0xFFFFFFFFULL);

    size_t total_tail_size = (size_t)cd_size + (archive_zip64 ? (56 + 20) : 0) + 22;
    NoThrowBuffer<uint8_t> tail_buf(total_tail_size);
    if (!tail_buf.ok()) {
        fclose(fp);
        return -6;
    }

    uint8_t* p_cd = tail_buf.data();
    for (size_t i = 0; i < num_arrays; i++) {
        bool ez64 = meta[i].is_zip64;
        uint16_t cdh_extra_len = ez64 ? 28 : 0;
        write_u32_le(p_cd + 0, 0x02014b50);
        write_u16_le(p_cd + 4, ez64 ? 45 : 20);
        write_u16_le(p_cd + 6, ez64 ? 45 : 20);
        write_u16_le(p_cd + 8, 0);
        write_u16_le(p_cd + 10, 0);
        write_u16_le(p_cd + 12, 0);
        write_u16_le(p_cd + 14, 0);
        write_u32_le(p_cd + 16, meta[i].crc32);
        write_u32_le(p_cd + 20, ez64 ? 0xFFFFFFFFU : (uint32_t)meta[i].uncomp_size);
        write_u32_le(p_cd + 24, ez64 ? 0xFFFFFFFFU : (uint32_t)meta[i].uncomp_size);
        write_u16_le(p_cd + 28, meta[i].name_len);
        write_u16_le(p_cd + 30, cdh_extra_len);
        write_u16_le(p_cd + 32, 0);
        write_u16_le(p_cd + 34, 0);
        write_u16_le(p_cd + 36, 0);
        write_u32_le(p_cd + 38, 0);
        write_u32_le(p_cd + 42, ez64 ? 0xFFFFFFFFU : (uint32_t)meta[i].offset);
        memcpy(p_cd + 46, meta[i].name, meta[i].name_len);
        if (ez64) {
            uint8_t* extra = p_cd + 46 + meta[i].name_len;
            write_u16_le(extra + 0, 0x0001);
            write_u16_le(extra + 2, 24);
            write_u64_le(extra + 4, meta[i].uncomp_size);
            write_u64_le(extra + 12, meta[i].uncomp_size);
            write_u64_le(extra + 20, meta[i].offset);
        }
        p_cd += 46 + meta[i].name_len + cdh_extra_len;
    }

    if (archive_zip64) {
        uint64_t zip64_eocd_offset = cd_offset + cd_size;
        // ZIP64 End of Central Directory Record (56 bytes)
        write_u32_le(p_cd + 0, 0x06064b50);
        write_u64_le(p_cd + 4, 44);
        write_u16_le(p_cd + 12, 45);
        write_u16_le(p_cd + 14, 45);
        write_u32_le(p_cd + 16, 0);
        write_u32_le(p_cd + 20, 0);
        write_u64_le(p_cd + 24, (uint64_t)num_arrays);
        write_u64_le(p_cd + 32, (uint64_t)num_arrays);
        write_u64_le(p_cd + 40, cd_size);
        write_u64_le(p_cd + 48, cd_offset);
        p_cd += 56;

        // ZIP64 End of Central Directory Locator (20 bytes)
        write_u32_le(p_cd + 0, 0x07064b50);
        write_u32_le(p_cd + 4, 0);
        write_u64_le(p_cd + 8, zip64_eocd_offset);
        write_u32_le(p_cd + 16, 1);
        p_cd += 20;
    }

    write_u32_le(p_cd + 0, 0x06054b50);
    write_u16_le(p_cd + 4, 0);
    write_u16_le(p_cd + 6, 0);
    write_u16_le(p_cd + 8, (archive_zip64 || num_arrays >= 0xFFFFULL) ? 0xFFFFU : (uint16_t)num_arrays);
    write_u16_le(p_cd + 10, (archive_zip64 || num_arrays >= 0xFFFFULL) ? 0xFFFFU : (uint16_t)num_arrays);
    write_u32_le(p_cd + 12, (archive_zip64 || cd_size >= 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)cd_size);
    write_u32_le(p_cd + 16, (archive_zip64 || cd_offset >= 0xFFFFFFFFULL) ? 0xFFFFFFFFU : (uint32_t)cd_offset);
    write_u16_le(p_cd + 20, 0);

    if (fwrite(tail_buf.data(), 1, total_tail_size, fp) != total_tail_size) {
        fclose(fp);
        return -5;
    }

    if (fclose(fp) != 0) {
        return -5;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// npz_save DEFLATE (Compressed) Fallback Implementation (ZIP32 + ZIP64)
// ---------------------------------------------------------------------------
struct NpzReadEntryState {
    const uint8_t* header;
    size_t header_len;
    const uint8_t* data;
    size_t data_len;
};

static size_t npz_read_entry_callback(void* opaque, mz_uint64 file_ofs, void* pBuf, size_t n) {
    struct NpzReadEntryState* s = (struct NpzReadEntryState*)opaque;
    size_t total_len = s->header_len + s->data_len;
    if (file_ofs >= total_len) return 0;
    if (file_ofs + n > total_len) n = (size_t)(total_len - file_ofs);

    size_t bytes_read = 0;
    if (file_ofs < s->header_len) {
        size_t h_available = s->header_len - (size_t)file_ofs;
        size_t to_copy = (n < h_available) ? n : h_available;
        memcpy(pBuf, s->header + file_ofs, to_copy);
        bytes_read += to_copy;
        file_ofs += to_copy;
        pBuf = (uint8_t*)pBuf + to_copy;
        n -= to_copy;
    }
    if (n > 0 && file_ofs >= s->header_len) {
        size_t d_offset = (size_t)file_ofs - s->header_len;
        memcpy(pBuf, s->data + d_offset, n);
        bytes_read += n;
    }
    return bytes_read;
}

static int npz_save_deflate(
    const char* filepath,
    size_t num_arrays,
    const char** entry_names,
    const uint8_t** header_bytes,
    const size_t* header_lens,
    const void** data_ptrs,
    const size_t* data_lens,
    int compress_level,
    bool force_zip64) {
    bool need_zip64 = force_zip64 || (num_arrays >= 0xFFFFULL);
    uint64_t total_uncomp_estimate = 0;
    for (size_t i = 0; i < num_arrays; i++) {
        size_t nlen = strlen(entry_names[i]);
        if (nlen > 0xFFFFULL) {
            return -7;
        }
        uint64_t entry_len = (uint64_t)header_lens[i] + (uint64_t)data_lens[i];
        if (entry_len >= 0xFFFFFFFFULL) {
            need_zip64 = true;
        }
        total_uncomp_estimate += 30ULL + (uint64_t)nlen + entry_len + 46ULL + (uint64_t)nlen;
        if (total_uncomp_estimate >= 0xFFFFFFFFULL) {
            need_zip64 = true;
        }
    }

    mz_zip_archive zip;
    mz_zip_zero_struct(&zip);
    zip.m_pAlloc = npz_alloc;
    zip.m_pFree = npz_free;
    zip.m_pRealloc = npz_realloc;

    mz_uint init_flags = need_zip64 ? MZ_ZIP_FLAG_WRITE_ZIP64 : 0;
    if (!mz_zip_writer_init_file_v2(&zip, filepath, 0, init_flags)) {
        // miniz returns straight after its failed fopen, so errno is intact.
        npz_record_errno();
        return -2;
    }

    for (size_t i = 0; i < num_arrays; i++) {
        struct NpzReadEntryState state;
        state.header = header_bytes[i];
        state.header_len = header_lens[i];
        state.data = (const uint8_t*)data_ptrs[i];
        state.data_len = data_lens[i];

        size_t total_len = state.header_len + state.data_len;
        mz_uint flags = (compress_level > 0) ? (mz_uint)compress_level : 0;

        mz_bool ok = mz_zip_writer_add_read_buf_callback(
            &zip,
            entry_names[i],
            npz_read_entry_callback,
            &state,
            total_len,
            NULL,
            NULL,
            0,
            flags,
            NULL,
            0,
            NULL,
            0);

        if (!ok) {
            mz_zip_writer_end(&zip);
            return -3;
        }
    }

    if (!mz_zip_writer_finalize_archive(&zip)) {
        mz_zip_writer_end(&zip);
        return -4;
    }

    if (!mz_zip_writer_end(&zip)) {
        return -5;
    }

    return 0;
}

NDARRAY_EXPORT int npz_save(
    const char* filepath,
    size_t num_arrays,
    const char** entry_names,
    const uint8_t** header_bytes,
    const size_t* header_lens,
    const void** data_ptrs,
    const size_t* data_lens,
    int compress_level) {
    if (!filepath || num_arrays == 0 || !entry_names || !header_bytes || !header_lens || !data_ptrs || !data_lens) {
        return -1;
    }
    npz_ensure_parent_dirs(filepath);

    bool force_zip64 = (compress_level & 0x100) != 0;
    int level = compress_level & 0xFF;

    if (level == 0) {
        return npz_save_stored(
            filepath,
            num_arrays,
            entry_names,
            header_bytes,
            header_lens,
            data_ptrs,
            data_lens,
            force_zip64);
    } else {
        return npz_save_deflate(
            filepath,
            num_arrays,
            entry_names,
            header_bytes,
            header_lens,
            data_ptrs,
            data_lens,
            level,
            force_zip64);
    }
}

// ---------------------------------------------------------------------------
// NpzReader Definition & Implementation
// ---------------------------------------------------------------------------
struct NpzEntryInfo {
    char name[512];
    uint16_t comp_method;
    uint32_t crc32;
    uint64_t comp_size;
    uint64_t uncomp_size;
    uint64_t local_header_offset;
    uint64_t data_offset;
    uint64_t header_len;
    bool is_directory;
    bool is_npy;
    int header_status;
};

struct NpzReader {
    FILE* fp;
#if defined(_WIN32)
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif
    const uint8_t* mmap_data;
    size_t file_size;
    size_t num_files;
    NpzEntryInfo* entries;

    mz_zip_archive zip;
    bool zip_initialized;
};

static size_t find_eocd(const uint8_t* data, size_t file_size) {
    if (file_size < 22) return (size_t)-1;
    size_t search_len = (file_size < 65557) ? file_size : 65557;
    size_t search_start = file_size - search_len;

    for (size_t i = file_size - 22;; i--) {
        if (read_u32_le(data + i) == 0x06054b50) {
            uint16_t comment_len = read_u16_le(data + i + 20);
            if (i + 22 + comment_len <= file_size) {
                return i;
            }
        }
        if (i <= search_start || i == 0) break;
    }
    return (size_t)-1;
}

NDARRAY_EXPORT void npz_close_reader(void* handle) {
    struct NpzReader* reader = (struct NpzReader*)handle;
    if (!reader) return;

    if (reader->zip_initialized) {
        mz_zip_reader_end(&reader->zip);
        reader->zip_initialized = false;
    }

    if (reader->entries) {
        free(reader->entries);
        reader->entries = NULL;
    }

    if (reader->mmap_data) {
#if defined(_WIN32)
        UnmapViewOfFile(reader->mmap_data);
        if (reader->hMapping) CloseHandle(reader->hMapping);
#else
        munmap((void*)reader->mmap_data, reader->file_size);
#endif
        reader->mmap_data = NULL;
    }

    if (reader->fp) {
        fclose(reader->fp);
        reader->fp = NULL;
    }

    free(reader);
}

NDARRAY_EXPORT void* npz_open_reader(const char* filepath, int64_t* out_num_entries) {
    if (!filepath || !out_num_entries) return NULL;

    FILE* fp = npz_fopen_utf8(filepath, "rb");
    if (!fp) return NULL;

    npz_fseek64(fp, 0, SEEK_END);
    int64_t sz = npz_ftell64(fp);
    if (sz < 22 || (uint64_t)sz > (uint64_t)SIZE_MAX) {
        fclose(fp);
        return NULL;
    }
    size_t file_size = (size_t)sz;
    npz_fseek64(fp, 0, SEEK_SET);

    const uint8_t* mmap_data = NULL;
#if defined(_WIN32)
    HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(fp));
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping) {
        mmap_data = (const uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, file_size);
    }
#else
    int fd = fileno(fp);
    mmap_data = (const uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_data == MAP_FAILED) {
        mmap_data = NULL;
    }
#endif

    if (!mmap_data) {
#if defined(_WIN32)
        if (hMapping) CloseHandle(hMapping);
#endif
        fclose(fp);
        return NULL;
    }

    struct NpzReader* reader = (struct NpzReader*)calloc(1, sizeof(struct NpzReader));
    if (!reader) {
#if defined(_WIN32)
        UnmapViewOfFile(mmap_data);
        if (hMapping) CloseHandle(hMapping);
#else
        munmap((void*)mmap_data, file_size);
#endif
        fclose(fp);
        return NULL;
    }

    reader->fp = fp;
    reader->file_size = file_size;
    reader->mmap_data = mmap_data;
#if defined(_WIN32)
    reader->hFile = hFile;
    reader->hMapping = hMapping;
#else
    reader->fd = fd;
#endif

    size_t eocd_pos = find_eocd(mmap_data, file_size);
    if (eocd_pos == (size_t)-1) {
        npz_close_reader(reader);
        return NULL;
    }

    const uint8_t* eocd = mmap_data + eocd_pos;
    uint64_t total_entries = read_u16_le(eocd + 10);
    uint64_t cd_size = read_u32_le(eocd + 12);
    uint64_t cd_offset = read_u32_le(eocd + 16);

    // Check for ZIP64 End of Central Directory Locator (20 bytes before EOCD)
    if (eocd_pos >= 20 && read_u32_le(mmap_data + eocd_pos - 20) == 0x07064b50) {
        const uint8_t* z64_loc = mmap_data + eocd_pos - 20;
        uint32_t total_disks = read_u32_le(z64_loc + 16);
        if (total_disks != 1) {
            npz_close_reader(reader);
            return NULL;
        }
        uint64_t z64_eocd_off = read_u64_le(z64_loc + 8);
        if (z64_eocd_off > file_size || file_size - z64_eocd_off < 56) {
            npz_close_reader(reader);
            return NULL;
        }
        const uint8_t* z64_eocd = mmap_data + z64_eocd_off;
        if (read_u32_le(z64_eocd) != 0x06064b50) {
            npz_close_reader(reader);
            return NULL;
        }
        total_entries = read_u64_le(z64_eocd + 32);
        cd_size = read_u64_le(z64_eocd + 40);
        cd_offset = read_u64_le(z64_eocd + 48);
    }

    if (cd_offset > file_size || cd_size > file_size - cd_offset ||
        total_entries > SIZE_MAX / sizeof(NpzEntryInfo)) {
        npz_close_reader(reader);
        return NULL;
    }

    reader->num_files = (size_t)total_entries;
    reader->entries = (NpzEntryInfo*)calloc((size_t)total_entries, sizeof(NpzEntryInfo));
    if (!reader->entries && total_entries > 0) {
        npz_close_reader(reader);
        return NULL;
    }

    uint64_t cur_cd = cd_offset;
    for (size_t i = 0; i < (size_t)total_entries; i++) {
        if (cur_cd > file_size || file_size - cur_cd < 46) {
            npz_close_reader(reader);
            return NULL;
        }
        const uint8_t* cdh = mmap_data + cur_cd;
        if (read_u32_le(cdh) != 0x02014b50) {
            npz_close_reader(reader);
            return NULL;
        }

        NpzEntryInfo* e = &reader->entries[i];
        e->comp_method = read_u16_le(cdh + 10);
        e->crc32 = read_u32_le(cdh + 16);
        e->comp_size = read_u32_le(cdh + 20);
        e->uncomp_size = read_u32_le(cdh + 24);
        uint16_t nlen = read_u16_le(cdh + 28);
        uint16_t elen = read_u16_le(cdh + 30);
        uint16_t clen = read_u16_le(cdh + 32);
        e->local_header_offset = read_u32_le(cdh + 42);

        uint64_t cdh_total = 46ULL + (uint64_t)nlen + (uint64_t)elen + (uint64_t)clen;
        if (cdh_total > file_size - cur_cd) {
            npz_close_reader(reader);
            return NULL;
        }

        if (e->uncomp_size == 0xFFFFFFFFULL ||
            e->comp_size == 0xFFFFFFFFULL ||
            e->local_header_offset == 0xFFFFFFFFULL) {
            const uint8_t* extra = cdh + 46 + nlen;
            uint16_t extra_rem = elen;
            bool found_z64 = false;
            while (extra_rem >= 4) {
                uint16_t field_id = read_u16_le(extra);
                uint16_t field_sz = read_u16_le(extra + 2);
                if (4 + (uint32_t)field_sz > (uint32_t)extra_rem) {
                    npz_close_reader(reader);
                    return NULL;
                }
                if (field_id == 0x0001) {
                    const uint8_t* fdata = extra + 4;
                    uint16_t frem = field_sz;
                    bool need_uncomp = (e->uncomp_size == 0xFFFFFFFFULL);
                    bool need_comp = (e->comp_size == 0xFFFFFFFFULL);
                    bool need_ofs = (e->local_header_offset == 0xFFFFFFFFULL);
                    // Compatibility with writers that include 8-byte comp_size whenever
                    // uncomp_size == 0xFFFFFFFF even if comp_size < 0xFFFFFFFF.
                    if (need_uncomp && !need_comp &&
                        field_sz == (uint16_t)((1 + 1 + (need_ofs ? 1 : 0)) * 8)) {
                        need_comp = true;
                    }
                    if (need_uncomp) {
                        if (frem < 8) {
                            npz_close_reader(reader);
                            return NULL;
                        }
                        e->uncomp_size = read_u64_le(fdata);
                        fdata += 8;
                        frem -= 8;
                    }
                    if (need_comp) {
                        if (frem < 8) {
                            npz_close_reader(reader);
                            return NULL;
                        }
                        e->comp_size = read_u64_le(fdata);
                        fdata += 8;
                        frem -= 8;
                    }
                    if (need_ofs) {
                        if (frem < 8) {
                            npz_close_reader(reader);
                            return NULL;
                        }
                        e->local_header_offset = read_u64_le(fdata);
                        fdata += 8;
                        frem -= 8;
                    }
                    found_z64 = true;
                    break;
                }
                extra += 4 + field_sz;
                extra_rem -= (uint16_t)(4 + field_sz);
            }
            if (!found_z64) {
                npz_close_reader(reader);
                return NULL;
            }
        }

        size_t copy_nlen = (nlen < sizeof(e->name) - 1) ? nlen : (sizeof(e->name) - 1);
        memcpy(e->name, cdh + 46, copy_nlen);
        e->name[copy_nlen] = '\0';

        cur_cd += cdh_total;

        if (copy_nlen > 0 && (e->name[copy_nlen - 1] == '/' || e->name[copy_nlen - 1] == '\\')) {
            e->is_directory = true;
            e->header_status = -3;
            continue;
        }

        if (copy_nlen < 4 || strcmp(e->name + copy_nlen - 4, ".npy") != 0) {
            e->is_npy = false;
            e->header_status = -4;
            continue;
        }
        e->is_npy = true;

        uint64_t lfh_off = e->local_header_offset;
        if (lfh_off > file_size || file_size - lfh_off < 30) {
            e->header_status = -2;
            continue;
        }
        const uint8_t* lfh = mmap_data + lfh_off;
        if (read_u32_le(lfh) != 0x04034b50) {
            e->header_status = -2;
            continue;
        }
        uint16_t lfh_flags = read_u16_le(lfh + 6);
        uint32_t lfh_crc32 = read_u32_le(lfh + 14);
        if ((lfh_flags & 8) == 0 && lfh_crc32 != 0 && lfh_crc32 != e->crc32) {
            e->header_status = -6;
            continue;
        }
        uint16_t lfh_nlen = read_u16_le(lfh + 26);
        uint16_t lfh_elen = read_u16_le(lfh + 28);
        uint64_t lfh_total = 30ULL + (uint64_t)lfh_nlen + (uint64_t)lfh_elen;
        if (lfh_total > file_size - lfh_off) {
            e->header_status = -2;
            continue;
        }
        e->data_offset = lfh_off + lfh_total;

        if (e->comp_size > file_size - e->data_offset) {
            e->header_status = -2;
            continue;
        }

        if (e->comp_method == 0) {
            if (e->comp_size != e->uncomp_size) {
                e->header_status = -2;
                continue;
            }
            if (e->uncomp_size < 10) {
                e->header_status = -7;
                continue;
            }
            const uint8_t* npy = mmap_data + e->data_offset;
            if (npy[0] != 0x93 || npy[1] != 'N' || npy[2] != 'U' ||
                npy[3] != 'M' || npy[4] != 'P' || npy[5] != 'Y') {
                e->header_status = -8;
                continue;
            }
            uint8_t major = npy[6];
            size_t total_header_len;
            if (major >= 2) {
                if (e->uncomp_size < 12) {
                    e->header_status = -7;
                    continue;
                }
                uint32_t hlen = read_u32_le(npy + 8);
                if ((uint64_t)hlen > (uint64_t)(SIZE_MAX - 12)) {
                    e->header_status = -11;
                    continue;
                }
                total_header_len = 12 + (size_t)hlen;
            } else {
                uint16_t hlen = read_u16_le(npy + 8);
                total_header_len = 10 + (size_t)hlen;
            }
            if (total_header_len > e->uncomp_size) {
                e->header_status = -11;
                continue;
            }
            e->header_len = total_header_len;
            e->header_status = 0;
        } else {
            e->header_status = 0;
        }
    }

    *out_num_entries = (int64_t)reader->num_files;
    return reader;
}

NDARRAY_EXPORT int npz_reader_get_entry_info(
    void* handle,
    size_t index,
    char* name_buf,
    size_t name_buf_len,
    uint8_t* header_buf,
    size_t header_buf_len,
    size_t* out_header_len,
    size_t* out_data_len) {
    struct NpzReader* reader = (struct NpzReader*)handle;
    if (!reader || index >= reader->num_files || index > (size_t)MZ_UINT32_MAX) return -1;
    if (!header_buf && header_buf_len > 0) return -1;

    NpzEntryInfo* e = &reader->entries[index];
    if (name_buf && name_buf_len > 0) {
        size_t nlen = strlen(e->name);
        size_t cplen = nlen < (name_buf_len - 1) ? nlen : (name_buf_len - 1);
        memcpy(name_buf, e->name, cplen);
        name_buf[cplen] = '\0';
    }

    if (e->header_status != 0) {
        return e->header_status;
    }

    if (e->comp_method == 0) {
        size_t total_header_len = (size_t)e->header_len;
        if (out_header_len) *out_header_len = total_header_len;
        if (out_data_len) *out_data_len = (size_t)(e->uncomp_size - total_header_len);
        if (header_buf_len < total_header_len) {
            return -9;
        }
        if (!header_buf) return -1;
        if (reader->mmap_data) {
            if (e->data_offset > reader->file_size ||
                e->uncomp_size > reader->file_size - e->data_offset) {
                return -2;
            }
            uint32_t computed_crc = npz_fast_crc32(
                0,
                reader->mmap_data + e->data_offset,
                (size_t)e->uncomp_size);
            if (computed_crc != e->crc32) {
                return -6;
            }
            memcpy(header_buf, reader->mmap_data + e->data_offset, total_header_len);
        } else {
            npz_fseek64(reader->fp, (int64_t)e->data_offset, SEEK_SET);
            if (fread(header_buf, 1, total_header_len, reader->fp) != total_header_len) {
                return -10;
            }
        }
        return 0;
    } else {
        if (!reader->zip_initialized) {
            mz_zip_zero_struct(&reader->zip);
            reader->zip.m_pAlloc = npz_alloc;
            reader->zip.m_pFree = npz_free;
            reader->zip.m_pRealloc = npz_realloc;
            if (reader->mmap_data) {
                if (!mz_zip_reader_init_mem(&reader->zip, reader->mmap_data, reader->file_size, 0)) {
                    return -5;
                }
            } else {
                if (!mz_zip_reader_init_cfile(&reader->zip, reader->fp, reader->file_size, 0)) {
                    return -5;
                }
            }
            reader->zip_initialized = true;
        }

        mz_zip_reader_extract_iter_state* iter = mz_zip_reader_extract_iter_new(&reader->zip, (mz_uint)index, 0);
        if (!iter) return -5;

        if (!header_buf || header_buf_len < 10) {
            mz_zip_reader_extract_iter_free(iter);
            return -6;
        }

        uint8_t npy_prefix[12];
        size_t n = mz_zip_reader_extract_iter_read(iter, npy_prefix, 10);
        if (n < 10) {
            mz_zip_reader_extract_iter_free(iter);
            return -7;
        }

        if (npy_prefix[0] != 0x93 || npy_prefix[1] != 'N' || npy_prefix[2] != 'U' ||
            npy_prefix[3] != 'M' || npy_prefix[4] != 'P' || npy_prefix[5] != 'Y') {
            mz_zip_reader_extract_iter_free(iter);
            return -8;
        }

        uint8_t major = npy_prefix[6];
        size_t prefix_len;
        size_t total_header_len;
        if (major >= 2) {
            if (mz_zip_reader_extract_iter_read(iter, npy_prefix + 10, 2) != 2) {
                mz_zip_reader_extract_iter_free(iter);
                return -7;
            }
            uint32_t hlen = read_u32_le(npy_prefix + 8);
            if ((uint64_t)hlen > (uint64_t)(SIZE_MAX - 12)) {
                mz_zip_reader_extract_iter_free(iter);
                return -11;
            }
            prefix_len = 12;
            total_header_len = 12 + (size_t)hlen;
        } else {
            uint16_t hlen = read_u16_le(npy_prefix + 8);
            prefix_len = 10;
            total_header_len = 10 + (size_t)hlen;
        }

        if (total_header_len > e->uncomp_size) {
            mz_zip_reader_extract_iter_free(iter);
            return -11;
        }

        if (out_header_len) *out_header_len = total_header_len;
        if (out_data_len) *out_data_len = (size_t)(e->uncomp_size - total_header_len);

        if (total_header_len > header_buf_len) {
            mz_zip_reader_extract_iter_free(iter);
            return -9;
        }

        memcpy(header_buf, npy_prefix, prefix_len);
        size_t rem = total_header_len - prefix_len;
        if (rem > 0) {
            n = mz_zip_reader_extract_iter_read(iter, header_buf + prefix_len, rem);
            if (n != rem) {
                mz_zip_reader_extract_iter_free(iter);
                return -10;
            }
        }
        mz_zip_reader_extract_iter_free(iter);

        return 0;
    }
}

NDARRAY_EXPORT int npz_reader_extract_data(
    void* handle,
    size_t index,
    size_t header_len,
    void* dest_ptr,
    size_t dest_capacity,
    size_t data_len) {
    struct NpzReader* reader = (struct NpzReader*)handle;
    if (!reader || index >= reader->num_files || index > (size_t)MZ_UINT32_MAX || !dest_ptr) return -1;
    if (data_len > dest_capacity) return -5;

    NpzEntryInfo* e = &reader->entries[index];
    if (header_len > e->uncomp_size || data_len != e->uncomp_size - header_len) {
        return -4;
    }

    if (e->comp_method == 0) {
        if (e->comp_size != e->uncomp_size) {
            return -4;
        }
        if (e->data_offset > reader->file_size ||
            e->uncomp_size > reader->file_size - e->data_offset) {
            return -4;
        }
        size_t src_offset = (size_t)e->data_offset + header_len;
        if (data_len > reader->file_size - src_offset) {
            return -4;
        }
        if (reader->mmap_data) {
            uint32_t computed_crc = npz_fast_crc32(
                0,
                reader->mmap_data + e->data_offset,
                (size_t)e->uncomp_size);
            if (computed_crc != e->crc32) {
                return -6;
            }
            memcpy(dest_ptr, reader->mmap_data + src_offset, data_len);
        } else {
            npz_fseek64(reader->fp, (int64_t)e->data_offset, SEEK_SET);
            uint32_t computed_crc = 0;
            size_t skip_remaining = header_len;
            uint8_t skip_buf[512];
            while (skip_remaining > 0) {
                size_t to_read = skip_remaining < 512 ? skip_remaining : 512;
                if (fread(skip_buf, 1, to_read, reader->fp) != to_read) {
                    return -4;
                }
                computed_crc = npz_fast_crc32(computed_crc, skip_buf, to_read);
                skip_remaining -= to_read;
            }
            if (fread(dest_ptr, 1, data_len, reader->fp) != data_len) {
                return -4;
            }
            computed_crc = npz_fast_crc32(computed_crc, dest_ptr, data_len);
            if (computed_crc != e->crc32) {
                return -6;
            }
        }
        return 0;
    } else {
        if (!reader->zip_initialized) {
            mz_zip_zero_struct(&reader->zip);
            reader->zip.m_pAlloc = npz_alloc;
            reader->zip.m_pFree = npz_free;
            reader->zip.m_pRealloc = npz_realloc;
            if (reader->mmap_data) {
                if (!mz_zip_reader_init_mem(&reader->zip, reader->mmap_data, reader->file_size, 0)) {
                    return -2;
                }
            } else {
                if (!mz_zip_reader_init_cfile(&reader->zip, reader->fp, reader->file_size, 0)) {
                    return -2;
                }
            }
            reader->zip_initialized = true;
        }

        mz_zip_reader_extract_iter_state* iter = mz_zip_reader_extract_iter_new(&reader->zip, (mz_uint)index, 0);
        if (!iter) return -2;

        uint32_t computed_crc = 0;
        size_t skip_remaining = header_len;
        uint8_t skip_buf[512];
        while (skip_remaining > 0) {
            size_t to_read = skip_remaining < 512 ? skip_remaining : 512;
            size_t n = mz_zip_reader_extract_iter_read(iter, skip_buf, to_read);
            if (n != to_read) {
                mz_zip_reader_extract_iter_free(iter);
                return -3;
            }
            computed_crc = npz_fast_crc32(computed_crc, skip_buf, to_read);
            skip_remaining -= to_read;
        }

        size_t n = mz_zip_reader_extract_iter_read(iter, dest_ptr, data_len);
        if (n == data_len) {
            computed_crc = npz_fast_crc32(computed_crc, dest_ptr, data_len);
        }
        mz_bool iter_ok = mz_zip_reader_extract_iter_free(iter);
        if (n != data_len) return -4;
        if (!iter_ok || computed_crc != e->crc32) {
            return -6;
        }

        return 0;
    }
}


NDARRAY_EXPORT int native_file_last_error(uint8_t* out_message, int64_t capacity) {
    int code = g_native_file_errno;
    if (out_message && capacity > 0) {
        const char* message = strerror(code);
        size_t len = message ? strlen(message) : 0;
        if ((int64_t)len >= capacity) len = (size_t)(capacity - 1);
        if (len > 0) memcpy(out_message, message, len);
        out_message[len] = '\0';
    }
    return code;
}

NDARRAY_EXPORT int native_file_write_all(
    const char* filepath,
    const uint8_t* header,
    int64_t header_len,
    const void* data,
    int64_t data_len) {
    if (!filepath || header_len < 0 || data_len < 0) {
        g_native_file_errno = EINVAL;
        return -1;
    }
    npz_ensure_parent_dirs(filepath);
    FILE* fp = npz_fopen_utf8(filepath, "wb");
    if (!fp) return -1;
    if (header_len > 0) {
        if (!header || (int64_t)fwrite(header, 1, (size_t)header_len, fp) != header_len) {
            npz_record_errno();
            fclose(fp);
            return -2;
        }
    }
    if (data_len > 0) {
        if (!data || (int64_t)fwrite(data, 1, (size_t)data_len, fp) != data_len) {
            npz_record_errno();
            fclose(fp);
            return -2;
        }
    }
    if (fclose(fp) != 0) {
        npz_record_errno();
        return -2;
    }
    return 0;
}

NDARRAY_EXPORT void* native_file_open_read(const char* filepath) {
    if (!filepath) {
        g_native_file_errno = EINVAL;
        return NULL;
    }
#if !defined(_WIN32)
    struct stat st;
    if (stat(filepath, &st) != 0) {
        npz_record_errno();
        return NULL;
    }
    if (S_ISDIR(st.st_mode)) {
        g_native_file_errno = EISDIR;
        return NULL;
    }
#endif
    return npz_fopen_utf8(filepath, "rb");
}

NDARRAY_EXPORT int64_t native_file_handle_size(void* handle) {
    FILE* fp = (FILE*)handle;
    if (!fp) {
        g_native_file_errno = EINVAL;
        return -1;
    }
    int64_t current = npz_ftell64(fp);
    if (current < 0 || npz_fseek64(fp, 0, SEEK_END) != 0) {
        npz_record_errno();
        return -1;
    }
    int64_t sz = npz_ftell64(fp);
    if (sz < 0) npz_record_errno();
    if (npz_fseek64(fp, current, SEEK_SET) != 0) {
        npz_record_errno();
        return -1;
    }
    return sz;
}

NDARRAY_EXPORT int native_file_handle_read(
    void* handle,
    int64_t offset,
    int64_t len,
    void* out_data,
    int64_t* out_read) {
    FILE* fp = (FILE*)handle;
    if (!fp || offset < 0 || len < 0 || !out_read || (len > 0 && !out_data)) {
        g_native_file_errno = EINVAL;
        return -1;
    }
    *out_read = 0;
    if (npz_fseek64(fp, offset, SEEK_SET) != 0) {
        npz_record_errno();
        return -2;
    }
    if (len > 0) {
        size_t n = fread(out_data, 1, (size_t)len, fp);
        *out_read = (int64_t)n;
        if ((int64_t)n != len && ferror(fp)) {
            npz_record_errno();
            return -2;
        }
    }
    return 0;
}

NDARRAY_EXPORT void native_file_close(void* handle) {
    if (handle) fclose((FILE*)handle);
}
