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

#define HAVE_LAPACK_CONFIG_H
#define HAVE_STDINT_H
#include <stdint.h>
#define int32_t int32_t
#include <lapacke.h>

void* get_dgetrf_ptr(void) {
    return (void*)&LAPACKE_dgetrf;
}

void* get_sgetrf_ptr(void) {
    return (void*)&LAPACKE_sgetrf;
}

void* get_zgetrf_ptr(void) {
    return (void*)&LAPACKE_zgetrf;
}

void* get_cgetrf_ptr(void) {
    return (void*)&LAPACKE_cgetrf;
}
