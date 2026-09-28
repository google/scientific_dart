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
@ffi.DefaultAsset('package:openblas/openblas_extensions')
library;

import 'dart:ffi' as ffi;

@ffi.Native<ffi.Pointer<ffi.Void> Function()>()
external ffi.Pointer<ffi.Void> get_dgetrf_ptr();

@ffi.Native<ffi.Pointer<ffi.Void> Function()>()
external ffi.Pointer<ffi.Void> get_sgetrf_ptr();

@ffi.Native<ffi.Pointer<ffi.Void> Function()>()
external ffi.Pointer<ffi.Void> get_zgetrf_ptr();

@ffi.Native<ffi.Pointer<ffi.Void> Function()>()
external ffi.Pointer<ffi.Void> get_cgetrf_ptr();
