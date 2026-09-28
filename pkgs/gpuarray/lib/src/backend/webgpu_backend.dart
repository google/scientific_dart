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

/// WebGPU compute hardware acceleration backend driver.
///
/// Automatically routes between [BrowserWebGpuBackend] (on Web with package:web / dart:js_interop)
/// and [WgpuNativeBackend] (on native desktop/server with wgpu-native C-FFI).
library;

export 'native/wgpu_native_backend.dart'
    if (dart.library.js_interop) 'web/webgpu_backend.dart';
