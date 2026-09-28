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

/// Web HTML/DOM virtualized renderer backend.
///
/// Provides HTML virtual list rendering with node recycling, XSS attribute escaping,
/// CSS selection overlays, and DOM event bindings.
library;

export '../src/backends/html/dom_node_pool.dart';
export '../src/backends/html/html_code_editor_stub.dart'
    if (dart.library.js_interop) '../src/backends/html/html_code_editor.dart';
export '../src/backends/html/html_renderer.dart';
