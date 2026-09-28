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

/// ANSI VT100 Terminal renderer backend.
///
/// Features double-buffered ANSI 24-bit TrueColor cell matrix rendering,
/// minimum stdout delta updates, and raw mode stdin escape sequence parser.
library;

export '../src/backends/terminal/vt100_encoder.dart';
export '../src/backends/terminal/terminal_input_parser.dart';
export '../src/backends/terminal/terminal_renderer.dart';
