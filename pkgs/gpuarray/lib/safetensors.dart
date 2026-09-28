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

/// SafeTensors Zero-Copy Tensor Serialization for [package:gpuarray].
///
/// Implements HuggingFace's SafeTensors standard specification for efficient,
/// zero-copy model weight saving and loading directly from GPU memory.
library;

export 'src/serialization/safetensors.dart';
