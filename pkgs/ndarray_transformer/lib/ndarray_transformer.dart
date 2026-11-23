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

/// Pure-Dart BERT / `all-MiniLM-L6-v2` transformer inference and semantic
/// search built on `package:ndarray`.
library;

export 'src/bert_model.dart';
export 'src/model_downloader.dart';
export 'src/safetensors.dart';
export 'src/semantic_search.dart';
export 'src/wordpiece_tokenizer.dart';
