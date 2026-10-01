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

/// Deep Learning & Neural Network Library for [package:gpuarray].
///
/// Features PyTorch-style [Module], trainable [Linear], [Conv2d], [LayerNorm],
/// [RMSNorm], [BatchNorm1d], [Embedding], [Dropout], [MultiheadAttention]
/// ([MultiHeadAttention]), [RotaryEmbedding], [SwiGLU], [GeGLU],
/// [TransformerEncoderLayer], and [TransformerDecoderLayer], activations
/// ([relu], [gelu], [silu], [softmax], [logSoftmax]), loss functions and
/// criteria ([mseLoss], [MSELoss], [l1Loss], [L1Loss], [binaryCrossEntropy],
/// [BCELoss], [crossEntropy], [CrossEntropyLoss], [LossReduction]), and
/// optimizers ([SGD], [Adam], [AdamW]).
library;

export 'src/nn/nn.dart';
