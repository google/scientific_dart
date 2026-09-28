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

import 'dart:typed_data';

import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray NumPy Binary Interoperability Examples ===\n');
  runNpySaveAndLoadExample();
  runNpzMultiSaveAndLoadExample();
}

void runNpySaveAndLoadExample() {
  print('--- NumPy .npy Save & Load (Zero-Copy Binary Block I/O) ---');
  // Create a 2x3 Float64 matrix
  final a = NDArray.fromList(
    Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    [2, 3],
    DType.float64,
  );

  print('Original Array a:');
  print(a.toList());

  const filepath = 'scratch/array_a.npy';

  // save() handles 64-byte padding aligned ASCII dictionary headers and raw C-heap byte dumps
  save(filepath, a);
  print('Array a successfully saved to: $filepath');

  // load() parses the header and block-copies data directly into new C memory!
  final loadedA = load(filepath);
  print('Array successfully loaded back from: $filepath');
  print('Loaded shape: ${loadedA.shape}, dtype: ${loadedA.dtype}');
  print('Loaded Data: ${loadedA.toList()}\n');
}

void runNpzMultiSaveAndLoadExample() {
  print('--- NumPy .npz Multi-Array Zip Archives ---');
  final weights = NDArray.fromList(Float32List.fromList([0.1, 0.2, 0.3]), [
    3,
  ], DType.float32);
  final biases = NDArray.fromList(Int32List.fromList([1, 2, 3]), [
    3,
  ], DType.int32);

  final modelMap = {'weights': weights, 'biases': biases};

  const npzPath = 'scratch/model.npz';

  // savez() archives multiple arrays into a standard zip file, where each variable is variable_name.npy!
  // It also supports compressed = true deflation for small files!
  savez(npzPath, modelMap, compressed: true);
  print('Multiple arrays archived to compressed zip at: $npzPath');

  // loadz() decodes the zip and extracts all variable keys back into distinct NDArray instances!
  final loadedModel = loadz(npzPath);
  print('Model map successfully unzipped and loaded from: $npzPath');

  final w = loadedModel['weights']!;
  final b = loadedModel['biases']!;

  print(
    'Loaded "weights" -> shape: ${w.shape}, dtype: ${w.dtype}, data: ${w.toList()}',
  );
  print(
    'Loaded "biases" -> shape: ${b.shape}, dtype: ${b.dtype}, data: ${b.toList()}',
  );
}
