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

import "package:ndarray/ndarray.dart";

void main() {
  NDArray.scope(() {
    // log2 example
    final a = NDArray.fromList([1.0, 2.0, 4.0, 8.0], [4], DType.float64);
    final resLog2 = log2(a);
    print("log2(a) = ${resLog2.toList()}");

    // log10 example
    final b = NDArray.fromList([1.0, 10.0, 100.0], [3], DType.float64);
    final resLog10 = log10(b);
    print("log10(b) = ${resLog10.toList()}");

    // reciprocal example
    final c = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
    final resReciprocal = reciprocal(c);
    print("reciprocal(c) = ${resReciprocal.toList()}");

    // positive example
    final d = NDArray.fromList([-1.0, 2.0, -3.0], [3], DType.float64);
    final resPositive = positive(d);
    print("positive(d) = ${resPositive.toList()}");
  });
}
