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

import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    final arr = NDArray<DTypeTag>.fromList([1.0, 2.0, 3.0], [3], DType.float64);

    // Constant padding
    final constantPadded = pad(
      arr,
      PadWidth.all(2),
      mode: PaddingMode.constant,
      constantValues: PadValues.all(0.0),
    );
    print('Constant padded: ${constantPadded.toList()}');
    // Output: [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]

    // Edge padding
    final edgePadded = pad(arr, PadWidth.all(2), mode: PaddingMode.edge);
    print('Edge padded: ${edgePadded.toList()}');
    // Output: [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]

    // Reflect padding
    final reflectPadded = pad(arr, PadWidth.all(2), mode: PaddingMode.reflect);
    print('Reflect padded: ${reflectPadded.toList()}');
    // Output: [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
  });
}
