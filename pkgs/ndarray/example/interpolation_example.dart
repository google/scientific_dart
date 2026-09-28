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
    // Define data points
    final xp = NDArray.fromList([0.0, 1.0, 2.0, 5.0], [4], DType.float64);
    final fp = NDArray.fromList([0.0, 2.0, 3.0, 10.0], [4], DType.float64);

    // Define points to evaluate
    final x = NDArray.fromList([-1.0, 0.5, 1.5, 3.0, 6.0], [5], DType.float64);

    print('xp (data points):');
    print(xp.toList());
    print('fp (data values):');
    print(fp.toList());
    print('x (eval points):');
    print(x.toList());

    // Perform interpolation
    final y = interp(x, xp, fp);
    print('Interpolated values:');
    print(y.toList());

    // Perform interpolation with custom boundary values
    final yCustom = interp(x, xp, fp, left: -99.0, right: 99.0);
    print('Interpolated values (custom boundaries):');
    print(yCustom.toList());
  });
}
