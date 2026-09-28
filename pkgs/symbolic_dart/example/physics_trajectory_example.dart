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

import 'package:ndarray/ndarray.dart' hide exp;
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  print('--- Damped Projectile Trajectory in 2D ---');
  final t = Symbol('t');

  // Parameters
  final v0x = Real(20.0);
  final v0y = Real(25.0);
  final g = Real(9.81);
  final gamma = Real(0.2); // Damping coefficient

  // Symbolic position x(t) and y(t) with air resistance
  // x(t) = (v0x / gamma) * (1 - exp(-gamma * t))
  final xt = (v0x / gamma) * (Expr.one - exp(-gamma * t));

  // y(t) = ( (v0y + g/gamma)/gamma ) * (1 - exp(-gamma*t)) - (g/gamma)*t
  final yt =
      ((v0y + (g / gamma)) / gamma) * (Expr.one - exp(-gamma * t)) -
      ((g / gamma) * t);

  print('Position x(t)     = $xt');
  print('Position y(t)     = $yt\n');

  // Symbolic velocity v(t) = dr/dt
  final vx = xt.diff(t);
  final vy = yt.diff(t);
  print('Velocity vx(t)    = $vx');
  print('Velocity vy(t)    = $vy\n');

  // Symbolic acceleration a(t) = dv/dt
  final ax = vx.diff(t);
  final ay = vy.diff(t);
  print('Acceleration ax(t) = $ax');
  print('Acceleration ay(t) = $ay\n');

  // Vectorized numerical evaluation across time steps t = 0.0 .. 3.0
  final timeGrid = NDArray.fromList(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    [7],
    DType.float64,
  );

  final lambdifyX = xt.lambdify([t]);
  final lambdifyY = yt.lambdify([t]);
  final lambdifyVy = vy.lambdify([t]);

  final xVals = lambdifyX.callArray([timeGrid]);
  final yVals = lambdifyY.callArray([timeGrid]);
  final vyVals = lambdifyVy.callArray([timeGrid]);

  print('Time (s) |   x (m)   |   y (m)   |  vy (m/s)');
  print('---------+-----------+-----------+-----------');
  for (var i = 0; i < timeGrid.shape[0]; i++) {
    final ti = timeGrid.getCell([i]);
    final xi = xVals.getCell([i]);
    final yi = yVals.getCell([i]);
    final vyi = vyVals.getCell([i]);
    print(
      '${ti.toStringAsFixed(1).padLeft(8)} | '
      '${xi.toStringAsFixed(2).padLeft(9)} | '
      '${yi.toStringAsFixed(2).padLeft(9)} | '
      '${vyi.toStringAsFixed(2).padLeft(9)}',
    );
  }
}
