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

import 'dart:math' as math;

import 'package:ndarray/ndarray.dart';

/// Helper function to compare two lists of integers.
bool _listEquals(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// A multidimensional Kalman Filter implementation using the `ndarray` package.
///
/// This filter estimates the state of a linear dynamic system from a series of
/// noisy measurements. It uses matrix operations for prediction and update steps.
///
/// Memory management is handled using [NDArray.scope] internally. The state [x]
/// and covariance [P] are updated in-place to avoid memory leaks and manual disposal.
final class KalmanFilter {
  /// The current state estimate vector (shape [N, 1]).
  final NDArray<Float64> x;

  /// The state covariance estimate matrix (shape [N, N]).
  final NDArray<Float64> P;

  /// The state transition model matrix (shape [N, N]).
  final NDArray<Float64> F;

  /// The transposed state transition model matrix (shape [N, N]).
  final NDArray<Float64> fT;

  /// The process noise covariance matrix (shape [N, N]).
  final NDArray<Float64> Q;

  /// The measurement model matrix (shape [M, N]).
  final NDArray<Float64> H;

  /// The transposed measurement model matrix (shape [N, M]).
  final NDArray<Float64> hT;

  /// The measurement noise covariance matrix (shape [M, M]).
  final NDArray<Float64> R;

  /// The identity matrix of the same dimension as the state (shape [N, N]).
  final NDArray<Float64> I;

  // Pre-allocated temporary buffers for predict() to avoid garbage collection churn.
  late final NDArray<Float64> _newX;
  late final NDArray<Float64> _fp;
  late final NDArray<Float64> _fpfT;

  // Pre-allocated temporary buffers for update() to avoid garbage collection churn.
  late final NDArray<Float64> _hx;
  late final NDArray<Float64> _y;
  late final NDArray<Float64> _hp;
  late final NDArray<Float64> _hphT;
  late final NDArray<Float64> _s;
  late final NDArray<Float64> _kTransposed;
  late final NDArray<Float64> _ky;
  late final NDArray<Float64> _khp;

  /// Creates a [KalmanFilter] with the initial state and models.
  ///
  /// The [x] and [P] arrays are used as internal storage and will be modified
  /// in-place during [predict] and [update] steps. They must be contiguous.
  KalmanFilter({
    required this.x,
    required this.P,
    required this.F,
    required this.Q,
    required this.H,
    required this.R,
  }) : fT = F.transposed,
       hT = H.transposed,
       I = NDArray<Float64>.eye(x.shape[0], DType.float64) {
    _validateInputs();
    _initBuffers();
  }

  void _validateInputs() {
    final n = x.shape[0];
    final m = R.shape[0];

    if (x.shape.length != 2 || x.shape[1] != 1) {
      throw ArgumentError(
        'State x must be a 2D column vector of shape [N, 1].',
      );
    }
    if (!_listEquals(P.shape, [n, n])) {
      throw ArgumentError('Covariance P must be $n x $n.');
    }
    if (!_listEquals(F.shape, [n, n])) {
      throw ArgumentError('Transition F must be $n x $n.');
    }
    if (!_listEquals(Q.shape, [n, n])) {
      throw ArgumentError('Process noise Q must be $n x $n.');
    }
    if (H.shape.length != 2 || H.shape[1] != n) {
      throw ArgumentError('Measurement model H must be $m x $n.');
    }
    if (!_listEquals(R.shape, [m, m])) {
      throw ArgumentError('Measurement noise R must be $m x $m.');
    }
    if (!x.isContiguous || !P.isContiguous) {
      throw ArgumentError('State x and covariance P must be contiguous.');
    }
  }

  void _initBuffers() {
    final n = x.shape[0];
    final m = R.shape[0];

    _newX = NDArray<Float64>.create([n, 1], DType.float64);
    _fp = NDArray<Float64>.create([n, n], DType.float64);
    _fpfT = NDArray<Float64>.create([n, n], DType.float64);

    _hx = NDArray<Float64>.create([m, 1], DType.float64);
    _y = NDArray<Float64>.create([m, 1], DType.float64);
    _hp = NDArray<Float64>.create([m, n], DType.float64);
    _hphT = NDArray<Float64>.create([m, m], DType.float64);
    _s = NDArray<Float64>.create([m, m], DType.float64);
    _kTransposed = NDArray<Float64>.create([m, n], DType.float64);
    _ky = NDArray<Float64>.create([n, 1], DType.float64);
    _khp = NDArray<Float64>.create([n, n], DType.float64);
  }

  /// Predicts the next state and covariance.
  ///
  /// Projects the state estimate and covariance forward in time:
  /// - $x = F \cdot x$
  /// - $P = F \cdot P \cdot F^T + Q$
  ///
  /// This method is entirely allocation-free, updating [x] and [P] in-place
  /// using pre-allocated buffers.
  void predict() {
    matmul<Float64>(F, x, out: _newX);
    matmul<Float64>(F, P, out: _fp);
    matmul<Float64>(_fp, fT, out: _fpfT);
    add<Float64>(_fpfT, Q, out: P);
    _newX.copy(out: x);
  }

  /// Updates the state and covariance with a new measurement [z].
  ///
  /// Adjusts the state and covariance estimates using the new measurement [z] (shape [M, 1]):
  /// - $\tilde{y} = z - H \cdot x$ (innovation)
  /// - $S = H \cdot P \cdot H^T + R$ (residual covariance)
  /// - $K = P \cdot H^T \cdot S^{-1}$ (Kalman gain, solved via $S \cdot K^T = H \cdot P$)
  /// - $x = x + K \cdot \tilde{y}$
  /// - $P = (I - K \cdot H) \cdot P = P - K \cdot H \cdot P$
  ///
  /// This method updates [x] and [P] in-place and minimizes allocations by reusing
  /// pre-allocated buffers.
  void update(NDArray<Float64> z) {
    // y = z - H * x
    matmul<Float64>(H, x, out: _hx);
    subtract<Float64>(z, _hx, out: _y);

    // S = H * P * H^T + R
    matmul<Float64>(H, P, out: _hp);
    matmul<Float64>(_hp, hT, out: _hphT);
    add<Float64>(_hphT, R, out: _s);

    // K = P * H^T * S^-1
    // Solve: S * K^T = H * P  => K^T = solve(S, H * P) => K = solve(S, H * P)^T
    solve<Float64>(_s, _hp, out: _kTransposed);
    final K = _kTransposed.transposed; // Zero-copy view

    // x = x + K * y
    matmul<Float64>(K, _y, out: _ky);
    add<Float64>(x, _ky, out: x);

    // P = P - K * H * P = P - K * hp
    matmul<Float64>(K, _hp, out: _khp);
    subtract<Float64>(P, _khp, out: P);
  }
}

void main() {
  print('=== Kalman Filter 2D Constant Velocity Simulation ===\n');

  NDArray.scope(() {
    const steps = 100;
    const dt = 1.0;

    // --- 1. Define Models ---

    // State transition matrix F (constant velocity model in 2D)
    // State vector: [x_pos, y_pos, x_vel, y_vel]^T
    final F = NDArray<Float64>.fromList(
      ([
        1.0, 0.0, dt, 0.0, //
        0.0, 1.0, 0.0, dt, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
      ]).map((e) => (e as num).toDouble()).toList(),
      [4, 4],
      DType.float64,
    );

    // Observation model H (we only measure position x and y)
    final H = NDArray<Float64>.fromList(
      ([
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
      ]).map((e) => (e as num).toDouble()).toList(),
      [2, 4],
      DType.float64,
    );

    // Process noise covariance Q (low noise on position, higher on velocity)
    final Q = NDArray<Float64>.fromList(
      ([
        0.01, 0.0, 0.0, 0.0, //
        0.0, 0.01, 0.0, 0.0, //
        0.0, 0.0, 0.1, 0.0, //
        0.0, 0.0, 0.0, 0.1, //
      ]).map((e) => (e as num).toDouble()).toList(),
      [4, 4],
      DType.float64,
    );

    // Measurement noise covariance R (GPS-like, stddev = 2.0, var = 4.0)
    final R = NDArray<Float64>.fromList(
      ([4.0, 0.0, 0.0, 4.0]).map((e) => (e as num).toDouble()).toList(),
      [2, 2],
      DType.float64,
    );

    // --- 2. Initialize State ---

    // True initial state (start at origin, moving at velocity (1, 1))
    var trueState = NDArray<Float64>.fromList(
      ([0.0, 0.0, 1.0, 1.0]).map((e) => (e as num).toDouble()).toList(),
      [4, 1],
      DType.float64,
    );

    // Initial estimated state (slightly off, high covariance)
    final estState = NDArray<Float64>.fromList(
      ([0.5, -0.5, 0.0, 0.0]).map((e) => (e as num).toDouble()).toList(),
      [4, 1],
      DType.float64,
    );
    final P = NDArray<Float64>.fromList(
      ([
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 10.0, 0.0, //
        0.0, 0.0, 0.0, 10.0, //
      ]).map((e) => (e as num).toDouble()).toList(),
      [4, 4],
      DType.float64,
    );

    final kf = KalmanFilter(x: estState, P: P, F: F, Q: Q, H: H, R: R);

    // Track statistics for Mean Squared Error (MSE)
    var totalMeasurementErrorSq = 0.0;
    var totalEstimationErrorSq = 0.0;

    print('Starting simulation of $steps steps...');
    print('True Initial State:      [0.00, 0.00, 1.00, 1.00]');
    print('Estimated Initial State: [0.50, -0.50, 0.00, 0.00]\n');

    const rngSeed = 42; // Seed for reproducibility

    for (var k = 1; k <= steps; k++) {
      NDArray.scope(() {
        // a. Simulate true state update: trueState = F * trueState + process_noise
        final nextTrue = matmul<Float64>(F, trueState);
        final w = multivariateNormal<Float64>(
          NDArray.zeros([4], DType.float64),
          Q,
          seed: rngSeed + k,
        ).reshape([4, 1]);
        final updatedTrue = add<Float64>(nextTrue, w).detachToParentScope();

        trueState.dispose();
        trueState = updatedTrue;

        // b. Generate measurement: z = H * trueState + measurement_noise
        final hTrue = matmul<Float64>(H, trueState);
        final v = multivariateNormal<Float64>(
          NDArray.zeros([2], DType.float64),
          R,
          seed: rngSeed + k * 100,
        ).reshape([2, 1]);
        final z = add<Float64>(hTrue, v);

        // c. Kalman Filter Steps: Predict then Update
        kf.predict();
        kf.update(z);

        // d. Calculate errors (comparing position coordinates only)
        final tx = trueState.getCell([0, 0]);
        final ty = trueState.getCell([1, 0]);
        final mx = z.getCell([0, 0]);
        final my = z.getCell([1, 0]);
        final ex = kf.x.getCell([0, 0]);
        final ey = kf.x.getCell([1, 0]);

        final mErrSq = math.pow(mx - tx, 2) + math.pow(my - ty, 2);
        final eErrSq = math.pow(ex - tx, 2) + math.pow(ey - ty, 2);

        totalMeasurementErrorSq += mErrSq;
        totalEstimationErrorSq += eErrSq;

        if (k % 20 == 0 || k == 1) {
          print('Step $k:');
          print(
            '  True position:      (${tx.toStringAsFixed(2)}, ${ty.toStringAsFixed(2)})',
          );
          print(
            '  Measured position:  (${mx.toStringAsFixed(2)}, ${my.toStringAsFixed(2)})',
          );
          print(
            '  Estimated position: (${ex.toStringAsFixed(2)}, ${ey.toStringAsFixed(2)})',
          );
        }
      });
    }

    final mseMeasurement = totalMeasurementErrorSq / steps;
    final mseEstimation = totalEstimationErrorSq / steps;

    print('\n--- Simulation Results ---');
    print('Measurement MSE (Position): ${mseMeasurement.toStringAsFixed(4)}');
    print(
      'Estimated State MSE (Position): ${mseEstimation.toStringAsFixed(4)}',
    );

    final noiseReduction = (1.0 - (mseEstimation / mseMeasurement)) * 100;
    print('Noise reduction: ${noiseReduction.toStringAsFixed(1)}%');

    // Print final states
    final tx = trueState.getCell([0, 0]);
    final ty = trueState.getCell([1, 0]);
    final tvx = trueState.getCell([2, 0]);
    final tvy = trueState.getCell([3, 0]);

    final ex = kf.x.getCell([0, 0]);
    final ey = kf.x.getCell([1, 0]);
    final evx = kf.x.getCell([2, 0]);
    final evy = kf.x.getCell([3, 0]);

    print(
      '\nFinal True State:      '
      '[${tx.toStringAsFixed(2)}, ${ty.toStringAsFixed(2)}, '
      '${tvx.toStringAsFixed(2)}, ${tvy.toStringAsFixed(2)}]',
    );
    print(
      'Final Estimated State: '
      '[${ex.toStringAsFixed(2)}, ${ey.toStringAsFixed(2)}, '
      '${evx.toStringAsFixed(2)}, ${evy.toStringAsFixed(2)}]',
    );

    // trueState is disposed here when the outer scope ends, as it was detached to parent scope.
    // kf.x and kf.P are also disposed when the outer scope ends because they were passed to kf
    // and they were tracked by the outer scope (created in main's scope).
  });
}
