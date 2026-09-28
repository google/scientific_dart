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
import '../gpu_array.dart';
import '../autograd/autograd.dart';

/// Base class for all parameter optimizers.
abstract class Optimizer {
  /// The parameters to optimize.
  final List<GpuArray> params;

  /// Learning rate.
  double lr;

  Optimizer(this.params, {required this.lr});

  /// Performs a single optimization step (parameter update).
  void step();

  /// Clears the gradients of all optimized [params].
  void zeroGrad() {
    for (final p in params) {
      p.zeroGrad();
    }
  }

  /// Disposes internal optimizer states.
  void dispose();
}

/// Stochastic Gradient Descent (SGD) optimizer with momentum and weight decay.
class SGD extends Optimizer {
  final double momentum;
  final double weightDecay;
  final bool nesterov;

  final Map<GpuArray, GpuArray> _velocity = {};

  SGD(
    super.params, {
    required super.lr,
    this.momentum = 0.0,
    this.weightDecay = 0.0,
    this.nesterov = false,
  });

  @override
  void step() {
    no_grad(() {
      for (final p in params) {
        final grad = p.grad;
        if (grad == null) continue;

        var dP = grad;
        GpuArray? dPDecayed;
        if (weightDecay != 0.0) {
          final pDecay = p * weightDecay;
          dPDecayed = dP + pDecay;
          pDecay.dispose();
          dP = dPDecayed;
        }

        if (momentum != 0.0) {
          var v = _velocity[p];
          if (v == null) {
            v = GpuArray.zeros(p.shape, p.dtype, device: p.device);
            _velocity[p] = v;
          }

          // v = momentum * v + dP
          final vScaled = v * momentum;
          final newV = vScaled + dP;
          vScaled.dispose();
          newV.buffer.copyToBuffer(v.buffer, v.byteSize);
          newV.dispose();

          if (nesterov) {
            final vNesterov = v * momentum;
            final dPNesterov = dP + vNesterov;
            vNesterov.dispose();
            dP = dPNesterov;
          } else {
            dP = v;
          }
        }

        // p = p - lr * dP
        final step = dP * lr;
        final newP = p - step;
        newP.buffer.copyToBuffer(p.buffer, p.byteSize);

        step.dispose();
        newP.dispose();
        dPDecayed?.dispose();
      }
    });
  }

  @override
  void dispose() {
    for (final v in _velocity.values) {
      v.dispose();
    }
    _velocity.clear();
  }
}

/// Adam optimizer (Adaptive Moment Estimation).
class Adam extends Optimizer {
  final double beta1;
  final double beta2;
  final double eps;
  final double weightDecay;

  int _stepCount = 0;
  final Map<GpuArray, GpuArray> _m = {};
  final Map<GpuArray, GpuArray> _v = {};

  Adam(
    super.params, {
    required super.lr,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
    this.weightDecay = 0.0,
  });

  @override
  void step() {
    no_grad(() {
      _stepCount++;
      final biasCorrection1 = 1.0 - math.pow(beta1, _stepCount);
      final biasCorrection2 = 1.0 - math.pow(beta2, _stepCount);

      for (final p in params) {
        final grad = p.grad;
        if (grad == null) continue;

        var g = grad;
        GpuArray? gDecayed;
        if (weightDecay != 0.0) {
          final pDecay = p * weightDecay;
          gDecayed = g + pDecay;
          pDecay.dispose();
          g = gDecayed;
        }

        var m = _m[p];
        if (m == null) {
          m = GpuArray.zeros(p.shape, p.dtype, device: p.device);
          _m[p] = m;
        }

        var v = _v[p];
        if (v == null) {
          v = GpuArray.zeros(p.shape, p.dtype, device: p.device);
          _v[p] = v;
        }

        // m = beta1 * m + (1 - beta1) * g
        final mScaled = m * beta1;
        final gScaledM = g * (1.0 - beta1);
        final newM = mScaled + gScaledM;
        mScaled.dispose();
        gScaledM.dispose();
        newM.buffer.copyToBuffer(m.buffer, m.byteSize);
        newM.dispose();

        // v = beta2 * v + (1 - beta2) * (g * g)
        final vScaled = v * beta2;
        final gSq = g * g;
        final gSqScaled = gSq * (1.0 - beta2);
        final newV = vScaled + gSqScaled;
        vScaled.dispose();
        gSq.dispose();
        gSqScaled.dispose();
        newV.buffer.copyToBuffer(v.buffer, v.byteSize);
        newV.dispose();

        // update = (m / biasCorrection1) / (sqrt(v / biasCorrection2) + eps)
        final mHat = m * (1.0 / biasCorrection1);
        final vHat = v * (1.0 / biasCorrection2);
        final vHatSqrt = vHat.sqrt();
        final denom = vHatSqrt + eps;
        final stepDir = mHat / denom;
        final step = stepDir * lr;
        final newP = p - step;
        newP.buffer.copyToBuffer(p.buffer, p.byteSize);

        mHat.dispose();
        vHat.dispose();
        vHatSqrt.dispose();
        denom.dispose();
        stepDir.dispose();
        step.dispose();
        newP.dispose();
        gDecayed?.dispose();
      }
    });
  }

  @override
  void dispose() {
    for (final m in _m.values) {
      m.dispose();
    }
    for (final v in _v.values) {
      v.dispose();
    }
    _m.clear();
    _v.clear();
  }
}

/// AdamW optimizer (Decoupled Weight Decay Adam).
class AdamW extends Optimizer {
  final double beta1;
  final double beta2;
  final double eps;
  final double weightDecay;

  int _stepCount = 0;
  final Map<GpuArray, GpuArray> _m = {};
  final Map<GpuArray, GpuArray> _v = {};

  AdamW(
    super.params, {
    required super.lr,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
    this.weightDecay = 0.01,
  });

  @override
  void step() {
    no_grad(() {
      _stepCount++;
      final biasCorrection1 = 1.0 - math.pow(beta1, _stepCount);
      final biasCorrection2 = 1.0 - math.pow(beta2, _stepCount);

      for (final p in params) {
        final grad = p.grad;
        if (grad == null) continue;

        // Weight decay step decoupled from gradient update
        if (weightDecay != 0.0) {
          final pDecayed = p * (1.0 - lr * weightDecay);
          pDecayed.buffer.copyToBuffer(p.buffer, p.byteSize);
          pDecayed.dispose();
        }

        var m = _m[p];
        if (m == null) {
          m = GpuArray.zeros(p.shape, p.dtype, device: p.device);
          _m[p] = m;
        }

        var v = _v[p];
        if (v == null) {
          v = GpuArray.zeros(p.shape, p.dtype, device: p.device);
          _v[p] = v;
        }

        final mScaled = m * beta1;
        final gScaledM = grad * (1.0 - beta1);
        final newM = mScaled + gScaledM;
        mScaled.dispose();
        gScaledM.dispose();
        newM.buffer.copyToBuffer(m.buffer, m.byteSize);
        newM.dispose();

        final vScaled = v * beta2;
        final gSq = grad * grad;
        final gSqScaled = gSq * (1.0 - beta2);
        final newV = vScaled + gSqScaled;
        vScaled.dispose();
        gSq.dispose();
        gSqScaled.dispose();
        newV.buffer.copyToBuffer(v.buffer, v.byteSize);
        newV.dispose();

        final mHat = m * (1.0 / biasCorrection1);
        final vHat = v * (1.0 / biasCorrection2);
        final vHatSqrt = vHat.sqrt();
        final denom = vHatSqrt + eps;
        final stepDir = mHat / denom;
        final step = stepDir * lr;
        final newP = p - step;
        newP.buffer.copyToBuffer(p.buffer, p.byteSize);

        mHat.dispose();
        vHat.dispose();
        vHatSqrt.dispose();
        denom.dispose();
        stepDir.dispose();
        step.dispose();
        newP.dispose();
      }
    });
  }

  @override
  void dispose() {
    for (final m in _m.values) {
      m.dispose();
    }
    for (final v in _v.values) {
      v.dispose();
    }
    _m.clear();
    _v.clear();
  }
}
