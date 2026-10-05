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

import '../autograd/autograd.dart';
import '../gpu_array.dart';
import 'nn_wgsl.dart';

/// Base class for all neural network parameter optimizers.
abstract class Optimizer {
  /// Trainable parameters managed by this optimizer.
  final List<GpuArray<DTypeTag>> params;

  /// Learning rate step size.
  double lr;

  bool _isDisposed = false;

  /// Creates an [Optimizer] for [params] with learning rate [lr].
  ///
  /// The [lr] must be non-negative.
  Optimizer(List<GpuArray<DTypeTag>> params, {required this.lr})
    : params = List<GpuArray<DTypeTag>>.unmodifiable(params) {
    if (lr < 0.0 || lr.isNaN) {
      throw ArgumentError.value(lr, 'lr', 'Must be non-negative.');
    }
  }

  /// Whether this optimizer and its internal state buffers have been disposed.
  bool get isDisposed => _isDisposed;

  void _checkNotDisposed() {
    if (_isDisposed) {
      throw StateError('Cannot use $runtimeType after it has been disposed.');
    }
  }

  /// Performs a single optimization step updating all [params].
  ///
  /// This optimizer must not be disposed ([isDisposed] must be `false`).
  void step();

  /// Clears the gradients of all optimized [params].
  ///
  /// This optimizer must not be disposed ([isDisposed] must be `false`).
  void zeroGrad() {
    _checkNotDisposed();
    for (final parameter in params) {
      parameter.zeroGrad();
    }
  }

  /// Disposes internal optimizer state buffers.
  void dispose() {
    _isDisposed = true;
  }
}

/// Stochastic Gradient Descent (SGD) optimizer with optional momentum, weight decay, and Nesterov acceleration.
final class SGD extends Optimizer {
  /// Momentum factor ($\mu \ge 0$).
  final double momentum;

  /// L2 weight decay penalty ($\lambda \ge 0$).
  final double weightDecay;

  /// Whether Nesterov momentum is enabled.
  final bool nesterov;

  final Map<GpuArray<DTypeTag>, GpuArray<DTypeTag>> _velocity = {};

  /// Creates an [SGD] optimizer.
  ///
  /// Both [momentum] and [weightDecay] must be non-negative, and [nesterov]
  /// requires a positive [momentum].
  SGD(
    super.params, {
    required super.lr,
    this.momentum = 0.0,
    this.weightDecay = 0.0,
    this.nesterov = false,
  }) {
    if (momentum < 0.0 || momentum.isNaN) {
      throw ArgumentError.value(momentum, 'momentum', 'Must be non-negative.');
    }
    if (weightDecay < 0.0 || weightDecay.isNaN) {
      throw ArgumentError.value(
        weightDecay,
        'weightDecay',
        'Must be non-negative.',
      );
    }
    if (nesterov && momentum <= 0.0) {
      throw ArgumentError.value(
        nesterov,
        'nesterov',
        'Must have positive momentum when Nesterov momentum is enabled.',
      );
    }
  }

  @override
  void step() {
    _checkNotDisposed();
    noGrad(() {
      for (final parameter in params) {
        final grad = parameter.grad;
        if (grad == null) continue;

        GpuArray<DTypeTag>? velocity;
        if (momentum != 0.0) {
          velocity = _velocity[parameter] ??= GpuArray.zeros(
            parameter.shape,
            parameter.dtype,
            device: parameter.device,
          )..detachFromScope();
        }

        final contiguousGrad = (grad.isContiguous && grad.offsetElements == 0)
            ? grad
            : grad.copy();
        try {
          dispatchSgdStep(
            parameter: parameter,
            grad: contiguousGrad,
            velocity: velocity,
            lr: lr,
            momentum: momentum,
            weightDecay: weightDecay,
            nesterov: nesterov,
          );
        } finally {
          if (!identical(contiguousGrad, grad)) {
            contiguousGrad.dispose();
          }
        }
      }
    });
  }

  @override
  void dispose() {
    if (isDisposed) return;
    for (final velocity in _velocity.values) {
      velocity.dispose();
    }
    _velocity.clear();
    super.dispose();
  }
}

/// Adam optimizer (Adaptive Moment Estimation).
final class Adam extends Optimizer {
  /// Exponential decay rate for first moment estimates ($\beta_1 \in [0, 1)$).
  final double beta1;

  /// Exponential decay rate for second moment estimates ($\beta_2 \in [0, 1)$).
  final double beta2;

  /// Term added to the denominator to improve numerical stability ($\epsilon > 0$).
  final double eps;

  /// L2 weight decay penalty ($\lambda \ge 0$).
  final double weightDecay;

  int _stepCount = 0;
  final Map<GpuArray<DTypeTag>, GpuArray<DTypeTag>> _firstMoment = {};
  final Map<GpuArray<DTypeTag>, GpuArray<DTypeTag>> _secondMoment = {};

  /// Creates an [Adam] optimizer.
  Adam(
    super.params, {
    required super.lr,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
    this.weightDecay = 0.0,
  }) {
    if (beta1 < 0.0 || beta1 >= 1.0 || beta1.isNaN) {
      throw ArgumentError.value(
        beta1,
        'beta1',
        'Must be in the half-open interval [0.0, 1.0).',
      );
    }
    if (beta2 < 0.0 || beta2 >= 1.0 || beta2.isNaN) {
      throw ArgumentError.value(
        beta2,
        'beta2',
        'Must be in the half-open interval [0.0, 1.0).',
      );
    }
    if (eps <= 0.0 || eps.isNaN) {
      throw ArgumentError.value(eps, 'eps', 'Must be positive.');
    }
    if (weightDecay < 0.0 || weightDecay.isNaN) {
      throw ArgumentError.value(
        weightDecay,
        'weightDecay',
        'Must be non-negative.',
      );
    }
  }

  @override
  void step() {
    _checkNotDisposed();
    noGrad(() {
      _stepCount++;
      for (final parameter in params) {
        final grad = parameter.grad;
        if (grad == null) continue;

        final firstMoment = _firstMoment[parameter] ??= GpuArray.zeros(
          parameter.shape,
          parameter.dtype,
          device: parameter.device,
        )..detachFromScope();
        final secondMoment = _secondMoment[parameter] ??= GpuArray.zeros(
          parameter.shape,
          parameter.dtype,
          device: parameter.device,
        )..detachFromScope();

        final contiguousGrad = (grad.isContiguous && grad.offsetElements == 0)
            ? grad
            : grad.copy();
        final invBiasCorrection1 = 1.0 / (1.0 - math.pow(beta1, _stepCount));
        final invBiasCorrection2 = 1.0 / (1.0 - math.pow(beta2, _stepCount));
        try {
          dispatchAdamStep(
            parameter: parameter,
            grad: contiguousGrad,
            firstMoment: firstMoment,
            secondMoment: secondMoment,
            lr: lr,
            beta1: beta1,
            beta2: beta2,
            eps: eps,
            weightDecay: weightDecay,
            invBiasCorrection1: invBiasCorrection1,
            invBiasCorrection2: invBiasCorrection2,
            decoupledWeightDecay: false,
          );
        } finally {
          if (!identical(contiguousGrad, grad)) {
            contiguousGrad.dispose();
          }
        }
      }
    });
  }

  @override
  void dispose() {
    if (isDisposed) return;
    for (final m in _firstMoment.values) {
      m.dispose();
    }
    for (final v in _secondMoment.values) {
      v.dispose();
    }
    _firstMoment.clear();
    _secondMoment.clear();
    super.dispose();
  }
}

/// AdamW optimizer (Adam with decoupled weight decay regularization).
final class AdamW extends Optimizer {
  /// Exponential decay rate for first moment estimates ($\beta_1 \in [0, 1)$).
  final double beta1;

  /// Exponential decay rate for second moment estimates ($\beta_2 \in [0, 1)$).
  final double beta2;

  /// Term added to the denominator to improve numerical stability ($\epsilon > 0$).
  final double eps;

  /// Decoupled weight decay coefficient ($\lambda \ge 0$).
  final double weightDecay;

  int _stepCount = 0;
  final Map<GpuArray<DTypeTag>, GpuArray<DTypeTag>> _firstMoment = {};
  final Map<GpuArray<DTypeTag>, GpuArray<DTypeTag>> _secondMoment = {};

  /// Creates an [AdamW] optimizer.
  AdamW(
    super.params, {
    required super.lr,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
    this.weightDecay = 0.01,
  }) {
    if (beta1 < 0.0 || beta1 >= 1.0 || beta1.isNaN) {
      throw ArgumentError.value(
        beta1,
        'beta1',
        'Must be in the half-open interval [0.0, 1.0).',
      );
    }
    if (beta2 < 0.0 || beta2 >= 1.0 || beta2.isNaN) {
      throw ArgumentError.value(
        beta2,
        'beta2',
        'Must be in the half-open interval [0.0, 1.0).',
      );
    }
    if (eps <= 0.0 || eps.isNaN) {
      throw ArgumentError.value(eps, 'eps', 'Must be positive.');
    }
    if (weightDecay < 0.0 || weightDecay.isNaN) {
      throw ArgumentError.value(
        weightDecay,
        'weightDecay',
        'Must be non-negative.',
      );
    }
  }

  @override
  void step() {
    _checkNotDisposed();
    noGrad(() {
      _stepCount++;
      for (final parameter in params) {
        final grad = parameter.grad;
        if (grad == null) continue;

        final firstMoment = _firstMoment[parameter] ??= GpuArray.zeros(
          parameter.shape,
          parameter.dtype,
          device: parameter.device,
        )..detachFromScope();
        final secondMoment = _secondMoment[parameter] ??= GpuArray.zeros(
          parameter.shape,
          parameter.dtype,
          device: parameter.device,
        )..detachFromScope();

        final contiguousGrad = (grad.isContiguous && grad.offsetElements == 0)
            ? grad
            : grad.copy();
        final invBiasCorrection1 = 1.0 / (1.0 - math.pow(beta1, _stepCount));
        final invBiasCorrection2 = 1.0 / (1.0 - math.pow(beta2, _stepCount));
        try {
          dispatchAdamStep(
            parameter: parameter,
            grad: contiguousGrad,
            firstMoment: firstMoment,
            secondMoment: secondMoment,
            lr: lr,
            beta1: beta1,
            beta2: beta2,
            eps: eps,
            weightDecay: weightDecay,
            invBiasCorrection1: invBiasCorrection1,
            invBiasCorrection2: invBiasCorrection2,
            decoupledWeightDecay: true,
          );
        } finally {
          if (!identical(contiguousGrad, grad)) {
            contiguousGrad.dispose();
          }
        }
      }
    });
  }

  @override
  void dispose() {
    if (isDisposed) return;
    for (final m in _firstMoment.values) {
      m.dispose();
    }
    for (final v in _secondMoment.values) {
      v.dispose();
    }
    _firstMoment.clear();
    _secondMoment.clear();
    super.dispose();
  }
}

/// Clips the total gradient norm of an iterable of [parameters] in place.
///
/// Computes the total $L_p$ norm (specified by [normType]) across all non-null
/// parameter gradients, scales every gradient in place by
/// `maxNorm / (totalNorm + 1e-6)` whenever `totalNorm > maxNorm`, and evaluates
/// to the pre-clipped total norm.
///
/// Both [maxNorm] and [normType] must be positive.
double clipGradNorm(
  Iterable<GpuArray<DTypeTag>> parameters,
  double maxNorm, {
  double normType = 2.0,
}) {
  if (maxNorm <= 0.0 || maxNorm.isNaN) {
    throw ArgumentError.value(maxNorm, 'maxNorm', 'Must be positive.');
  }
  if (normType <= 0.0 || normType.isNaN) {
    throw ArgumentError.value(normType, 'normType', 'Must be positive.');
  }

  return noGrad(() {
    final activeGrads = <GpuArray<DTypeTag>>[];
    for (final parameter in parameters) {
      final grad = parameter.grad;
      if (grad != null && grad.size > 0) {
        activeGrads.add(grad);
      }
    }
    if (activeGrads.isEmpty) return 0.0;

    var totalNorm = 0.0;
    if (normType == double.infinity) {
      for (final grad in activeGrads) {
        final absGrad = grad.abs();
        final maxGrad = absGrad.max();
        final gradMaxVal = (maxGrad.scalar as num).toDouble();
        maxGrad.dispose();
        absGrad.dispose();
        if (gradMaxVal > totalNorm) {
          totalNorm = gradMaxVal;
        }
      }
    } else if (normType == 2.0) {
      var sumSquares = 0.0;
      for (final grad in activeGrads) {
        final sq = grad * grad;
        final sqSum = sq.sum();
        sumSquares += (sqSum.scalar as num).toDouble();
        sqSum.dispose();
        sq.dispose();
      }
      totalNorm = math.sqrt(sumSquares);
    } else {
      var sumPowers = 0.0;
      for (final grad in activeGrads) {
        final absGrad = grad.abs();
        final powGrad = absGrad.pow(normType);
        final powSum = powGrad.sum();
        sumPowers += (powSum.scalar as num).toDouble();
        powSum.dispose();
        powGrad.dispose();
        absGrad.dispose();
      }
      totalNorm = math.pow(sumPowers, 1.0 / normType).toDouble();
    }

    if (totalNorm > maxNorm) {
      final clipCoefficient = maxNorm / (totalNorm + 1e-6);
      for (final grad in activeGrads) {
        final scaled = grad * clipCoefficient;
        scaled.buffer.copyToBuffer(grad.buffer, grad.byteSize);
        scaled.dispose();
      }
    }

    return totalNorm;
  });
}

/// Clips all non-null gradients of [parameters] elementwise to
/// `[-clipValue, clipValue]` in place.
///
/// The [clipValue] must be non-negative.
void clipGradValue(Iterable<GpuArray<DTypeTag>> parameters, double clipValue) {
  if (clipValue < 0.0 || clipValue.isNaN) {
    throw ArgumentError.value(clipValue, 'clipValue', 'Must be non-negative.');
  }

  noGrad(() {
    for (final parameter in parameters) {
      final grad = parameter.grad;
      if (grad == null || grad.size == 0) continue;
      grad.clip(-clipValue, clipValue, out: grad);
    }
  });
}
