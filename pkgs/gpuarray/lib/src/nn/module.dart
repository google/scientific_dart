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

import '../device.dart';
import '../gpu_array.dart';

/// Base class for all neural network modules.
abstract class Module {
  bool _isTraining = true;

  /// Whether the module is currently in training mode.
  bool get isTraining => _isTraining;

  /// Whether the module is currently in training mode.
  ///
  /// Alias for [isTraining] for PyTorch parity.
  bool get training => _isTraining;

  /// Sets the module and all registered child submodules into training mode when [mode] is `true`,
  /// or evaluation mode when [mode] is `false`.
  void train({bool mode = true}) {
    _isTraining = mode;
    for (final child in _submodules) {
      child.train(mode: mode);
    }
  }

  /// Sets the module and all registered child submodules into evaluation mode.
  void eval() => train(mode: false);

  /// Submodules registered under this module.
  final List<Module> _submodules = [];

  /// Explicitly registered trainable parameters.
  final List<GpuArray<DTypeTag>> _parameters = [];

  /// Explicitly registered named trainable parameters.
  final Map<String, GpuArray<DTypeTag>> _namedParameters = {};

  /// Registers a trainable [parameter] under [name] and returns it.
  T registerParameter<T extends GpuArray<DTypeTag>>(String name, T parameter) {
    if (name.isEmpty) {
      throw ArgumentError.value(name, 'name', 'Must not be empty.');
    }
    _parameters.add(parameter);
    _namedParameters[name] = parameter;
    return parameter;
  }

  /// Registers a child [module] and returns it.
  T registerModule<T extends Module>(T module) {
    _submodules.add(module);
    return module;
  }

  /// All trainable parameters of this module and its recursive submodules.
  List<GpuArray<DTypeTag>> get parameters {
    final collected = <GpuArray<DTypeTag>>[..._parameters];
    for (final child in _submodules) {
      collected.addAll(child.parameters);
    }
    return List<GpuArray<DTypeTag>>.unmodifiable(collected);
  }

  /// Collects a map of all named parameters of this module and its submodules.
  Map<String, GpuArray<DTypeTag>> namedParameters({String prefix = ''}) {
    final map = <String, GpuArray<DTypeTag>>{};
    for (final entry in _namedParameters.entries) {
      final key = prefix.isEmpty ? entry.key : '$prefix.${entry.key}';
      map[key] = entry.value;
    }
    for (var i = 0; i < _submodules.length; i++) {
      final child = _submodules[i];
      final childPrefix = prefix.isEmpty ? '$i' : '$prefix.$i';
      map.addAll(child.namedParameters(prefix: childPrefix));
    }
    return map;
  }

  /// Clears accumulated gradients on all parameters of this module and its submodules.
  void zeroGrad() {
    for (final parameter in parameters) {
      parameter.zeroGrad();
    }
  }

  /// Moves module parameters to [device].
  void to(GpuDevice device) {
    // Parameters allocated on GPU remain resident on the target device.
  }

  /// Computes the forward pass of this module for [input].
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input);

  /// Invokes [forward] on [input].
  GpuArray<DTypeTag> call(GpuArray<DTypeTag> input) => forward(input);
}

/// Sequential container passing the output of each submodule as input to the next.
final class Sequential extends Module {
  /// Ordered list of child modules executed sequentially in [forward].
  final List<Module> layers;

  /// Creates a [Sequential] container from [layers].
  Sequential(List<Module> layers) : layers = List<Module>.unmodifiable(layers) {
    for (final layer in this.layers) {
      registerModule(layer);
    }
  }

  @override
  GpuArray<DTypeTag> forward(GpuArray<DTypeTag> input) {
    var current = input;
    for (final layer in layers) {
      current = layer.forward(current);
    }
    return current;
  }
}
