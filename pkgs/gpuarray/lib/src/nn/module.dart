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

import '../gpu_array.dart';
import '../device.dart';

/// Base class for all neural network modules.
abstract class Module {
  bool _isTraining = true;

  /// Whether the module is currently in training mode (affects Dropout, BatchNorm, etc.).
  bool get isTraining => _isTraining;

  /// Sets the module in training mode.
  void train([bool mode = true]) {
    _isTraining = mode;
    for (final child in _submodules) {
      child.train(mode);
    }
  }

  /// Sets the module in evaluation mode.
  void eval() => train(false);

  /// Submodules registered under this module.
  final List<Module> _submodules = [];

  /// Explicitly registered parameters.
  final List<GpuArray> _parameters = [];

  /// Explicitly registered named parameters.
  final Map<String, GpuArray> _namedParams = {};

  /// Registers a trainable parameter.
  T registerParameter<T extends GpuArray>(String name, T param) {
    _parameters.add(param);
    _namedParams[name] = param;
    return param;
  }

  /// Registers a child submodule.
  T registerModule<T extends Module>(T module) {
    _submodules.add(module);
    return module;
  }

  /// Returns all trainable parameters of this module and its submodules.
  List<GpuArray> parameters() {
    final list = <GpuArray>[..._parameters];
    for (final child in _submodules) {
      list.addAll(child.parameters());
    }
    return list;
  }

  /// Returns all named parameters of this module and its submodules.
  Map<String, GpuArray> namedParameters({String prefix = ''}) {
    final map = <String, GpuArray>{};
    for (final entry in _namedParams.entries) {
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

  /// Zeroes out the gradients of all parameters.
  void zeroGrad() {
    for (final p in parameters()) {
      p.zeroGrad();
    }
  }

  /// Moves all parameters to [device].
  void to(GpuDevice device) {
    // Parameters on GPU stay on device
  }

  /// Defines the computation performed at every call.
  GpuArray forward(GpuArray input);

  /// Callable invocation executing [forward].
  GpuArray call(GpuArray input) => forward(input);
}

/// A sequential container passing the output of each submodule as input to the next.
class Sequential extends Module {
  final List<Module> layers;

  Sequential(this.layers) {
    for (final layer in layers) {
      registerModule(layer);
    }
  }

  @override
  GpuArray forward(GpuArray input) {
    var current = input;
    for (final layer in layers) {
      current = layer.forward(current);
    }
    return current;
  }
}
