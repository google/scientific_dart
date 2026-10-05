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

import '../device.dart';
import '../exceptions.dart';
import '../gpu_array.dart';
import '../serialization/safetensors.dart' as safetensors;

/// Base class for all neural network modules.
abstract class Module implements ScopedResource {
  bool _isTraining = true;
  bool _isDisposed = false;

  /// Whether the module is currently in training mode.
  bool get isTraining => _isTraining;

  /// Whether the module is currently in training mode.
  ///
  /// Alias for [isTraining] for PyTorch parity.
  bool get training => _isTraining;

  @override
  bool get isDisposed => _isDisposed;

  /// Verifies that this module has not been disposed.
  void checkNotDisposed() {
    if (_isDisposed) {
      throw StateError('Cannot use $runtimeType after it has been disposed.');
    }
  }

  /// Sets the module and all registered child submodules into training mode when [mode] is `true`,
  /// or evaluation mode when [mode] is `false`.
  void train({bool mode = true}) {
    checkNotDisposed();
    _isTraining = mode;
    for (final child in _submodules) {
      child.train(mode: mode);
    }
  }

  /// Sets the module and all registered child submodules into evaluation mode.
  void eval() => train(mode: false);

  /// Submodules registered under this module.
  final List<Module> _submodules = [];

  /// Named submodules registered under this module.
  final Map<String, Module> _namedSubmodules = {};

  /// Explicitly registered trainable parameters.
  final List<GpuArray<DTypeTag>> _parameters = [];

  /// Explicitly registered named trainable parameters.
  final Map<String, GpuArray<DTypeTag>> _namedParameters = {};

  /// Explicitly registered persistent non-trainable buffers.
  final List<GpuArray<DTypeTag>> _buffers = [];

  /// Explicitly registered named persistent non-trainable buffers.
  final Map<String, GpuArray<DTypeTag>> _namedBuffers = {};

  /// Registers a trainable [parameter] under [name] and returns it.
  T registerParameter<T extends GpuArray<DTypeTag>>(String name, T parameter) {
    if (name.isEmpty) {
      throw ArgumentError.value(name, 'name', 'Must not be empty.');
    }
    _parameters.add(parameter);
    _namedParameters[name] = parameter;
    return parameter;
  }

  /// Registers a persistent non-trainable [buffer] under [name] and returns it.
  GpuArray<T> registerBuffer<T extends DTypeTag>(
    String name,
    GpuArray<T> buffer,
  ) {
    if (name.isEmpty) {
      throw ArgumentError.value(name, 'name', 'Must not be empty.');
    }
    _buffers.add(buffer);
    _namedBuffers[name] = buffer;
    return buffer;
  }

  /// Registers a child [module] under optional [name] and returns it.
  M registerModule<M extends Module>(M module, [String? name]) {
    final effectiveName = name ?? '${_submodules.length}';
    if (effectiveName.isEmpty) {
      throw ArgumentError.value(name, 'name', 'Must not be empty.');
    }
    _submodules.add(module);
    _namedSubmodules[effectiveName] = module;
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
    for (final entry in _namedSubmodules.entries) {
      final childPrefix = prefix.isEmpty ? entry.key : '$prefix.${entry.key}';
      map.addAll(entry.value.namedParameters(prefix: childPrefix));
    }
    return map;
  }

  /// All persistent non-trainable buffers of this module and its recursive submodules.
  List<GpuArray<DTypeTag>> get buffers {
    final collected = <GpuArray<DTypeTag>>[..._buffers];
    for (final child in _submodules) {
      collected.addAll(child.buffers);
    }
    return List<GpuArray<DTypeTag>>.unmodifiable(collected);
  }

  /// Collects a map of all named persistent buffers of this module and its submodules.
  Map<String, GpuArray<DTypeTag>> namedBuffers({String prefix = ''}) {
    final map = <String, GpuArray<DTypeTag>>{};
    for (final entry in _namedBuffers.entries) {
      final key = prefix.isEmpty ? entry.key : '$prefix.${entry.key}';
      map[key] = entry.value;
    }
    for (final entry in _namedSubmodules.entries) {
      final childPrefix = prefix.isEmpty ? entry.key : '$prefix.${entry.key}';
      map.addAll(entry.value.namedBuffers(prefix: childPrefix));
    }
    return map;
  }

  /// Collects a dictionary of all named parameters and persistent buffers of this module and its submodules.
  Map<String, GpuArray<DTypeTag>> stateDict({String prefix = ''}) {
    checkNotDisposed();
    final map = <String, GpuArray<DTypeTag>>{};
    map.addAll(namedParameters(prefix: prefix));
    map.addAll(namedBuffers(prefix: prefix));
    return map;
  }

  /// Copies parameters and buffers from [state] into this module and its submodules in place.
  ///
  /// When [strict] is `true`, the keys of [state] must exactly match [stateDict].
  void loadStateDict(
    Map<String, GpuArray<DTypeTag>> state, {
    bool strict = true,
  }) {
    checkNotDisposed();
    final current = stateDict();
    if (strict) {
      final missingKeys = current.keys
          .where((k) => !state.containsKey(k))
          .toList();
      final unexpectedKeys = state.keys
          .where((k) => !current.containsKey(k))
          .toList();
      if (missingKeys.isNotEmpty || unexpectedKeys.isNotEmpty) {
        throw ArgumentError.value(
          state,
          'state',
          'Must match module stateDict keys (missing: $missingKeys, unexpected: $unexpectedKeys).',
        );
      }
    }

    for (final entry in state.entries) {
      final destination = current[entry.key];
      if (destination == null) continue;
      final source = entry.value;
      if (source.isDisposed) {
        throw StateError(
          'Cannot load disposed tensor "${entry.key}" into stateDict.',
        );
      }
      if (destination.shape.length != source.shape.length ||
          !_shapesEqual(destination.shape, source.shape)) {
        throw GpuShapeMismatchException(
          'loadStateDict(${entry.key})',
          destination.shape,
          source.shape,
        );
      }
    }

    for (final entry in state.entries) {
      final destination = current[entry.key];
      if (destination == null) continue;
      final source = entry.value;
      if (identical(source.buffer, destination.buffer)) continue;

      final sameDevice = identical(source.device, destination.device)
          ? source
          : source.toDevice(destination.device);
      final matchedDType = sameDevice.dtype == destination.dtype
          ? sameDevice
          : sameDevice.astype(destination.dtype);
      final contiguous =
          (matchedDType.isContiguous && matchedDType.offsetElements == 0)
          ? matchedDType
          : matchedDType.copy();

      if (destination.byteSize > 0) {
        contiguous.buffer.copyToBuffer(
          destination.buffer,
          destination.byteSize,
        );
      }

      if (!identical(contiguous, matchedDType)) {
        contiguous.dispose();
      }
      if (!identical(matchedDType, sameDevice)) {
        matchedDType.dispose();
      }
      if (!identical(sameDevice, source)) {
        sameDevice.dispose();
      }
    }
  }

  static bool _shapesEqual(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// Serializes this module's [stateDict] into the binary SafeTensors format.
  Uint8List saveToSafetensors({Map<String, String>? metadata}) {
    checkNotDisposed();
    return safetensors.saveSafetensors(stateDict(), metadata: metadata);
  }

  /// Loads parameters and persistent buffers into this module from binary SafeTensors [bytes].
  void loadFromSafetensors(Uint8List bytes, {bool strict = true}) {
    checkNotDisposed();
    final current = stateDict();
    final targetDevice = current.values.isNotEmpty
        ? current.values.first.device
        : GpuDevice.defaultDevice;
    final loaded = safetensors.loadSafetensors(bytes, device: targetDevice);
    try {
      loadStateDict(loaded, strict: strict);
    } finally {
      for (final tensor in loaded.values) {
        tensor.dispose();
      }
    }
  }

  /// Saves this module's [stateDict] to a `.safetensors` file at [filePath].
  void saveSafetensorsFile(String filePath, {Map<String, String>? metadata}) {
    checkNotDisposed();
    safetensors.saveSafetensorsFile(filePath, stateDict(), metadata: metadata);
  }

  /// Loads parameters and persistent buffers into this module from a `.safetensors` file at [filePath].
  void loadSafetensorsFile(String filePath, {bool strict = true}) {
    checkNotDisposed();
    final current = stateDict();
    final targetDevice = current.values.isNotEmpty
        ? current.values.first.device
        : GpuDevice.defaultDevice;
    final loaded = safetensors.loadSafetensorsFile(
      filePath,
      device: targetDevice,
    );
    try {
      loadStateDict(loaded, strict: strict);
    } finally {
      for (final tensor in loaded.values) {
        tensor.dispose();
      }
    }
  }

  /// Clears accumulated gradients on all parameters of this module and its submodules.
  void zeroGrad() {
    checkNotDisposed();
    for (final parameter in parameters) {
      parameter.zeroGrad();
    }
  }

  /// Moves all parameters and persistent buffers of this module and its submodules to [device] in place.
  void to(GpuDevice device) {
    checkNotDisposed();
    if (device.isDisposed) {
      throw GpuDeviceDisposedException(device.name);
    }
    for (final parameter in _parameters) {
      parameter.moveToDevice(device);
    }
    for (final buffer in _buffers) {
      buffer.moveToDevice(device);
    }
    for (final child in _submodules) {
      child.to(device);
    }
  }

  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    for (final parameter in _parameters) {
      final grad = parameter.grad;
      if (grad != null && !grad.isDisposed) {
        grad.dispose();
      }
      if (!parameter.isDisposed) {
        parameter.dispose();
      }
    }
    for (final buffer in _buffers) {
      if (!buffer.isDisposed) {
        buffer.dispose();
      }
    }
    for (final child in _submodules) {
      child.dispose();
    }
  }

  @override
  ScopedResource detachFromScope() {
    checkNotDisposed();
    for (final parameter in _parameters) {
      parameter.detachFromScope();
    }
    for (final buffer in _buffers) {
      buffer.detachFromScope();
    }
    for (final child in _submodules) {
      child.detachFromScope();
    }
    return this;
  }

  @override
  ScopedResource detachToParentScope() {
    checkNotDisposed();
    for (final parameter in _parameters) {
      parameter.detachToParentScope();
    }
    for (final buffer in _buffers) {
      buffer.detachToParentScope();
    }
    for (final child in _submodules) {
      child.detachToParentScope();
    }
    return this;
  }

  /// Computes the forward pass of this module for [input].
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input);

  /// Invokes [forward] on [input].
  GpuArray<T> call<T extends DTypeTag>(GpuArray<T> input) => forward<T>(input);
}

/// Sequential container passing the output of each submodule as input to the next.
final class Sequential extends Module {
  /// Ordered list of child modules executed sequentially in [forward].
  final List<Module> layers;

  /// Creates a [Sequential] container from [layers].
  Sequential(List<Module> layers) : layers = List<Module>.unmodifiable(layers) {
    for (var i = 0; i < this.layers.length; i++) {
      registerModule(this.layers[i], '$i');
    }
  }

  @override
  GpuArray<T> forward<T extends DTypeTag>(GpuArray<T> input) {
    checkNotDisposed();
    var current = input;
    for (final layer in layers) {
      current = layer.forward<T>(current);
    }
    return current;
  }
}
