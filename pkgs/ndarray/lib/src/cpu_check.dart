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

// ignore_for_file: non_constant_identifier_names
@ffi.DefaultAsset('package:ndarray/ndarray_cpu_check')
library;

import 'dart:ffi' as ffi;

import 'package:meta/meta.dart';

/// Bitmask flag for CPUID leaf 1 ECX bit 27 (`OSXSAVE`).
@internal
const int x86FeatureOsxsave = 0x01;

/// Bitmask flag for `XGETBV(0)` XMM+YMM OS state support (`(xcr0 & 0x6) == 0x6`).
@internal
const int x86FeatureYmmState = 0x02;

/// Bitmask flag for CPUID leaf 1 ECX bit 28 (`AVX`).
@internal
const int x86FeatureAvx = 0x04;

/// Bitmask flag for CPUID leaf 1 ECX bit 12 (`FMA`).
@internal
const int x86FeatureFma = 0x08;

/// Bitmask flag for CPUID leaf 1 ECX bit 29 (`F16C`).
@internal
const int x86FeatureF16c = 0x10;

/// Bitmask flag for CPUID leaf 7 subleaf 0 EBX bit 5 (`AVX2`).
@internal
const int x86FeatureAvx2 = 0x20;

/// Combined bitmask of all x86-64 ISA features required by default builds.
@internal
const int x86FeatureAllDefault =
    x86FeatureOsxsave |
    x86FeatureYmmState |
    x86FeatureAvx |
    x86FeatureFma |
    x86FeatureF16c |
    x86FeatureAvx2;

@ffi.Native<ffi.Int32 Function()>()
external int ndarray_x86_cpu_features();

@ffi.Native<ffi.Int32 Function()>()
external int ndarray_x86_required_features();

bool _cpuChecked = false;

/// Verifies that the host CPU supports all ISA extensions required by the
/// compiled `ndarray` native library before `libndarray` is loaded.
///
/// Throws an [UnsupportedError] if any required CPU feature is missing.
@internal
void ensureCpuSupported() {
  if (_cpuChecked) return;
  final int requiredFeatures;
  final int actualFeatures;
  try {
    requiredFeatures = ndarray_x86_required_features();
    if (requiredFeatures == 0) {
      _cpuChecked = true;
      return;
    }
    actualFeatures = ndarray_x86_cpu_features();
  } on Object catch (e) {
    if (e is! ArgumentError) rethrow;
    // If the optional ndarray_cpu_check asset was not bundled (e.g. a
    // prebuilt fetch on a host without a C compiler), skip the pre-check.
    _cpuChecked = true;
    return;
  }
  verifyCpuFeatureMask(
    actualFeatures: actualFeatures,
    requiredFeatures: requiredFeatures,
  );
  _cpuChecked = true;
}

/// Validates that [actualFeatures] satisfies all bits in [requiredFeatures].
///
/// Throws an [UnsupportedError] naming the missing features and how to
/// configure a baseline source build via `hooks.user_defines.ndarray`
/// (`buildMode: source`, `x86Flags: ""`).
@internal
void verifyCpuFeatureMask({
  required int actualFeatures,
  int requiredFeatures = x86FeatureAllDefault,
}) {
  final missing = requiredFeatures & ~actualFeatures;
  if (missing == 0) return;

  final missingNames = <String>[
    if ((missing & x86FeatureOsxsave) != 0) 'OSXSAVE',
    if ((missing & x86FeatureYmmState) != 0) 'OS AVX/YMM state (XGETBV)',
    if ((missing & x86FeatureAvx) != 0) 'AVX',
    if ((missing & x86FeatureAvx2) != 0) 'AVX2',
    if ((missing & x86FeatureFma) != 0) 'FMA',
    if ((missing & x86FeatureF16c) != 0) 'F16C',
  ];

  throw UnsupportedError(
    'The host x86-64 CPU does not support instruction set extensions '
    'required by this build of package:ndarray '
    '(missing: ${missingNames.join(', ')}).\n'
    'To compile package:ndarray from source for baseline x86-64 CPUs '
    'without AVX2/FMA/F16C, add the following to your workspace '
    'pubspec.yaml:\n'
    'hooks:\n'
    '  user_defines:\n'
    '    ndarray:\n'
    '      buildMode: source\n'
    '      x86Flags: ""\n'
    '(or set environment variables NDARRAY_BUILD_MODE=source and '
    'NDARRAY_X86_FLAGS="").',
  );
}
