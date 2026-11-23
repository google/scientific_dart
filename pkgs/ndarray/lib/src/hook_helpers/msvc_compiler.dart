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

import 'dart:io';

import 'package:code_assets/code_assets.dart';

/// Process invocation callback used by [resolveWindowsCppCompiler] so Windows
/// compiler probing can be unit-tested on any host platform.
typedef CompilerProcessRunner =
    Future<ProcessResult> Function(
      String executable,
      List<String> arguments, {
      String? workingDirectory,
      Map<String, String>? environment,
    });

void _ignoreLog(String _) {}

String? _envValue(Map<String, String> environment, String name) {
  final target = name.toLowerCase();
  for (final entry in environment.entries) {
    if (entry.key.toLowerCase() == target) {
      final trimmed = entry.value.trim();
      if (trimmed.isNotEmpty) return trimmed;
    }
  }
  return null;
}

String _joinWindowsPath(String base, String relative) {
  final trimmed = base.replaceFirst(RegExp(r'[\\/]+$'), '');
  return '$trimmed\\$relative';
}

/// Returns candidate `clang-cl.exe` executables for [arch], ordered so Visual
/// Studio's MSVC-matched LLVM toolchain takes precedence over standalone LLVM
/// installations and `PATH`.
List<String> windowsClangClCandidates(
  Architecture arch, {
  required Map<String, String> environment,
}) {
  final candidates = <String>[];
  final seen = <String>{};

  void addCandidate(String path) {
    final key = path.replaceAll('/', r'\').toLowerCase();
    if (seen.add(key)) {
      candidates.add(path);
    }
  }

  final vsLlvmEnvVar = switch (arch) {
    Architecture.arm64 => 'llvmArm64',
    Architecture.x64 => 'llvmX64',
    _ => null,
  };
  if (vsLlvmEnvVar != null) {
    if (_envValue(environment, vsLlvmEnvVar) case final vsLlvmDir?) {
      addCandidate(_joinWindowsPath(vsLlvmDir, 'clang-cl.exe'));
    }
  }

  if (_envValue(environment, 'VCINSTALLDIR') case final vcInstallDir?) {
    final relativeLlvmBin = switch (arch) {
      Architecture.arm64 => r'Tools\Llvm\ARM64\bin\clang-cl.exe',
      Architecture.x64 => r'Tools\Llvm\x64\bin\clang-cl.exe',
      _ => r'Tools\Llvm\bin\clang-cl.exe',
    };
    addCandidate(_joinWindowsPath(vcInstallDir, relativeLlvmBin));
  }

  final programFiles =
      _envValue(environment, 'ProgramFiles') ?? r'C:\Program Files';
  addCandidate(_joinWindowsPath(programFiles, r'LLVM\bin\clang-cl.exe'));
  addCandidate('clang-cl.exe');

  return candidates;
}

/// Returns a minimal C++17 source snippet that verifies MSVC STL `<atomic>`
/// compatibility (such as `STL1000` in `yvals_core.h`) and target architecture
/// macros for [arch].
String windowsCompilerProbeSource(Architecture arch) {
  final archGuard = switch (arch) {
    Architecture.x64 =>
      '#if !defined(_M_X64)\n#error "Expected x64 target (_M_X64)"\n#endif\n',
    Architecture.arm64 =>
      '#if !defined(_M_ARM64)\n#error "Expected arm64 target (_M_ARM64)"\n#endif\n',
    Architecture.ia32 =>
      '#if !defined(_M_IX86)\n#error "Expected x86 target (_M_IX86)"\n#endif\n',
    _ => '',
  };
  return '''
$archGuard#include <atomic>
#include <cstdint>
std::atomic<uint8_t> probe_n8(0);
std::atomic<uint64_t> probe_n64(0);
int main(void) {
  ++probe_n8;
  ++probe_n64;
  return static_cast<int>(probe_n8.load() + probe_n64.load() - 2);
}
''';
}

String _summarizeProbeFailure(ProcessResult result) {
  final combined = '${result.stderr}\n${result.stdout}';
  String? firstNonEmpty;
  for (final rawLine in combined.split('\n')) {
    final line = rawLine.trim();
    if (line.isEmpty) continue;
    firstNonEmpty ??= line;
    if (line.toLowerCase().contains('error')) {
      return line;
    }
  }
  return firstNonEmpty ?? '';
}

/// Selects a Windows C++ compiler compatible with the active MSVC STL in
/// [environment] for [arch].
///
/// Tries [windowsClangClCandidates] in order, probing each candidate by
/// compiling and linking [windowsCompilerProbeSource] with [environment] (and
/// [x86Flags] on [Architecture.x64]). Returns the first candidate that
/// succeeds, or falls back to [fallbackCompiler] (typically `cl.exe`) when no
/// compatible `clang-cl.exe` is available.
Future<String> resolveWindowsCppCompiler({
  required String fallbackCompiler,
  required Architecture arch,
  required Map<String, String> environment,
  List<String> x86Flags = const <String>[],
  CompilerProcessRunner runProcess = Process.run,
  Directory? probeParentDir,
  void Function(String message) onLog = _ignoreLog,
}) async {
  final candidates = windowsClangClCandidates(arch, environment: environment);
  final parentDir = probeParentDir ?? Directory.systemTemp;
  final probeDir = parentDir.createTempSync('ndarray_clang_probe_');
  try {
    final probeSrc = File.fromUri(probeDir.uri.resolve('probe.cpp'));
    probeSrc.writeAsStringSync(windowsCompilerProbeSource(arch));

    final baseArgs = <String>[
      '/nologo',
      '/std:c++17',
      '/MD',
      '/EHsc',
      if (arch == Architecture.x64) ...x86Flags,
      'probe.cpp',
    ];

    for (var i = 0; i < candidates.length; i++) {
      final candidate = candidates[i];
      final probeArgs = <String>[
        ...baseArgs,
        '/Fo:probe_$i.obj',
        '/Fe:probe_$i.exe',
      ];
      try {
        final result = await runProcess(
          candidate,
          probeArgs,
          workingDirectory: probeDir.path,
          environment: environment,
        );
        if (result.exitCode == 0) {
          onLog('Using Windows C++ compiler: $candidate');
          return candidate;
        }
        final detail = _summarizeProbeFailure(result);
        final suffix = detail.isEmpty ? '' : ': $detail';
        onLog(
          'Skipping incompatible Windows C++ compiler $candidate '
          '(exit ${result.exitCode}$suffix).',
        );
      } on ProcessException {
        // Candidate executable is absent or not runnable on this host.
      }
    }
  } finally {
    try {
      probeDir.deleteSync(recursive: true);
    } on FileSystemException {
      // Best-effort cleanup; AV scanners on Windows may transiently lock probe.exe.
    }
  }

  onLog(
    'No compatible clang-cl found; falling back to Windows C++ compiler: '
    '$fallbackCompiler',
  );
  return fallbackCompiler;
}
