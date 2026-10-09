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

import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/dart/element/element.dart';
import 'package:analyzer/dart/element/type.dart';
import 'package:analyzer/source/line_info.dart';
import 'package:test/test.dart';

/// Public barrel libraries whose export namespaces define the package API.
const _barrelLibraries = <String>[
  'lib/gpuarray.dart',
  'lib/linalg.dart',
  'lib/fft.dart',
  'lib/random.dart',
  'lib/autograd.dart',
  'lib/nn.dart',
  'lib/jit.dart',
  'lib/wgsl.dart',
  'lib/serialization.dart',
  'lib/safetensors.dart',
];

/// Whether [tp] is bounded by the wildcard `AnySpec` row (spelled through the
/// alias or expanded).
bool _isAnySpecBound(TypeParameterElement tp) {
  final bound = tp.bound;
  if (bound is! InterfaceType) return false;
  return bound.alias?.element.name == 'AnySpec' ||
      bound.getDisplayString() ==
          'DTypeSpec<DTypeTag, dynamic, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag, DTypeTag>';
}

void main() {
  final pkgRoot = Directory.current.path.endsWith('pkgs/gpuarray')
      ? Directory.current
      : Directory('pkgs/gpuarray');

  List<File> dartFilesIn(String relativeDir) {
    final dir = Directory('${pkgRoot.path}/$relativeDir');
    if (!dir.existsSync()) return [];
    return dir
        .listSync(recursive: true)
        .whereType<File>()
        .where((f) => f.path.endsWith('.dart'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
  }

  final libFiles = dartFilesIn('lib');
  final hookFiles = dartFilesIn('hook');
  final testFiles = dartFilesIn('test');
  final benchmarkFiles = dartFilesIn('benchmark');
  final exampleFiles = dartFilesIn('example');
  final allPackageDartFiles = [
    ...libFiles,
    ...hookFiles,
    ...testFiles,
    ...benchmarkFiles,
    ...exampleFiles,
  ];

  String relPath(File file) =>
      file.path.substring(pkgRoot.path.length + 1).replaceAll(r'\', '/');

  group('Codebase structural & lrn-review invariants', () {
    test('All Dart files have standard Google Apache 2.0 copyright header', () {
      final violations = <String>[];
      for (final file in allPackageDartFiles) {
        final content = file.readAsStringSync();
        if (!content.startsWith('// Copyright 2026 Google LLC\n')) {
          violations.add(
            '${relPath(file)}: missing // Copyright 2026 Google LLC header',
          );
        }
      }
      expect(violations, isEmpty, reason: violations.join('\n'));
    });

    test('No if-else chains dispatching on DType in lib/', () {
      final violations = <String>[];
      final dtypeRegex = RegExp(r'\b[a-zA-Z0-9_.]*dtype\s*(?:==|!=)\s*DType\.');
      for (final file in libFiles) {
        final result = parseString(content: file.readAsStringSync());
        final visitor = _IfDTypeVisitor(
          relPath(file),
          result.lineInfo,
          dtypeRegex,
          violations,
        );
        result.unit.accept(visitor);
      }
      expect(
        violations,
        isEmpty,
        reason:
            'Always use a switch expression/statement to dispatch on DType:\n'
            '${violations.join('\n')}',
      );
    });

    test(
      'Error contract: no bare ArgumentError(), messages start with "Must "',
      () {
        final violations = <String>[];
        for (final file in [...libFiles, ...hookFiles]) {
          final result = parseString(content: file.readAsStringSync());
          final visitor = _ErrorContractVisitor(
            relPath(file),
            result.lineInfo,
            violations,
          );
          result.unit.accept(visitor);
        }
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test(
      'Class modifiers: all concrete classes in lib/ are marked final, base, interface, or sealed',
      () {
        final violations = <String>[];
        for (final file in libFiles) {
          final result = parseString(content: file.readAsStringSync());
          final visitor = _ClassModifierVisitor(
            relPath(file),
            result.lineInfo,
            violations,
          );
          result.unit.accept(visitor);
        }
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test(
      'No namespace-only classes with only static members in lib/ or hook/',
      () {
        final violations = <String>[];
        for (final file in [...libFiles, ...hookFiles]) {
          final result = parseString(content: file.readAsStringSync());
          final visitor = _StaticNamespaceClassVisitor(
            relPath(file),
            result.lineInfo,
            violations,
          );
          result.unit.accept(visitor);
        }
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test(
      'Getters & boolean parameters: no parameterless getX()/findX() or positional bool parameters in public API',
      () {
        final violations = <String>[];
        for (final file in libFiles) {
          final result = parseString(content: file.readAsStringSync());
          final visitor = _GetterAndBoolParamVisitor(
            relPath(file),
            result.lineInfo,
            violations,
          );
          result.unit.accept(visitor);
        }
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test(
      'Dartdoc: all public APIs in lib/ have /// docs and follow Effective Dart / lrn-review rules',
      () {
        final violations = <String>[];
        for (final file in libFiles) {
          final path = relPath(file);
          final result = parseString(content: file.readAsStringSync());
          final visitor = _DartdocVisitor(
            path,
            result.lineInfo,
            violations,
            requirePublicDocs: true,
          );
          result.unit.accept(visitor);
        }
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test('Naming: no prohibited abbreviated words in identifiers in lib/', () {
      final violations = <String>[];
      for (final file in libFiles) {
        final path = relPath(file);
        final result = parseString(content: file.readAsStringSync());
        final visitor = _IdentifierAbbreviationVisitor(
          path,
          result.lineInfo,
          violations,
        );
        result.unit.accept(visitor);
      }
      expect(violations, isEmpty, reason: violations.join('\n'));
    });

    test(
      'No 8-argument DTypeSpec< bounds in lib/ outside allow-listed two-projection linalg functions (Rule 1)',
      () {
        const allowedTwoProjection = <String, Set<String>>{
          'lib/src/linalg/solvers.dart': {'lstsq', 'slogdet'},
          'lib/src/linalg/decompositions.dart': {'svd', 'eigh'},
        };
        final expectedTwoProjectionSites = <String>{
          for (final entry in allowedTwoProjection.entries)
            for (final fn in entry.value) '${entry.key}:$fn',
        };
        final seenTwoProjectionSites = <String>{};
        final violations = <String>[];

        for (final file in libFiles) {
          final path = relPath(file);
          final result = parseString(
            content: file.readAsStringSync(),
            throwIfDiagnostics: false,
          );
          final visitor = _GpuDTypeSpecUsageVisitor(
            filePath: path,
            lineInfo: result.lineInfo,
            allowedTwoProjection: allowedTwoProjection,
            seenTwoProjectionSites: seenTwoProjectionSites,
            violations: violations,
          );
          result.unit.accept(visitor);
        }

        expect(
          violations,
          isEmpty,
          reason:
              '8-argument `DTypeSpec<...>` bounds in gpuarray lib/ are only allowed '
              'on two-projection linalg functions (svd, eigh, lstsq, slogdet):\n'
              '${violations.join('\n')}',
        );
        expect(
          seenTwoProjectionSites,
          equals(expectedTwoProjectionSites),
          reason:
              'Expected visitor to observe all 4 allow-listed two-projection functions '
              'in gpuarray (svd, eigh, lstsq, slogdet).',
        );
      },
    );

    test(
      'Semantic AST analysis: no dynamic in public signatures and defensive copies on collection fields',
      () async {
        final collection = AnalysisContextCollection(
          includedPaths: [
            Directory('${pkgRoot.path}/lib').absolute.path,
            Directory('${pkgRoot.path}/hook').absolute.path,
          ],
        );
        final violations = <String>[];
        final verifiedProjectingFunctions = <String>{};
        final verifiedProjectingExtensions = <String>{};

        for (final context in collection.contexts) {
          for (final filePath in context.contextRoot.analyzedFiles()) {
            if (!filePath.endsWith('.dart')) continue;
            final result = await context.currentSession.getResolvedUnit(
              filePath,
            );
            if (result is! ResolvedUnitResult) continue;
            final relative = filePath
                .substring(pkgRoot.absolute.path.length + 1)
                .replaceAll(r'\', '/');
            final visitor = _SemanticInvariantVisitor(
              relative,
              result.lineInfo,
              violations,
              verifiedProjectingFunctions: verifiedProjectingFunctions,
              verifiedProjectingExtensions: verifiedProjectingExtensions,
            );
            result.unit.accept(visitor);
          }
        }
        final barrels = _barrelLibraries;
        final coreBarrelParsed = parseString(
          content: File('${pkgRoot.path}/lib/gpuarray.dart').readAsStringSync(),
        );
        final forbiddenCoreExportPattern = RegExp(
          r'(?:^|/)(?:nn|linalg|fft|random|jit|wgsl|serialization|safetensors)(?:\.dart|/|$)',
        );
        for (final directive in coreBarrelParsed.unit.directives) {
          if (directive is ExportDirective) {
            final uri = directive.uri.stringValue ?? '';
            if (forbiddenCoreExportPattern.hasMatch(uri)) {
              violations.add(
                'lib/gpuarray.dart: must not export domain module "$uri"',
              );
            }
          }
        }
        final libContext = collection.contextFor(
          Directory('${pkgRoot.path}/lib').absolute.path,
        );
        const forbiddenCoreBarrelSymbols = <String>{
          // Domain symbols that belong exclusively in their domain barrels
          'Module',
          'Linear',
          'Conv2d',
          'Optimizer',
          'SGD',
          'Adam',
          'AdamW',
          'relu',
          'mseLoss',
          'crossEntropy',
          'svd',
          'eigh',
          'rfft',
          'irfft',
          'GpuRandom',
          'SafetensorsFile',
          'WgslJitCompiler',
          'CompiledWgslKernel',
          'Expr',
        };
        const removedLegacyAliases = <String>{
          'no_grad',
          'log_softmax',
          'mse_loss',
          'l1_loss',
          'binary_cross_entropy',
          'cross_entropy',
          'batch_norm_1d',
          'scaled_dot_product_attention',
          'MultiHeadAttention',
          'svdvals',
          'standard_normal',
          'take_along_axis',
          'put_along_axis',
          'column_stack',
          'array_split',
          'expand_dims',
          'broadcast_to',
          'broadcast_arrays',
          'atleast_1d',
          'atleast_2d',
          'atleast_3d',
          'var_',
          'count_nonzero',
          'nan_to_num',
        };
        for (final barrelRel in barrels) {
          final barrelPath = File('${pkgRoot.path}/$barrelRel').absolute.path;
          final libRes = await libContext.currentSession.getResolvedLibrary(
            barrelPath,
          );
          if (libRes is! ResolvedLibraryResult) {
            violations.add('$barrelRel: failed to resolve library');
            continue;
          }
          final exportNames = libRes.element.exportNamespace.definedNames2;
          if (exportNames.isEmpty) {
            violations.add('$barrelRel: exportNamespace is empty');
          }
          for (final entry in exportNames.entries) {
            final name = entry.key;
            final cleanName = name.endsWith('=')
                ? name.substring(0, name.length - 1)
                : name;
            final el = entry.value;
            if (barrelRel == 'lib/gpuarray.dart' &&
                forbiddenCoreBarrelSymbols.contains(cleanName)) {
              violations.add(
                '$barrelRel: must not re-export domain symbol "$cleanName"',
              );
            }
            if (removedLegacyAliases.contains(cleanName)) {
              violations.add(
                '$barrelRel: must not export removed legacy alias "$cleanName"',
              );
            }
            if (cleanName.contains('_')) {
              violations.add(
                '$barrelRel: must not export snake_case symbol "$cleanName"',
              );
            }
            if (el is ClassElement && cleanName.endsWith('Backward')) {
              violations.add(
                '$barrelRel: must not export internal autograd node "$cleanName"',
              );
            }
            if (el.metadata.annotations.any((a) => a.isInternal)) {
              violations.add(
                '$barrelRel: exported symbol "$name" (${el.kind.displayName}) is annotated @internal',
              );
            }
            if (el.library?.uri.scheme == 'package' &&
                el.library?.uri.pathSegments.firstOrNull == 'gpuarray') {
              final docComment =
                  el.documentationComment ??
                  (el is PropertyAccessorElement
                      ? el.variable.documentationComment
                      : null);
              if (docComment == null) {
                violations.add(
                  '$barrelRel: exported symbol "$name" (${el.kind.displayName}) is missing /// dartdoc',
                );
              }
              if (el is InstanceElement) {
                for (final method in el.methods) {
                  if (method.isPublic &&
                      !method.metadata.annotations.any((a) => a.isInternal) &&
                      (method.name?.contains('_') ?? false)) {
                    violations.add(
                      '$barrelRel: exported ${el.displayName}.${method.name} uses snake_case',
                    );
                  }
                }
                for (final getter in el.getters) {
                  if (getter.isPublic &&
                      !getter.metadata.annotations.any((a) => a.isInternal) &&
                      (getter.name?.contains('_') ?? false)) {
                    violations.add(
                      '$barrelRel: exported ${el.displayName}.${getter.name} uses snake_case',
                    );
                  }
                }
                for (final setter in el.setters) {
                  final setterName = setter.name?.replaceAll('=', '') ?? '';
                  if (setter.isPublic &&
                      !setter.metadata.annotations.any((a) => a.isInternal) &&
                      setterName.contains('_')) {
                    violations.add(
                      '$barrelRel: exported ${el.displayName}.$setterName= uses snake_case',
                    );
                  }
                }
              }
            }
          }
        }

        expect(violations, isEmpty, reason: violations.join('\n'));
        expect(
          verifiedProjectingFunctions.length,
          equals(34),
          reason:
              'Expected Rule 2 semantic check to verify all 34 single-slot projecting functions in gpuarray.',
        );
        expect(
          verifiedProjectingExtensions,
          containsAll(<String>{
            'GpuArrayDivide',
            'GpuArraySpecComponentExtension',
          }),
          reason:
              'Expected Rule 2 semantic check to verify GpuArrayDivide and GpuArraySpecComponentExtension in gpuarray.',
        );
      },
    );

    test(
      'Static dtype typing parity with package:ndarray: shape-derived SelfOf bounds, marker-bounded bitwise/shift operator extensions and escape hatches',
      () async {
        final libDir = Directory('${pkgRoot.path}/lib').absolute.path;
        final collection = AnalysisContextCollection(includedPaths: [libDir]);
        final session = collection.contextFor(libDir).currentSession;
        final violations = <String>[];

        // Every exported executable, de-duplicated across barrels.
        final exported = <String, ExecutableElement>{};
        for (final barrelRel in _barrelLibraries) {
          final barrelPath = File('${pkgRoot.path}/$barrelRel').absolute.path;
          final libRes = await session.getResolvedLibrary(barrelPath);
          if (libRes is! ResolvedLibraryResult) {
            violations.add('$barrelRel: failed to resolve library');
            continue;
          }
          for (final entry
              in libRes.element.exportNamespace.definedNames2.entries) {
            final el = entry.value;
            if (el is ExecutableElement) {
              exported.putIfAbsent('${el.library.uri}#${entry.key}', () => el);
            }
          }
        }
        expect(exported, isNotEmpty);

        bool isOutParameter(FormalParameterElement p) {
          final pName = p.name ?? '';
          return pName == 'out' ||
              (p.isNamed &&
                  pName.startsWith('out') &&
                  pName.length > 3 &&
                  pName[3].toUpperCase() == pName[3]);
        }

        // Shape-derived SelfOf rule (package:ndarray check 6b): a type
        // parameter that binds two or more non-`out` input positions
        // (`GpuArray<T>`, `DType<T>`, or `List<GpuArray<T>>` which counts as
        // two because a list literal LUBs its elements) can be inferred as the
        // least upper bound of distinct concrete tags, so it must be bounded
        // by `SelfOf<...>`. Conversely, `SelfOf` on a parameter that binds
        // fewer than two input positions is over-constraining.
        //
        // Functions that deliberately accept mixed dtypes and promote or cast
        // at run time keep `T extends DTypeTag` so `T` is the LUB of the
        // operands: the `dtype:` parameter of the accumulating reductions
        // requests a result dtype that may differ from the input's, so
        // `sum(i32, dtype: DType.float64)` infers `T = DTypeTag` and the
        // result dtype is only known at run time.
        const mixedDTypePromotingOps = <String>{'sum', 'nansum', 'prod'};
        int inputBindingCount(DartType type, TypeParameterElement tp) {
          if (type is InterfaceType) {
            final elName = type.element.name;
            if ((elName == 'GpuArray' || elName == 'DType') &&
                type.typeArguments.length == 1) {
              final arg = type.typeArguments.single;
              return (arg is TypeParameterType && arg.element == tp) ? 1 : 0;
            }
            if ((elName == 'List' || elName == 'Iterable') &&
                type.typeArguments.length == 1) {
              return 2 * inputBindingCount(type.typeArguments.single, tp);
            }
          }
          return 0;
        }

        final selfOfBoundedFunctions = <String>{};
        for (final entry in exported.entries) {
          final el = entry.value;
          final name = entry.key.split('#').last;
          if (el.typeParameters.isEmpty) continue;
          for (final tp in el.typeParameters) {
            var bindings = 0;
            for (final p in el.formalParameters) {
              if (isOutParameter(p)) continue;
              bindings += inputBindingCount(p.type, tp);
            }
            final bound = tp.bound;
            final isSelfOf =
                bound is InterfaceType && bound.element.name == 'SelfOf';
            if (isSelfOf) selfOfBoundedFunctions.add(name);
            if (bindings >= 2 &&
                !isSelfOf &&
                !mixedDTypePromotingOps.contains(name)) {
              violations.add(
                'Exported function `$name`: type parameter `${tp.name}` binds $bindings input positions but is bounded by `${bound?.getDisplayString()}`; bound it by `SelfOf<...>` so mixed-dtype calls are rejected at compile time (or add it to `mixedDTypePromotingOps` if it promotes dtypes at run time).',
              );
            }
            if (bindings < 2 && isSelfOf) {
              violations.add(
                'Exported function `$name`: type parameter `${tp.name}` is bounded by `${bound.getDisplayString()}` but binds only $bindings input position(s); `SelfOf` is only warranted for same-dtype multi-input parameters.',
              );
            }
            if (isSelfOf && mixedDTypePromotingOps.contains(name)) {
              violations.add(
                'Exported function `$name` is listed in `mixedDTypePromotingOps` but bounds `${tp.name}` by `SelfOf<...>`; remove it from the allow-list.',
              );
            }
          }
        }
        for (final name in mixedDTypePromotingOps) {
          if (!exported.keys.any((k) => k.endsWith('#$name'))) {
            violations.add(
              '`mixedDTypePromotingOps` lists `$name`, which is not an exported function.',
            );
          }
        }

        // Run-time-typed results default to `AnySpec` (package:ndarray's
        // choose / multi_dot / lstsq / average rule): a type parameter that
        // occurs in the result (return type or `out:`) but is determined
        // neither by a non-`out` input nor through the bound of a determined
        // type parameter (`T extends SelfOf<DivideOf<R>>` determines `R`) is
        // only known at run time, so it must be bounded by `AnySpec` for the
        // un-annotated result to compose with every other operation.
        void collectTypeParams(DartType? type, Set<TypeParameterElement> into) {
          if (type == null) return;
          if (type is TypeParameterType) {
            into.add(type.element);
          } else if (type is InterfaceType) {
            for (final arg in type.typeArguments) {
              collectTypeParams(arg, into);
            }
          } else if (type is RecordType) {
            for (final f in type.positionalFields) {
              collectTypeParams(f.type, into);
            }
            for (final f in type.namedFields) {
              collectTypeParams(f.type, into);
            }
          } else if (type is FunctionType) {
            collectTypeParams(type.returnType, into);
            for (final p in type.formalParameters) {
              collectTypeParams(p.type, into);
            }
          }
        }

        final runtimeDTypedResultFunctions = <String>{};
        for (final entry in exported.entries) {
          final el = entry.value;
          final name = entry.key.split('#').last;
          if (el.typeParameters.isEmpty) continue;
          final resultTypeParams = <TypeParameterElement>{};
          collectTypeParams(el.returnType, resultTypeParams);
          final determined = <TypeParameterElement>{};
          for (final p in el.formalParameters) {
            if (isOutParameter(p)) {
              collectTypeParams(p.type, resultTypeParams);
            } else {
              collectTypeParams(p.type, determined);
            }
          }
          var grew = true;
          while (grew) {
            grew = false;
            for (final tp in el.typeParameters) {
              if (!determined.contains(tp)) continue;
              final viaBound = <TypeParameterElement>{};
              collectTypeParams(tp.bound, viaBound);
              for (final dependent in viaBound) {
                if (determined.add(dependent)) grew = true;
              }
            }
          }
          for (final tp in el.typeParameters) {
            if (!resultTypeParams.contains(tp) || determined.contains(tp)) {
              continue;
            }
            // Dependent `Out extends T` parameters are validated by 6c.
            if (tp.bound is TypeParameterType) continue;
            if (_isAnySpecBound(tp)) {
              runtimeDTypedResultFunctions.add(name);
            } else {
              violations.add(
                'Exported function `$name`: result type parameter `${tp.name}` is determined only by `out:`; bound it by `AnySpec` instead of `${tp.bound?.getDisplayString()}` so the run-time-typed default result composes with every operation.',
              );
            }
          }
        }

        // Same-dtype binary functions must give both operands the same static
        // type (no `Object?` second operand); scalars go through the methods.
        const sameDTypeBinaryOps = <String>{
          'add',
          'subtract',
          'multiply',
          'divide',
          'floorDivide',
          'pow',
          'power',
          'remainder',
          'mod',
          'fmod',
          'maximum',
          'minimum',
          'atan2',
          'hypot',
          'copysign',
          'gcd',
          'lcm',
          'bitwiseAnd',
          'bitwiseOr',
          'bitwiseXor',
          'leftShift',
          'rightShift',
          'equal',
          'notEqual',
          'greater',
          'greaterEqual',
          'less',
          'lessEqual',
          'matmul',
          'dot',
          'vdot',
          'tensordot',
          'kron',
          'inner',
          'outer',
          'cross',
          'mseLoss',
          'l1Loss',
          'binaryCrossEntropy',
        };
        for (final fnName in sameDTypeBinaryOps) {
          final el = exported.entries
              .where((e) => e.key.endsWith('#$fnName'))
              .map((e) => e.value)
              .firstOrNull;
          if (el == null) {
            violations.add('Expected exported binary operation `$fnName`.');
            continue;
          }
          final p0Type = el.formalParameters[0].type.getDisplayString();
          final p1Type = el.formalParameters[1].type
              .getDisplayString()
              .replaceFirst(RegExp(r'\?$'), '');
          if (p0Type != p1Type) {
            violations.add(
              'Same-dtype binary operation `$fnName` must give its first two parameters identical static types (got `$p0Type` vs `$p1Type`).',
            );
          }
        }

        // Expected bound families (grouped as in the CHANGELOG).
        const expectedBounds = <String, Set<String>>{
          'SelfOf<DTypeTag>': {
            'add',
            'subtract',
            'multiply',
            'floorDivide',
            'pow',
            'power',
            'remainder',
            'mod',
            'fmod',
            'maximum',
            'minimum',
            'atan2',
            'hypot',
            'copysign',
            'equal',
            'notEqual',
            'greater',
            'greaterEqual',
            'less',
            'lessEqual',
            'matmul',
            'dot',
            'vdot',
            'multiDot',
            'einsum',
            'tensordot',
            'kron',
            'inner',
            'outer',
            'cross',
            'mseLoss',
            'l1Loss',
            'binaryCrossEntropy',
            'scaledDotProductAttention',
          },
          'SelfOf<DivideOf<R>>': {'divide'},
          'SelfOf<RealOf<BitwiseDType>>': {
            'bitwiseAnd',
            'bitwiseOr',
            'bitwiseXor',
          },
          'SelfOf<RealOf<IntegerDType>>': {
            'gcd',
            'lcm',
            'leftShift',
            'rightShift',
          },
        };
        for (final family in expectedBounds.entries) {
          for (final fnName in family.value) {
            final el = exported.entries
                .where((e) => e.key.endsWith('#$fnName'))
                .map((e) => e.value)
                .firstOrNull;
            final actual = el?.typeParameters.firstOrNull?.bound
                ?.getDisplayString();
            if (actual != family.key) {
              violations.add(
                'Exported function `$fnName` must bound its first type parameter by `${family.key}` (got `$actual`).',
              );
            }
          }
        }
        const unaryBitwiseOps = <String>{'bitwiseNot', 'invert'};
        for (final fnName in unaryBitwiseOps) {
          final el = exported.entries
              .where((e) => e.key.endsWith('#$fnName'))
              .map((e) => e.value)
              .firstOrNull;
          final actual = el?.typeParameters.firstOrNull?.bound
              ?.getDisplayString();
          if (actual != 'RealOf<BitwiseDType>') {
            violations.add(
              'Exported function `$fnName` must bound its type parameter by `RealOf<BitwiseDType>` (got `$actual`).',
            );
          }
        }
        final expectedSelfOfFunctions = <String>{
          for (final family in expectedBounds.values) ...family,
        };

        // Operator extensions: exact declarations, exact operator sets, and no
        // bitwise/shift operator left on the `GpuArray` class itself.
        final gpuArrayFile = File('${pkgRoot.path}/lib/src/gpu_array.dart');
        final gpuArraySource = gpuArrayFile.readAsStringSync();
        const requiredDeclarations = <String>[
          'extension GpuArrayBitwise<T extends BitwiseDType> on GpuArray<T> {',
          'extension GpuArrayBitwiseSpec on GpuArray<AnyBitwiseSpec> {',
          'extension GpuArrayShift<T extends IntegerDType> on GpuArray<T> {',
          'extension GpuArrayShiftSpec on GpuArray<AnyIntegerSpec> {',
        ];
        for (final declaration in requiredDeclarations) {
          if (!gpuArraySource.contains(declaration)) {
            violations.add(
              'lib/src/gpu_array.dart: missing declaration `$declaration`',
            );
          }
        }
        const bitwiseOperators = <String>{'&', '|', '^', '~'};
        const shiftOperators = <String>{'<<', '>>'};
        const operatorExtensions = <String, Set<String>>{
          'GpuArrayBitwise': bitwiseOperators,
          'GpuArrayBitwiseSpec': bitwiseOperators,
          'GpuArrayShift': shiftOperators,
          'GpuArrayShiftSpec': shiftOperators,
        };
        const escapeHatches = <String, String>{
          'asAnySpec': 'GpuArray<AnySpec>',
          'asBitwiseDType': 'GpuArray<AnyBitwiseSpec>',
          'asIntegerDType': 'GpuArray<AnyIntegerSpec>',
        };
        final parsed = parseString(
          content: gpuArraySource,
          throwIfDiagnostics: false,
        );
        final seenOperatorExtensions = <String>{};
        for (final declaration in parsed.unit.declarations) {
          if (declaration is ClassDeclaration &&
              declaration.namePart.typeName.lexeme == 'GpuArray') {
            final getterReturnTypes = <String, String?>{};
            final body = declaration.body;
            final classMembers = body is BlockClassBody
                ? body.members
                : const <ClassMember>[];
            for (final member in classMembers) {
              if (member is! MethodDeclaration) continue;
              final memberName = member.name.lexeme;
              if (member.isOperator &&
                  (bitwiseOperators.contains(memberName) ||
                      shiftOperators.contains(memberName))) {
                violations.add(
                  'lib/src/gpu_array.dart: `GpuArray` must not declare `operator $memberName` directly; it belongs on the marker-bounded extensions.',
                );
              }
              if (member.isGetter) {
                getterReturnTypes[memberName] = member.returnType?.toSource();
              }
            }
            for (final hatch in escapeHatches.entries) {
              final actual = getterReturnTypes[hatch.key];
              if (actual != hatch.value) {
                violations.add(
                  'lib/src/gpu_array.dart: `GpuArray` must declare `${hatch.value} get ${hatch.key}` (got `$actual`).',
                );
              }
            }
          } else if (declaration is ExtensionDeclaration) {
            final extName = declaration.name?.lexeme;
            final declaredOperators = <String>{
              for (final member in declaration.body.members)
                if (member is MethodDeclaration && member.isOperator)
                  member.name.lexeme,
            };
            final expectedOperators = operatorExtensions[extName];
            if (expectedOperators == null) {
              for (final op in declaredOperators) {
                if (bitwiseOperators.contains(op) ||
                    shiftOperators.contains(op)) {
                  violations.add(
                    'lib/src/gpu_array.dart: extension `$extName` must not declare `operator $op`; only the GpuArrayBitwise / GpuArrayShift families may.',
                  );
                }
              }
              continue;
            }
            seenOperatorExtensions.add(extName!);
            if (declaredOperators.length != expectedOperators.length ||
                !declaredOperators.containsAll(expectedOperators)) {
              violations.add(
                'lib/src/gpu_array.dart: extension `$extName` must declare exactly the operators $expectedOperators (got $declaredOperators).',
              );
            }
          }
        }
        expect(
          seenOperatorExtensions,
          equals(operatorExtensions.keys.toSet()),
          reason: 'Expected all four operator extensions in gpu_array.dart.',
        );

        expect(violations, isEmpty, reason: violations.join('\n'));
        expect(
          selfOfBoundedFunctions,
          equals(expectedSelfOfFunctions),
          reason:
              'Expected the shape-derived SelfOf rule to cover exactly the same-dtype binary and multi-array functions of package:gpuarray.',
        );
        expect(
          runtimeDTypedResultFunctions,
          equals({
            'where',
            'select',
            'concatenate',
            'stack',
            'vstack',
            'hstack',
            'dstack',
            'columnStack',
            'permutation',
          }),
          reason:
              'Exactly these exported functions have a run-time-determined result dtype typed only by `out:` (bounded by `AnySpec`); every other result dtype must be a static projection of an input.',
        );
      },
    );

    test(
      'Static dtype typing parity with package:ndarray (6c): `out:` buffers are typed by dependent `Out extends T` parameters or `AnySpec`-bounded result parameters',
      () async {
        final libDir = Directory('${pkgRoot.path}/lib').absolute.path;
        final collection = AnalysisContextCollection(includedPaths: [libDir]);
        final session = collection.contextFor(libDir).currentSession;
        final violations = <String>[];

        // Every exported element (functions, classes, extensions),
        // de-duplicated across barrels.
        final exported = <String, Element>{};
        for (final barrelRel in _barrelLibraries) {
          final barrelPath = File('${pkgRoot.path}/$barrelRel').absolute.path;
          final libRes = await session.getResolvedLibrary(barrelPath);
          if (libRes is! ResolvedLibraryResult) {
            violations.add('$barrelRel: failed to resolve library');
            continue;
          }
          for (final entry
              in libRes.element.exportNamespace.definedNames2.entries) {
            final el = entry.value;
            if (el is ExecutableElement ||
                el is InterfaceElement ||
                el is ExtensionElement) {
              exported.putIfAbsent('${el.library?.uri}#${entry.key}', () => el);
            }
          }
        }
        expect(exported, isNotEmpty);

        bool isOutParameter(FormalParameterElement p) {
          final pName = p.name ?? '';
          return p.isNamed &&
              (pName == 'out' ||
                  (pName.startsWith('out') &&
                      pName.length > 3 &&
                      pName[3].toUpperCase() == pName[3]));
        }

        int inputBindingCount(DartType type, TypeParameterElement tp) {
          if (type is InterfaceType) {
            final elName = type.element.name;
            if ((elName == 'GpuArray' || elName == 'DType') &&
                type.typeArguments.length == 1) {
              final arg = type.typeArguments.single;
              return (arg is TypeParameterType && arg.element == tp) ? 1 : 0;
            }
            if ((elName == 'List' || elName == 'Iterable') &&
                type.typeArguments.length == 1) {
              return 2 * inputBindingCount(type.typeArguments.single, tp);
            }
          } else if (type is RecordType) {
            var n = 0;
            for (final f in type.positionalFields) {
              n += inputBindingCount(f.type, tp);
            }
            for (final f in type.namedFields) {
              n += inputBindingCount(f.type, tp);
            }
            return n;
          }
          return 0;
        }

        int arrayBindingCount(DartType type, TypeParameterElement tp) {
          if (type is InterfaceType) {
            final elName = type.element.name;
            if (elName == 'GpuArray' && type.typeArguments.length == 1) {
              final arg = type.typeArguments.single;
              return (arg is TypeParameterType && arg.element == tp) ? 1 : 0;
            }
            if ((elName == 'List' || elName == 'Iterable') &&
                type.typeArguments.length == 1) {
              return 2 * arrayBindingCount(type.typeArguments.single, tp);
            }
          } else if (type is RecordType) {
            var n = 0;
            for (final f in type.positionalFields) {
              n += arrayBindingCount(f.type, tp);
            }
            for (final f in type.namedFields) {
              n += arrayBindingCount(f.type, tp);
            }
            return n;
          }
          return 0;
        }

        // 6c. `out:` buffers must not widen inference. If a type parameter `T`
        // binds exactly one input position (or is a projection result `R`)
        // and also types an `out` parameter, `f(f64, out: f32)` infers the LUB
        // `DTypeTag` and only fails at run time. Such an `out` must instead be
        // typed by a dependent parameter `Out extends T` (named `Out` for a
        // parameter called `out`, `OutXyz` for `outXyz`, `OutField` for a
        // field of a record-typed `out`), which binds nothing but that `out`
        // position. The acceptable alternatives are a `SelfOf`-bounded `T`
        // that binds two or more inputs (rule 6b already covers `out`) and an
        // `AnySpec`-bounded result parameter. The latter is *required* when `T`
        // is bound only by an optional `DType<T>?` (creation-style functions
        // such as `uniform`, `normal`): with `T extends AnySpec` and
        // `out: GpuArray<T>`, either `dtype:` or `out:` alone infers `T`, a
        // disagreement between them is rejected (their LUB is never a row),
        // and the default result composes; a dependent `Out` would instead
        // make `uniform([5], out: f32)` infer `T = DTypeTag`.
        //
        // Method-level type parameters (declared on a method of an exported
        // class or extension, not on the class) are subject to the same rule;
        // class-level parameters are fixed by the receiver and are safe.
        String dependentNameFor(FormalParameterElement p, String? field) {
          final pName = p.name ?? '';
          final base = pName == 'out' ? 'Out' : 'Out${pName.substring(3)}';
          if (field == null) return base;
          return '$base${field[0].toUpperCase()}${field.substring(1)}';
        }

        var dependentOutParameters = 0;
        var optionalDTypeAnySpecParameters = 0;
        void checkOutParameters(String owner, ExecutableElement el) {
          if (el.typeParameters.isEmpty) return;
          final typeParams = el.typeParameters.toSet();
          final nonOutParams = [
            for (final p in el.formalParameters)
              if (!isOutParameter(p)) p,
          ];
          for (final tp in el.typeParameters) {
            var inputs = 0;
            var arrayInputs = 0;
            var boundOnlyByOptionalDType = true;
            for (final p in nonOutParams) {
              final n = inputBindingCount(p.type, tp);
              inputs += n;
              arrayInputs += arrayBindingCount(p.type, tp);
              if (n > 0 &&
                  (p.isRequired || arrayBindingCount(p.type, tp) > 0)) {
                boundOnlyByOptionalDType = false;
              }
            }
            final outSites = <String>[];
            for (final p in el.formalParameters) {
              if (!isOutParameter(p)) continue;
              final t = p.type;
              if (t is RecordType) {
                for (final f in t.namedFields) {
                  if (inputBindingCount(f.type, tp) > 0) {
                    outSites.add(dependentNameFor(p, f.name));
                  }
                }
                for (final f in t.positionalFields) {
                  if (inputBindingCount(f.type, tp) > 0) {
                    outSites.add(dependentNameFor(p, null));
                  }
                }
              } else if (inputBindingCount(t, tp) > 0) {
                outSites.add(dependentNameFor(p, null));
              }
            }
            final bound = tp.bound;
            final dependsOn = bound is TypeParameterType ? bound.element : null;
            if (dependsOn != null) {
              if (!typeParams.contains(dependsOn)) {
                violations.add(
                  '$owner: `${tp.name} extends ${dependsOn.name}` must depend on a type parameter of the same declaration.',
                );
              } else if (inputs != 0 || outSites.length != 1) {
                violations.add(
                  '$owner: dependent type parameter `${tp.name}` must bind exactly one `out` position and no input position (binds $inputs input(s) and ${outSites.length} out site(s)).',
                );
              } else if (outSites.single != tp.name) {
                violations.add(
                  '$owner: dependent out type parameter must be named `${outSites.single}` (got `${tp.name}`).',
                );
              } else {
                final baseInputs = nonOutParams.fold<int>(
                  0,
                  (n, p) => n + inputBindingCount(p.type, dependsOn),
                );
                final baseArrayInputs = nonOutParams.fold<int>(
                  0,
                  (n, p) => n + arrayBindingCount(p.type, dependsOn),
                );
                final baseOptionalOnly =
                    baseArrayInputs == 0 &&
                    baseInputs > 0 &&
                    nonOutParams.every(
                      (p) =>
                          inputBindingCount(p.type, dependsOn) == 0 ||
                          !p.isRequired,
                    );
                if (baseOptionalOnly) {
                  violations.add(
                    '$owner: `${dependsOn.name}` is bound only by an optional `DType<${dependsOn.name}>?`; bound it by `AnySpec` and type `out` as `GpuArray<${dependsOn.name}>` instead of the dependent `${tp.name}` so `out:` alone still infers the dtype.',
                  );
                } else {
                  dependentOutParameters++;
                }
              }
              continue;
            }
            if (outSites.isEmpty) continue;
            final isSelfOf =
                bound is InterfaceType && bound.element.name == 'SelfOf';
            if (inputs >= 2 && isSelfOf) continue;
            // A row bound (`AnySpec`) rejects LUBs by itself, whether the
            // parameter is determined only by `out` or also by a `DType<T>`.
            if (_isAnySpecBound(tp)) {
              if (arrayInputs == 0 && inputs > 0 && boundOnlyByOptionalDType) {
                optionalDTypeAnySpecParameters++;
              }
              continue;
            }
            violations.add(
              '$owner: `out` parameter(s) ${outSites.map((s) => '`$s`').join(', ')} are typed by `${tp.name}` which binds $inputs input position(s); a mismatched `out:` dtype would widen inference to the LUB instead of failing. Type each such `out` by a dependent parameter (`${outSites.first} extends ${tp.name}`), or bound `${tp.name}` by `AnySpec` if it is only determined by an optional `dtype:`/`out:`.',
            );
          }
        }

        for (final entry in exported.entries) {
          final name = entry.key.split('#').last;
          final el = entry.value;
          if (el is ExecutableElement) {
            checkOutParameters('Exported function `$name`', el);
          } else if (el is InterfaceElement) {
            for (final m in el.methods) {
              if (m.isPrivate) continue;
              checkOutParameters('Exported method `$name.${m.name}`', m);
            }
          } else if (el is ExtensionElement) {
            for (final m in el.methods) {
              if (m.isPrivate) continue;
              checkOutParameters(
                'Exported extension method `$name.${m.name}`',
                m,
              );
            }
          }
        }

        expect(violations, isEmpty, reason: violations.join('\n'));
        expect(
          optionalDTypeAnySpecParameters,
          greaterThanOrEqualTo(1),
          reason:
              'Expected the creation-style functions (random family, ...) to use `T extends AnySpec` with `out: GpuArray<T>`.',
        );
        expect(
          dependentOutParameters,
          greaterThanOrEqualTo(1),
          reason:
              'Expected rule 6c to find a dependent `Out extends T` parameter on every `out:`-bearing function whose result dtype follows an input or a projection.',
        );
      },
    );

    test(
      'R4.1: wgpu_bindings.dart uses @ffi.DefaultAsset and @ffi.Native with zero DynamicLibrary.open in lib/',
      () {
        final bindingsFile = File(
          '${pkgRoot.path}/lib/src/backend/native/wgpu_bindings.dart',
        );
        expect(bindingsFile.existsSync(), isTrue);
        final bindingsSrc = bindingsFile.readAsStringSync();
        expect(
          bindingsSrc,
          contains("@ffi.DefaultAsset('package:gpuarray/wgpu_native')"),
        );
        expect(bindingsSrc, contains('@ffi.Native<'));

        final violations = <String>[];
        for (final file in libFiles) {
          final src = file.readAsStringSync();
          if (src.contains('DynamicLibrary.open')) {
            violations.add('${relPath(file)}: contains DynamicLibrary.open');
          }
        }
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test(
      'R4.1: zero mock/CPU-fallback tokens, zero forbidden package:ndarray imports, and zero NativeFinalizer externalSize',
      () {
        final violations = <String>[];
        const forbiddenTokens = [
          'CpuVectorBackend',
          'isMock',
          'isSimulated',
          'cpu_kernel',
          'withTemporaryNDArrayView',
          'GpuDevice.cpu',
        ];
        for (final file in [...libFiles, ...testFiles, ...benchmarkFiles]) {
          final path = relPath(file);
          if (path == 'test/meta/codebase_invariants_test.dart') continue;
          final src = file.readAsStringSync();
          for (final token in forbiddenTokens) {
            if (RegExp('\\b${RegExp.escape(token)}\\b').hasMatch(src)) {
              violations.add('$path: contains forbidden token "$token"');
            }
          }
        }

        const allowedNdarrayFiles = {
          'lib/gpuarray.dart',
          'lib/src/interop.dart',
          'lib/src/gpu_array.dart',
          'lib/src/dtype.dart',
        };
        for (final file in libFiles) {
          final path = relPath(file);
          final result = parseString(content: file.readAsStringSync());
          final finalizerVisitor = _FinalizerExternalSizeVisitor(
            path,
            result.lineInfo,
            violations,
          );
          result.unit.accept(finalizerVisitor);

          if (!allowedNdarrayFiles.contains(path)) {
            for (final directive in result.unit.directives) {
              if (directive is UriBasedDirective) {
                final uri = directive.uri.stringValue ?? '';
                if (uri.startsWith('package:ndarray/')) {
                  violations.add(
                    '$path: forbidden package:ndarray import/export "$uri"',
                  );
                }
              }
            }
          }
        }
        expect(violations, isEmpty, reason: violations.join('\n'));
      },
    );

    test('README.md Dart code blocks are syntactically valid Dart', () {
      final readme = File('${pkgRoot.path}/README.md');
      if (!readme.existsSync()) return;
      final content = readme.readAsStringSync();
      final blockRegex = RegExp(r'```dart\s*\n([\s\S]*?)```');
      final violations = <String>[];
      var blockIndex = 0;
      for (final match in blockRegex.allMatches(content)) {
        blockIndex++;
        final snippet = match.group(1)!;
        final direct = parseString(content: snippet, throwIfDiagnostics: false);
        if (direct.errors.isEmpty) continue;
        final wrapped = parseString(
          content: 'Future<void> _readmeSnippet() async {\n$snippet\n}',
          throwIfDiagnostics: false,
        );
        if (wrapped.errors.isNotEmpty) {
          violations.add(
            'README.md block #$blockIndex has syntax errors: '
            '${direct.errors.map((e) => e.message).join('; ')}',
          );
        }
      }
      expect(violations, isEmpty, reason: violations.join('\n'));
    });
  });
}

class _IfDTypeVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final RegExp dtypeRegex;
  final List<String> violations;

  _IfDTypeVisitor(
    this.filePath,
    this.lineInfo,
    this.dtypeRegex,
    this.violations,
  );

  @override
  void visitIfStatement(IfStatement node) {
    if (node.elseStatement is IfStatement) {
      final condSource = node.expression.toSource();
      if (dtypeRegex.hasMatch(condSource)) {
        final line = lineInfo.getLocation(node.offset).lineNumber;
        violations.add(
          '$filePath:$line: if-else chain on DType ($condSource); use switch',
        );
      }
    }
    super.visitIfStatement(node);
  }
}

class _ErrorContractVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations;

  _ErrorContractVisitor(this.filePath, this.lineInfo, this.violations);

  @override
  void visitInstanceCreationExpression(InstanceCreationExpression node) {
    final typeName = node.constructorName.type.name.lexeme;
    final ctorName = node.constructorName.name?.name;
    final line = lineInfo.getLocation(node.offset).lineNumber;

    if (typeName == 'ArgumentError' && ctorName == null) {
      violations.add(
        '$filePath:$line: bare ArgumentError(...) is prohibited; '
        'use ArgumentError.value(val, "name", "Must ...")',
      );
    } else if ((typeName == 'ArgumentError' || typeName == 'RangeError') &&
        ctorName == 'value') {
      final args = node.argumentList.arguments;
      if (args.length < 3) {
        violations.add(
          '$filePath:$line: $typeName.value must include value, name, and message',
        );
      } else {
        final msgExpr = args[2];
        if (msgExpr is SimpleStringLiteral) {
          final msg = msgExpr.value;
          if (!msg.startsWith('Must ')) {
            violations.add(
              '$filePath:$line: $typeName.value message "$msg" must start with "Must be ..." or "Must not ..."',
            );
          }
        } else if (msgExpr is AdjacentStrings) {
          final first = msgExpr.strings.first;
          if (first is SimpleStringLiteral &&
              !first.value.startsWith('Must ')) {
            violations.add(
              '$filePath:$line: $typeName.value message must start with "Must be ..." or "Must not ..."',
            );
          }
        } else if (msgExpr is StringInterpolation) {
          final first = msgExpr.elements.first;
          if (first is InterpolationString &&
              !first.value.startsWith('Must ')) {
            violations.add(
              '$filePath:$line: $typeName.value message must start with "Must be ..." or "Must not ..."',
            );
          }
        }
      }
    } else if ((typeName == 'Exception' || typeName == 'Error') &&
        ctorName == null) {
      violations.add(
        '$filePath:$line: do not instantiate raw $typeName(); use a specific subclass',
      );
    }
    super.visitInstanceCreationExpression(node);
  }
}

class _ClassModifierVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations;

  _ClassModifierVisitor(this.filePath, this.lineInfo, this.violations);

  @override
  void visitClassDeclaration(ClassDeclaration node) {
    final hasModifier =
        node.abstractKeyword != null ||
        node.baseKeyword != null ||
        node.interfaceKeyword != null ||
        node.finalKeyword != null ||
        node.sealedKeyword != null ||
        node.mixinKeyword != null;
    if (!hasModifier) {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      final className = node.namePart.typeName.lexeme;
      violations.add(
        '$filePath:$line: class $className must have an explicit class modifier (e.g. final, base, interface, sealed, abstract)',
      );
    }
    super.visitClassDeclaration(node);
  }
}

class _StaticNamespaceClassVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations;

  _StaticNamespaceClassVisitor(this.filePath, this.lineInfo, this.violations);

  @override
  void visitClassDeclaration(ClassDeclaration node) {
    if (node.extendsClause != null ||
        node.implementsClause != null ||
        node.withClause != null) {
      super.visitClassDeclaration(node);
      return;
    }
    final body = node.body;
    if (body is! BlockClassBody || body.members.isEmpty) {
      super.visitClassDeclaration(node);
      return;
    }
    var hasStaticMember = false;
    var hasInstanceMember = false;
    for (final member in body.members) {
      if (member is ConstructorDeclaration) {
        final ctorName = member.name?.lexeme;
        if ((ctorName == null || !Identifier.isPrivateName(ctorName)) &&
            member.factoryKeyword == null) {
          hasInstanceMember = true;
        }
      } else if (member is MethodDeclaration) {
        if (member.isStatic) {
          hasStaticMember = true;
        } else {
          hasInstanceMember = true;
        }
      } else if (member is FieldDeclaration) {
        if (member.isStatic) {
          hasStaticMember = true;
        } else {
          hasInstanceMember = true;
        }
      }
    }
    if (hasStaticMember && !hasInstanceMember) {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      final className = node.namePart.typeName.lexeme;
      violations.add(
        '$filePath:$line: class $className contains only static members; '
        'avoid Java-style static namespace classes (use top-level functions/constants or extension types)',
      );
    }
    super.visitClassDeclaration(node);
  }
}

class _GetterAndBoolParamVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations;

  _GetterAndBoolParamVisitor(this.filePath, this.lineInfo, this.violations);

  @override
  void visitMethodDeclaration(MethodDeclaration node) {
    if (node.externalKeyword != null) {
      super.visitMethodDeclaration(node);
      return;
    }
    final name = node.name.lexeme;
    final line = lineInfo.getLocation(node.offset).lineNumber;
    if (!Identifier.isPrivateName(name) &&
        !node.isGetter &&
        !node.isOperator &&
        (node.parameters?.parameters.isEmpty ?? false)) {
      if ((name.startsWith('get') &&
              name.length > 3 &&
              name[3].toUpperCase() == name[3]) ||
          (name.startsWith('find') &&
              name.length > 4 &&
              name[4].toUpperCase() == name[4])) {
        violations.add(
          '$filePath:$line: parameterless method $name() must be a getter',
        );
      }
    }
    if (!Identifier.isPrivateName(name) && !node.isOperator) {
      _checkParams(node.parameters, name, line);
    }
    super.visitMethodDeclaration(node);
  }

  @override
  void visitFunctionDeclaration(FunctionDeclaration node) {
    if (node.parent is! CompilationUnit) {
      super.visitFunctionDeclaration(node);
      return;
    }
    final name = node.name.lexeme;
    final line = lineInfo.getLocation(node.offset).lineNumber;
    if (!Identifier.isPrivateName(name)) {
      _checkParams(node.functionExpression.parameters, name, line);
    }
    super.visitFunctionDeclaration(node);
  }

  void _checkParams(
    FormalParameterList? params,
    String callableName,
    int line,
  ) {
    if (params == null) return;
    for (final param in params.parameters) {
      if (param.isNamed) continue;
      final src = param.toSource();
      if (RegExp(
        r'^(?:required\s+|final\s+|covariant\s+)*bool\s+\w+',
      ).hasMatch(src)) {
        violations.add(
          '$filePath:$line: $callableName has positional bool parameter "${param.name?.lexeme}"; prefer named bool parameter',
        );
      }
    }
  }
}

class _DartdocVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations;
  final bool requirePublicDocs;

  static final _errorMentionRegex = RegExp(
    r'\b(?:Throws|throws|Throw|throw)\s+(?:an?\s+)?\[?(?:ArgumentError|RangeError|StateError|UnsupportedError|IndexError|ConcurrentModificationError|TypeError)\]?',
  );
  static final _fillerStartRegex = RegExp(
    r'^\s*///\s*(?:Calculates and returns|Computes and returns|Creates and returns|Returns a\b|Returns the\b|Returns an\b|Gets the\b|Gets a\b)',
  );

  _DartdocVisitor(
    this.filePath,
    this.lineInfo,
    this.violations, {
    required this.requirePublicDocs,
  });

  void _inspectComment(Comment? comment, String symbolName, int line) {
    if (comment == null) {
      if (requirePublicDocs && !Identifier.isPrivateName(symbolName)) {
        violations.add(
          '$filePath:$line: public declaration "$symbolName" is missing /// dartdoc',
        );
      }
      return;
    }
    final lines = comment.tokens.map((t) => t.lexeme).toList();
    if (lines.isEmpty) return;
    final fullText = lines.join('\n');
    if (_errorMentionRegex.hasMatch(fullText)) {
      violations.add(
        '$filePath:$line: dartdoc on "$symbolName" mentions an Error class in Throws clause; state precondition directly instead',
      );
    }
    if (_fillerStartRegex.hasMatch(lines.first)) {
      violations.add(
        '$filePath:$line: dartdoc on "$symbolName" starts with redundant filler ("${lines.first.trim()}")',
      );
    }
  }

  @override
  void visitClassDeclaration(ClassDeclaration node) {
    final name = node.namePart.typeName.lexeme;
    if (!Identifier.isPrivateName(name)) {
      _inspectComment(
        node.documentationComment,
        name,
        lineInfo.getLocation(node.offset).lineNumber,
      );
      final body = node.body;
      if (body is BlockClassBody) {
        for (final member in body.members) {
          if (member is MethodDeclaration &&
              !Identifier.isPrivateName(member.name.lexeme)) {
            final hasOverride = member.metadata.any(
              (Annotation m) => m.name.name == 'override',
            );
            if (!hasOverride) {
              _inspectComment(
                member.documentationComment,
                '$name.${member.name.lexeme}',
                lineInfo.getLocation(member.offset).lineNumber,
              );
            }
          } else if (member is FieldDeclaration && !member.isStatic) {
            for (final variable in member.fields.variables) {
              if (!Identifier.isPrivateName(variable.name.lexeme)) {
                final hasOverride = member.metadata.any(
                  (Annotation m) => m.name.name == 'override',
                );
                if (!hasOverride) {
                  _inspectComment(
                    member.documentationComment,
                    '$name.${variable.name.lexeme}',
                    lineInfo.getLocation(variable.offset).lineNumber,
                  );
                }
              }
            }
          }
        }
      }
    }
    super.visitClassDeclaration(node);
  }

  @override
  void visitFunctionDeclaration(FunctionDeclaration node) {
    if (node.parent is! CompilationUnit) {
      super.visitFunctionDeclaration(node);
      return;
    }
    final name = node.name.lexeme;
    if (!Identifier.isPrivateName(name)) {
      _inspectComment(
        node.documentationComment,
        name,
        lineInfo.getLocation(node.offset).lineNumber,
      );
    }
    super.visitFunctionDeclaration(node);
  }

  @override
  void visitEnumDeclaration(EnumDeclaration node) {
    final name = node.namePart.typeName.lexeme;
    if (!Identifier.isPrivateName(name)) {
      _inspectComment(
        node.documentationComment,
        name,
        lineInfo.getLocation(node.offset).lineNumber,
      );
    }
    super.visitEnumDeclaration(node);
  }

  @override
  void visitExtensionDeclaration(ExtensionDeclaration node) {
    final extName = node.name?.lexeme;
    if (extName != null && !Identifier.isPrivateName(extName)) {
      _inspectComment(
        node.documentationComment,
        extName,
        lineInfo.getLocation(node.offset).lineNumber,
      );
      for (final member in node.body.members) {
        if (member is MethodDeclaration &&
            !Identifier.isPrivateName(member.name.lexeme)) {
          _inspectComment(
            member.documentationComment,
            '$extName.${member.name.lexeme}',
            lineInfo.getLocation(member.offset).lineNumber,
          );
        }
      }
    }
    super.visitExtensionDeclaration(node);
  }
}

class _IdentifierAbbreviationVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations;

  static const _prohibitedWords = <String>{
    'idx',
    'cnt',
    'buf',
    'chr',
    'bts',
    'msg',
    'cfg',
    'ctx',
    'cb',
    'len',
    'pos',
    'str',
  };

  _IdentifierAbbreviationVisitor(this.filePath, this.lineInfo, this.violations);

  List<String> _splitCamelCase(String identifier) {
    final stripped = identifier.replaceAll('_', '');
    if (stripped.isEmpty) return const [];
    final words = <String>[];
    final pattern = RegExp(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)');
    for (final match in pattern.allMatches(stripped)) {
      words.add(match.group(0)!.toLowerCase());
    }
    return words;
  }

  void _checkName(String name, int offset) {
    for (final word in _splitCamelCase(name)) {
      if (_prohibitedWords.contains(word)) {
        final line = lineInfo.getLocation(offset).lineNumber;
        violations.add(
          '$filePath:$line: identifier "$name" contains prohibited abbreviated word "$word"',
        );
      }
    }
  }

  @override
  void visitClassDeclaration(ClassDeclaration node) {
    final nameToken = node.namePart.typeName;
    _checkName(nameToken.lexeme, nameToken.offset);
    super.visitClassDeclaration(node);
  }

  @override
  void visitConstructorDeclaration(ConstructorDeclaration node) {
    if (node.name != null) {
      _checkName(node.name!.lexeme, node.name!.offset);
    }
    for (final param in node.parameters.parameters) {
      if (param.name != null) {
        _checkName(param.name!.lexeme, param.name!.offset);
      }
    }
    super.visitConstructorDeclaration(node);
  }

  @override
  void visitMethodDeclaration(MethodDeclaration node) {
    _checkName(node.name.lexeme, node.name.offset);
    if (node.parameters != null) {
      for (final param in node.parameters!.parameters) {
        if (param.name != null) {
          _checkName(param.name!.lexeme, param.name!.offset);
        }
      }
    }
    super.visitMethodDeclaration(node);
  }

  @override
  void visitFunctionDeclaration(FunctionDeclaration node) {
    _checkName(node.name.lexeme, node.name.offset);
    final params = node.functionExpression.parameters;
    if (params != null) {
      for (final param in params.parameters) {
        if (param.name != null) {
          _checkName(param.name!.lexeme, param.name!.offset);
        }
      }
    }
    super.visitFunctionDeclaration(node);
  }

  @override
  void visitVariableDeclaration(VariableDeclaration node) {
    _checkName(node.name.lexeme, node.name.offset);
    super.visitVariableDeclaration(node);
  }
}

class _FinalizerExternalSizeVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations;

  _FinalizerExternalSizeVisitor(this.filePath, this.lineInfo, this.violations);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name == 'attach') {
      for (final arg in node.argumentList.arguments) {
        if (arg.toSource().startsWith('externalSize:')) {
          final line = lineInfo.getLocation(arg.offset).lineNumber;
          violations.add(
            '$filePath:$line: externalSize passed to NativeFinalizer.attach',
          );
        }
      }
    }
    super.visitMethodInvocation(node);
  }
}

class _SemanticInvariantVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final List<String> violations;
  final Set<String> verifiedProjectingFunctions;
  final Set<String> verifiedProjectingExtensions;

  static const _allowedUntypedReturnMembers = <String>{
    'scalar',
    'toList',
    'toNestedList',
  };

  /// `DTypeSpec` row aliases whose element slot is `dynamic` by design; a
  /// signature spelled with one of them (`asAnySpec`, the `GpuArrayBitwiseSpec`
  /// operators) is deliberately run-time typed, not accidentally `dynamic`.
  static const _wildcardRowAliases = <String>{'AnySpec', 'AnyBitwiseSpec'};

  static const _kProjectionInterfaceNames = <String>{
    'RealOf',
    'ElementOf',
    'RealFloatOf',
    'ComplexOf',
    'InexactOf',
    'AccumulatorOf',
    'DoublePrecisionOf',
    'DivideOf',
  };

  static const _allowListedTwoProjectionFunctions = <String>{
    'svd',
    'eigh',
    'lstsq',
    'slogdet',
  };

  _SemanticInvariantVisitor(
    this.filePath,
    this.lineInfo,
    this.violations, {
    required this.verifiedProjectingFunctions,
    required this.verifiedProjectingExtensions,
  });

  bool _containsDynamic(DartType? type) {
    if (type == null) return false;
    if (type is DynamicType) return true;
    if (type is InterfaceType) {
      if (_wildcardRowAliases.contains(type.alias?.element.name)) return false;
      return type.typeArguments.any(_containsDynamic);
    }
    if (type is RecordType) {
      return type.positionalFields.any((f) => _containsDynamic(f.type)) ||
          type.namedFields.any((f) => _containsDynamic(f.type));
    }
    if (type is FunctionType) {
      return _containsDynamic(type.returnType) ||
          type.formalParameters.any((p) => _containsDynamic(p.type));
    }
    return false;
  }

  void _collectTypeParams(DartType? type, Set<TypeParameterElement> target) {
    if (type == null) return;
    if (type is TypeParameterType) {
      target.add(type.element);
    } else if (type is InterfaceType) {
      for (final arg in type.typeArguments) {
        _collectTypeParams(arg, target);
      }
    } else if (type is RecordType) {
      for (final f in type.positionalFields) {
        _collectTypeParams(f.type, target);
      }
      for (final f in type.namedFields) {
        _collectTypeParams(f.type, target);
      }
    } else if (type is FunctionType) {
      _collectTypeParams(type.returnType, target);
      for (final p in type.formalParameters) {
        _collectTypeParams(p.type, target);
      }
    }
  }

  bool _isOutParameter(FormalParameterElement p) {
    final pName = p.name ?? '';
    return pName == 'out' ||
        (p.isNamed &&
            pName.startsWith('out') &&
            pName.length > 3 &&
            pName[3].toUpperCase() == pName[3]);
  }

  bool _isDirectlyDeterminedByInput(DartType type, TypeParameterElement tp) {
    if (type is InterfaceType) {
      final elName = type.element.name;
      if ((elName == 'GpuArray' || elName == 'NDArray' || elName == 'DType') &&
          type.typeArguments.length == 1) {
        final arg = type.typeArguments.single;
        if (arg is TypeParameterType && arg.element == tp) {
          return true;
        }
      }
      if ((elName == 'List' || elName == 'Iterable') &&
          type.typeArguments.length == 1) {
        return _isDirectlyDeterminedByInput(type.typeArguments.single, tp);
      }
    } else if (type is RecordType) {
      return type.positionalFields.any(
            (f) => _isDirectlyDeterminedByInput(f.type, tp),
          ) ||
          type.namedFields.any((f) => _isDirectlyDeterminedByInput(f.type, tp));
    } else if (type is FunctionType) {
      final ret = type.returnType;
      if (ret is TypeParameterType && ret.element == tp) {
        return true;
      }
    }
    return false;
  }

  bool _isBoundThroughProjection(
    TypeParameterElement targetTp,
    List<TypeParameterElement> allTypeParams,
    List<DartType> nonOutInputTypes, {
    Set<TypeParameterElement>? visited,
  }) {
    final seen = visited ?? <TypeParameterElement>{};
    if (!seen.add(targetTp)) return false;

    // Form B: input parameter is GpuArray<XOf<R>>
    for (final inputType in nonOutInputTypes) {
      if (inputType is InterfaceType &&
          (inputType.element.name == 'GpuArray' ||
              inputType.element.name == 'NDArray') &&
          inputType.typeArguments.length == 1) {
        final inner = inputType.typeArguments.single;
        if (inner is InterfaceType &&
            _kProjectionInterfaceNames.contains(inner.element.name) &&
            inner.typeArguments.length == 1) {
          final projArg = inner.typeArguments.single;
          if (projArg is TypeParameterType && projArg.element == targetTp) {
            return true;
          }
        }
      }
    }

    // Form A: another type parameter T has bound XOf<R> or SelfOf<XOf<R>> (and
    // T is determined by input or chained projection)
    for (final tp in allTypeParams) {
      var bound = tp.bound;
      while (bound is InterfaceType &&
          bound.element.name == 'SelfOf' &&
          bound.typeArguments.length == 1) {
        bound = bound.typeArguments.single;
      }
      if (bound is InterfaceType &&
          _kProjectionInterfaceNames.contains(bound.element.name) &&
          bound.typeArguments.length == 1) {
        final projArg = bound.typeArguments.single;
        if (projArg is TypeParameterType && projArg.element == targetTp) {
          final tpFromInput = nonOutInputTypes.any(
            (t) => _isDirectlyDeterminedByInput(t, tp),
          );
          if (tpFromInput ||
              _isBoundThroughProjection(
                tp,
                allTypeParams,
                nonOutInputTypes,
                visited: seen,
              )) {
            return true;
          }
        }
      }
    }
    return false;
  }

  void _checkRawGpuArrayTypeAnnotation(TypeAnnotation? typeNode, String owner) {
    if (typeNode == null) return;
    typeNode.accept(
      _RawGpuArrayTypeVisitor(filePath, lineInfo, owner, violations),
    );
  }

  @override
  void visitConstructorDeclaration(ConstructorDeclaration node) {
    final element = node.declaredFragment?.element;
    if (element != null &&
        element.isPublic &&
        (element.enclosingElement is! ClassElement ||
            (element.enclosingElement as ClassElement).isPublic)) {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      for (final param in element.formalParameters) {
        if (_containsDynamic(param.type)) {
          violations.add(
            '$filePath:$line: public constructor "${element.displayName}" parameter "${param.name}" has dynamic in type (${param.type})',
          );
        }
      }
      node.parameters.accept(
        _RawGpuArrayTypeVisitor(
          filePath,
          lineInfo,
          'public constructor "${element.displayName}"',
          violations,
        ),
      );
    }
    super.visitConstructorDeclaration(node);
  }

  @override
  void visitFieldDeclaration(FieldDeclaration node) {
    for (final variable in node.fields.variables) {
      final element = variable.declaredFragment?.element;
      if (element is FieldElement &&
          element.isPublic &&
          (element.enclosingElement is! ClassElement ||
              (element.enclosingElement as ClassElement).isPublic)) {
        final line = lineInfo.getLocation(variable.offset).lineNumber;
        if (_containsDynamic(element.type)) {
          violations.add(
            '$filePath:$line: public field "${element.name}" has dynamic in type (${element.type})',
          );
        }
        _checkRawGpuArrayTypeAnnotation(
          node.fields.type,
          'public field "${element.name}"',
        );
      }
    }
    super.visitFieldDeclaration(node);
  }

  @override
  void visitMethodDeclaration(MethodDeclaration node) {
    final element = node.declaredFragment?.element;
    if (element != null &&
        element.isPublic &&
        (element.enclosingElement is! ClassElement ||
            (element.enclosingElement as ClassElement).isPublic)) {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      if (_containsDynamic(element.returnType) &&
          !_allowedUntypedReturnMembers.contains(element.name)) {
        violations.add(
          '$filePath:$line: public method/getter "${element.name}" has dynamic in return type (${element.returnType})',
        );
      }
      _checkRawGpuArrayTypeAnnotation(
        node.returnType,
        'public method "${element.name}"',
      );
      for (final param in element.formalParameters) {
        if (_containsDynamic(param.type)) {
          violations.add(
            '$filePath:$line: public method "${element.name}" parameter "${param.name}" has dynamic in type (${param.type})',
          );
        }
        final enclosing = element.enclosingElement;
        if (!element.isStatic &&
            enclosing is ClassElement &&
            enclosing.name == 'GpuArray' &&
            enclosing.typeParameters.isNotEmpty) {
          final classTypeParam = enclosing.typeParameters.first;
          final paramType = param.type;
          if (paramType is InterfaceType &&
              paramType.typeArguments.any(
                (t) => t is TypeParameterType && t.element == classTypeParam,
              )) {
            violations.add(
              '$filePath:$line: GpuArray<T> instance method "${element.name}" parameter "${param.name}" uses covariant class type parameter T (${param.type}); declare on GpuArrayTypedOperationsExtension instead so widened receivers throw GpuDTypeMismatchException instead of TypeError',
            );
          }
        }
      }
      node.parameters?.accept(
        _RawGpuArrayTypeVisitor(
          filePath,
          lineInfo,
          'public method "${element.name}"',
          violations,
        ),
      );
    }
    super.visitMethodDeclaration(node);
  }

  @override
  void visitExtensionDeclaration(ExtensionDeclaration node) {
    final element = node.declaredFragment?.element;
    if (element != null &&
        element.isPublic &&
        element.typeParameters.isNotEmpty) {
      final extName = element.name ?? '<unnamed>';
      final line = lineInfo.getLocation(node.offset).lineNumber;
      for (final m in [...element.methods, ...element.getters]) {
        if (!m.isPublic) continue;
        final allTypeParams = [...element.typeParameters, ...m.typeParameters];
        final resultTypeParams = <TypeParameterElement>{};
        _collectTypeParams(m.returnType, resultTypeParams);
        final nonOutInputTypes = <DartType>[element.extendedType];
        for (final p in m.formalParameters) {
          if (_isOutParameter(p)) {
            _collectTypeParams(p.type, resultTypeParams);
          } else {
            nonOutInputTypes.add(p.type);
          }
        }
        for (final tp in allTypeParams) {
          if (!resultTypeParams.contains(tp)) continue;
          // Dependent `Out extends T` parameters are validated by 6c.
          if (tp.bound is TypeParameterType) continue;
          final direct = nonOutInputTypes.any(
            (t) => _isDirectlyDeterminedByInput(t, tp),
          );
          if (direct) continue;
          if (_isBoundThroughProjection(tp, allTypeParams, nonOutInputTypes)) {
            verifiedProjectingExtensions.add(extName);
          } else {
            violations.add(
              '$filePath:$line: extension "$extName.${m.name}" result/out type parameter "${tp.name}" is not directly determined by receiver/input and is not bound through a `*Of<${tp.name}>` projection interface (Rule 2)',
            );
          }
        }
      }
    }
    super.visitExtensionDeclaration(node);
  }

  @override
  void visitFunctionDeclaration(FunctionDeclaration node) {
    if (node.parent is! CompilationUnit) {
      super.visitFunctionDeclaration(node);
      return;
    }
    final element = node.declaredFragment?.element;
    if (element != null && element.isPublic) {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      if (_containsDynamic(element.returnType)) {
        violations.add(
          '$filePath:$line: public function "${element.name}" has dynamic in return type (${element.returnType})',
        );
      }
      _checkRawGpuArrayTypeAnnotation(
        node.returnType,
        'public function "${element.name}"',
      );
      for (final param in element.formalParameters) {
        if (_containsDynamic(param.type)) {
          violations.add(
            '$filePath:$line: public function "${element.name}" parameter "${param.name}" has dynamic in type (${param.type})',
          );
        }
      }
      node.functionExpression.parameters?.accept(
        _RawGpuArrayTypeVisitor(
          filePath,
          lineInfo,
          'public function "${element.name}"',
          violations,
        ),
      );

      final fnName = element.name ?? '';
      if (element.typeParameters.isNotEmpty &&
          !_allowListedTwoProjectionFunctions.contains(fnName)) {
        final resultTypeParams = <TypeParameterElement>{};
        _collectTypeParams(element.returnType, resultTypeParams);
        final nonOutInputTypes = <DartType>[];
        var hasOutParam = false;
        for (final p in element.formalParameters) {
          if (_isOutParameter(p)) {
            hasOutParam = true;
            _collectTypeParams(p.type, resultTypeParams);
          } else {
            nonOutInputTypes.add(p.type);
          }
        }
        final nonOutTypeParams = <TypeParameterElement>{};
        for (final t in nonOutInputTypes) {
          _collectTypeParams(t, nonOutTypeParams);
        }
        // Output-only single-type-param functions (e.g., where, select,
        // concatenate, stack, vstack, hstack, dstack, columnStack,
        // permutation) determine T from `{out}`; the exported ones must bound
        // it by `AnySpec`, which the typing-parity test checks over the barrel
        // export namespaces.
        if (!(nonOutTypeParams.isEmpty &&
            element.typeParameters.length == 1 &&
            hasOutParam)) {
          for (final tp in element.typeParameters) {
            if (!resultTypeParams.contains(tp)) continue;
            // Dependent `Out extends T` parameters are validated by 6c.
            if (tp.bound is TypeParameterType) continue;
            final direct = nonOutInputTypes.any(
              (t) => _isDirectlyDeterminedByInput(t, tp),
            );
            if (direct) continue;
            if (_isBoundThroughProjection(
              tp,
              element.typeParameters,
              nonOutInputTypes,
            )) {
              verifiedProjectingFunctions.add('$filePath:$fnName');
            } else {
              violations.add(
                '$filePath:$line: public function "$fnName" result/out type parameter "${tp.name}" is not directly determined by an input parameter and is not bound through a `*Of<${tp.name}>` projection interface (Rule 2)',
              );
            }
          }
        }
      }
    }
    super.visitFunctionDeclaration(node);
  }
}

class _RawGpuArrayTypeVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final String owner;
  final List<String> violations;

  _RawGpuArrayTypeVisitor(
    this.filePath,
    this.lineInfo,
    this.owner,
    this.violations,
  );

  @override
  void visitNamedType(NamedType node) {
    if (node.name.lexeme == 'GpuArray' && node.typeArguments == null) {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      violations.add(
        '$filePath:$line: $owner uses untyped GpuArray without explicit <DTypeTag> type argument',
      );
    }
    super.visitNamedType(node);
  }
}

class _GpuDTypeSpecUsageVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final LineInfo lineInfo;
  final Map<String, Set<String>> allowedTwoProjection;
  final Set<String> seenTwoProjectionSites;
  final List<String> violations;

  _GpuDTypeSpecUsageVisitor({
    required this.filePath,
    required this.lineInfo,
    required this.allowedTwoProjection,
    required this.seenTwoProjectionSites,
    required this.violations,
  });

  @override
  void visitNamedType(NamedType node) {
    if (node.name.lexeme == 'DTypeSpec') {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      final argCount = node.typeArguments?.arguments.length ?? 0;
      if (argCount != 8) {
        violations.add(
          '$filePath:$line — `DTypeSpec` referenced with $argCount type arguments (expected 8).',
        );
      } else {
        AstNode? current = node.parent;
        String? ownerFn;
        while (current != null) {
          if (current is FunctionDeclaration) {
            ownerFn = current.name.lexeme;
            break;
          }
          current = current.parent;
        }
        final allowedFns = allowedTwoProjection[filePath];
        if (ownerFn != null &&
            allowedFns != null &&
            allowedFns.contains(ownerFn)) {
          final site = '$filePath:$ownerFn';
          if (!seenTwoProjectionSites.add(site)) {
            violations.add(
              '$filePath:$line — duplicate 8-arg `DTypeSpec` bound in `$ownerFn`.',
            );
          }
        } else {
          violations.add(
            '$filePath:$line — forbidden 8-arg `DTypeSpec<...>` bound in `${ownerFn ?? '<unknown>'}`; use a single-slot `*Of<R>` projection interface instead.',
          );
        }
      }
    }
    super.visitNamedType(node);
  }
}
