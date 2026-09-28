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

/// Base exception class for all ndarray-related errors.
final class NdArrayException implements Exception {
  /// The error message.
  final String message;

  /// Creates a new [NdArrayException] with the given [message].
  const NdArrayException(this.message);

  @override
  String toString() => 'NdArrayException: $message';
}

/// Exception thrown when a linear algebra operation fails.
final class LinAlgException extends NdArrayException {
  /// Creates a new [LinAlgException] with the given [message].
  const LinAlgException(super.message);

  @override
  String toString() => 'LinAlgException: $message';
}

/// Exception thrown when a matrix is singular and cannot be inverted or solved.
final class SingularMatrixException extends LinAlgException {
  /// Creates a new [SingularMatrixException] with the given [message].
  const SingularMatrixException(super.message);

  @override
  String toString() => 'SingularMatrixException: $message';
}

/// Exception thrown when a numerical method cannot find a real solution.
final class NoRealSolutionException extends NdArrayException {
  /// Creates a new [NoRealSolutionException] with the given [message].
  const NoRealSolutionException(super.message);

  @override
  String toString() => 'NoRealSolutionException: $message';
}

/// Exception thrown when a numerical solver exceeds its maximum iteration count or fails to converge.
final class IterationsExceededException extends LinAlgException {
  /// Creates a new [IterationsExceededException] with the given [message].
  const IterationsExceededException(super.message);

  @override
  String toString() => 'IterationsExceededException: $message';
}

/// Exception thrown when a matrix is expected to be positive-definite but is not.
final class NonPositiveDefiniteException extends LinAlgException {
  /// Creates a new [NonPositiveDefiniteException] with the given [message].
  const NonPositiveDefiniteException(super.message);

  @override
  String toString() => 'NonPositiveDefiniteException: $message';
}
