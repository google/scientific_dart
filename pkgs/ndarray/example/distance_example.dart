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

import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    print('=== Example 1: Atom Contact Map (Euclidean Distance) ===');
    // Simulate 3D coordinates for 5 atoms in a protein.
    // Shape: [5, 3] (5 atoms, 3 coordinates x, y, z each)
    final atoms = NDArray.fromList(
      [
        0.0, 0.0, 0.0, // Atom 0
        1.0, 1.0, 1.0, // Atom 1
        2.0, 0.0, 0.0, // Atom 2
        0.0, 2.0, 0.0, // Atom 3
        0.0, 0.0, 2.0, // Atom 4
      ],
      [5, 3],
      DType.float64,
    );

    print('Atom coordinates:');
    for (var i = 0; i < atoms.shape[0]; i++) {
      print(
        'Atom $i: [${atoms.getCell([i, 0])}, ${atoms.getCell([i, 1])}, ${atoms.getCell([i, 2])}]',
      );
    }

    // Compute pairwise distances (condensed representation)
    final condensedDist = pdist(atoms, metric: DistanceMetric.euclidean);
    print('\nCondensed pairwise distances (size ${condensedDist.shape[0]}):');
    final distList = List.generate(
      condensedDist.shape[0],
      (i) => condensedDist.getCell([i]),
    );
    print(distList.map((d) => d.toStringAsFixed(4)).toList());

    // Interpret condensed distances:
    // Pairs are in order: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    final pairs = [
      '0-1',
      '0-2',
      '0-3',
      '0-4',
      '1-2',
      '1-3',
      '1-4',
      '2-3',
      '2-4',
      '3-4',
    ];
    print('\nPairwise distances:');
    for (var i = 0; i < condensedDist.shape[0]; i++) {
      print(
        'Atom pair ${pairs[i]}: ${condensedDist.getCell([i]).toStringAsFixed(4)}',
      );
    }

    // Identify contacts (distance < 1.8)
    final contactThreshold = 1.8;
    print('\nContacts (distance < $contactThreshold):');
    for (var i = 0; i < condensedDist.shape[0]; i++) {
      final dist = condensedDist.getCell([i]);
      if (dist < contactThreshold) {
        print(
          'Atom pair ${pairs[i]} is in contact (dist: ${dist.toStringAsFixed(4)})',
        );
      }
    }

    // Also demonstrate cdist to get a full 5x5 distance matrix
    final fullDistMatrix = cdist(
      atoms,
      atoms,
      metric: DistanceMetric.euclidean,
    );
    print('\nFull Distance Matrix (5x5):');
    for (var i = 0; i < fullDistMatrix.shape[0]; i++) {
      final row = List.generate(
        fullDistMatrix.shape[1],
        (j) => fullDistMatrix.getCell([i, j]).toStringAsFixed(4),
      );
      print('Atom $i: $row');
    }

    print('\n=== Example 2: Sequence Divergence (Hamming Distance) ===');
    // Represent DNA sequences as integer arrays.
    // A=0, C=1, G=2, T=3
    // Sequence length = 10
    // We have 3 sequences in set A, and 2 in set B.
    final setA = NDArray.fromList(
      [
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, // Seq A0: ACGTACGTAC
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
        2,
        2, // Seq A1: ACGTACGTAG (differs at last 2 positions)
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // Seq A2: TTTTTTTTTT (very different)
      ],
      [3, 10],
      DType.int32,
    );

    final setB = NDArray.fromList(
      [
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, // Seq B0: ACGTACGTAC (identical to A0)
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
        0,
        2, // Seq B1: ACGTACGTAT (differs from A0 at last position)
      ],
      [2, 10],
      DType.int32,
    );

    print('Set A sequences:');
    for (var i = 0; i < setA.shape[0]; i++) {
      final seq = List.generate(setA.shape[1], (j) => setA.getCell([i, j]));
      print('Seq A$i: $seq');
    }
    print('Set B sequences:');
    for (var i = 0; i < setB.shape[0]; i++) {
      final seq = List.generate(setB.shape[1], (j) => setB.getCell([i, j]));
      print('Seq B$i: $seq');
    }

    // Compute distance matrix between Set A and Set B
    final divergence = cdist(setA, setB, metric: DistanceMetric.hamming);

    print('\nDivergence matrix (Set A x Set B) (Hamming Distance):');
    for (var i = 0; i < divergence.shape[0]; i++) {
      final row = List.generate(
        divergence.shape[1],
        (j) => divergence.getCell([i, j]).toStringAsFixed(2),
      );
      print('Seq A$i vs Set B: $row');
    }
  });
}
