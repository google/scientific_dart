# ndarray_ma (Experimental)

<img src="doc/images/mascot.png" align="right" width="220" alt="ndarray_ma mascot">

> [!WARNING]
> **Experimental**: This package is experimental and under active development. Its API may change significantly before reaching a stable release.

Masked arrays for the `ndarray` package.

A `MaskedArray<T>` wraps a standard `NDArray<T>` and associates it with a boolean `NDArray<Boolean>` mask of the same shape. True values in the mask indicate invalid or missing data.

All arithmetic operations and statistical reductions automatically bypass masked elements.

## Features

- **Runtime Type Preservation**: Operations preserve the correct runtime type (e.g., adding two `MaskedArray<Float64>` results in a `MaskedArray<Float64>`).
- **Memory Management**: Built with integration for `NDArray.scope` to prevent memory leaks from intermediate allocations.
- **Custom Fill Values**: Supports custom fill values that are preserved and coerced during arithmetic operations.
- **Safe Division**: Division automatically masks division-by-zero elements.

## Usage

### Creation

```dart
import 'package:ndarray/ndarray.dart';
import 'package:ndarray_ma/ndarray_ma.dart';

void main() {
  final data = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  final mask = NDArray.fromList([false, true, false, false], [2, 2], DType.boolean);

  // Create from data and mask
  final marr = MaskedArray(data, mask);

  // Factories
  final marrZeros = MaskedArray.zeros([2, 2], DType.float64);
  final marrInvalid = MaskedArray.maskedInvalid(data); // Masks NaNs/Infs
  final marrEqual = MaskedArray.maskedEqual(data, 2.0); // Masks elements equal to 2.0
}
```

### Arithmetic

Arithmetic operations bypass masked elements and propagate masks.

```dart
  final a = MaskedArray(
    NDArray.fromList([1.0, 2.0], [2], DType.float64),
    NDArray.fromList([false, true], [2], DType.boolean),
  );
  final b = MaskedArray(
    NDArray.fromList([10.0, 20.0], [2], DType.float64),
    NDArray.fromList([false, false], [2], DType.boolean),
  );

  final res = a.add(b);
  print(res.filled()); // [11.0, 1e20] (masked element filled with default)
```

### Reductions

Reductions ignore masked elements.

```dart
  final data = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
  final mask = NDArray.fromList([false, true, false, false], [4], DType.boolean);
  final marr = MaskedArray(data, mask);

  print(marr.sum().scalar); // 8 (1 + 3 + 4, 2 is ignored)
  print(marr.mean().scalar); // 2.666... (8 / 3)
}
```

## License

This package is licensed under the **[Apache License, Version 2.0](LICENSE)**.

## Disclaimer

This is not an official Google product.
