# Prebuilt Binary Artifacts & SLSA Provenance (`intl4x` style)

The native packages in this workspace (`pocketfft`, `openblas`, `ndarray`, `symbolic_dart`, and `gpuarray`) use Dart Native Assets (`hook/build.dart`) modeled after `package:intl4x` / `package:icu4x`.

## Build Modes (`fetch`, `local`, `source`)

Each package supports three build modes configured via `hooks.user_defines` in the workspace root `pubspec.yaml` (or overridden via environment variables):

1. **`fetch` (default for package consumers)**:
   Downloads prebuilt dynamic libraries for `(targetOS, targetArchitecture)` from GitHub Releases (`https://github.com/google/scientific_dart/releases/download/<version>/<artifact>`), caches them in `input.outputDirectoryShared`, and verifies their SHA-256 digest against the pinned `fileHashes` table in `lib/src/hook_helpers/hashes.dart`.
   ```yaml
   hooks:
     user_defines:
       ndarray:
         buildMode: fetch
       openblas:
         buildMode: fetch
       pocketfft:
         buildMode: fetch
   ```

2. **`local` (offline / custom prebuilt binaries)**:
   Uses locally existing dynamic library files from `localPath` (or environment variables `LOCAL_NDARRAY_BINARY`, `LOCAL_OPENBLAS_BINARY`, `LOCAL_POCKETFFT_BINARY`).
   ```yaml
   hooks:
     user_defines:
       ndarray:
         buildMode: local
         localPath: path/to/libndarray.so
       openblas:
         buildMode: local
         localPath: path/to/directory_containing_openblas_libs/
       pocketfft:
         buildMode: local
         localPath: path/to/libpocketfft.so
   ```

3. **`source` (default when developing inside this workspace)**:
   Compiles the native C/C++/Fortran libraries from source on the host machine. The root `pubspec.yaml` of this repository configures `buildMode: source` so local changes to C/C++ files in `hook/` are compiled automatically during development.
   ```yaml
   hooks:
     user_defines:
       ndarray:
         buildMode: source
       openblas:
         buildMode: source
       pocketfft:
         buildMode: source
   ```

### Environment Variable Overrides

You can override the build mode without editing `pubspec.yaml`:
- `SCIENTIFIC_DART_BUILD_MODE=fetch|local|source` (applies to all packages)
- `NDARRAY_BUILD_MODE=fetch|local|source` + `LOCAL_NDARRAY_BINARY=/path/to/lib`
- `OPENBLAS_BUILD_MODE=fetch|local|source` + `LOCAL_OPENBLAS_BINARY=/path/to/dir`
- `POCKETFFT_BUILD_MODE=fetch|local|source` + `LOCAL_POCKETFFT_BINARY=/path/to/lib`

---

## Publication Workflow (`pub.dev` & Native Artifacts)

Because `pub.dev` package tarballs are immutable and must contain the pinned SHA-256 digests (`lib/src/hook_helpers/hashes.dart`) *before* `dart pub publish` is run, the release process depends on whether native code (`hook/` C/C++ sources or OpenBLAS/PocketFFT versions) changed since the last artifact release:

### Case A: Dart-Only Changes (No `hook/` or C/C++ changes)
**Everything works out of the box—no artifact rebuild needed.**
- Keep the existing `releaseTag` (e.g., `artifacts-v0.0.2`) and `fileHashes` in `lib/src/hook_helpers/hashes.dart`.
- Bump the package version in `pubspec.yaml` / `CHANGELOG.md` and run `dart pub publish`. Consumers of the new Dart version will continue fetching and verifying the existing `artifacts-v0.0.2` binaries.

### Case B: Native C/C++ or Build Hook Changes
When `hook/*.cpp`, `hook/*.c`, `hook/*.h`, or native library versions change, you must build and attest the binaries **before** publishing to `pub.dev`:

1. **Trigger the GitHub Actions Artifact Build**:
   Either push an `artifacts-v<version>` tag or trigger `.github/workflows/artifacts.yml` via `gh workflow run`:
   ```bash
   git tag artifacts-v0.0.3
   git push origin artifacts-v0.0.3
   # Or without creating a git tag first:
   gh workflow run artifacts.yml -f tag=artifacts-v0.0.3
   ```
2. **Wait for `.github/workflows/artifacts.yml` to finish (~5 mins)**:
   - Compiles the 20 native binaries across `linux-x64`, `linux-arm64`, `macos-arm64`, `macos-x64`, and `windows-x64` via `dart tool/build_artifacts.dart`.
   - Signs SLSA v1.0 build provenance attestations (`actions/attest-build-provenance@v2`) and publishes the GitHub Release `artifacts-v0.0.3` with `SHA256SUMS.txt` and `provenance.intoto.jsonl`.
3. **Verify SLSA Provenance & Pin SHA-256 Hashes Locally**:
   ```bash
   dart tool/regenerate_hashes.dart artifacts-v0.0.3
   ```
   This downloads all 20 release binaries, runs `gh attestation verify <binary> --repo google/scientific_dart` on each one, and writes the new `releaseTag`, `nativeSourceHash` (combined SHA-256 of `hook/`), and `fileHashes` into `pkgs/{pocketfft,openblas,ndarray}/lib/src/hook_helpers/hashes.dart`.
4. **Commit & Publish to `pub.dev`**:
   Commit the updated `hashes.dart` files, push to `main`, and publish the packages (`dart pub publish`).

---

## Automatic Staleness Detection (`nativeSourceHash`)

Each `hashes.dart` file records `nativeSourceHash` — the combined SHA-256 digest of the package's `hook/` native source files (`*.dart`, `*.c`, `*.cpp`, `*.h`, `*.def`) at the time `artifacts-v<version>` was built.

If native sources in `hook/` are modified without rebuilding prebuilt artifacts:
1. **`FetchMode` guard (`hook/build.dart`)**: Refuses to link stale prebuilt binaries that do not match `hook/` and throws an error instructing the author to rebuild release artifacts or switch to `buildMode: source`.
2. **`SourceMode` warning (`hook/build.dart`)**: Prints a warning during local builds when `hook/` differs from `nativeSourceHash`.
3. **CI & Pre-publish check (`tool/check_artifact_hashes.dart`)**:
   ```bash
   dart tool/check_artifact_hashes.dart [--strict]
   ```
   Emits GitHub Actions `::warning::` annotations on PRs/pushes when `hook/` has diverged from `nativeSourceHash` (and exits with code `1` when `--strict` is passed).
