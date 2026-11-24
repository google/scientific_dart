## 0.1.1-wip

- Add Android (`arm64`, `x64`, NDK API >= 24, 16 KB page alignment) and iOS (`iphoneos-arm64`, `iphonesimulator-arm64`, `iphonesimulator-x64`, iOS >= 13.0) native build hook support.
- Replace `-Wl,-reexport_framework,Accelerate` on Apple platforms with 18 thin C `cblas_*` forwarders linking `-framework Accelerate` directly.

## 0.1.0

- Initial version.
