## 0.1.1-wip

- Added `ResourceScope.using((scope) { ... })` with `scope.defer(callback)` (LIFO cleanup callbacks executed before resource disposal) and `scope.escape(resource)` (promoting resources to the parent scope).

## 0.1.0

- Initial release of `resource_scope` for automatic zone-based scoped resource management.
