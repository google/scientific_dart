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

/**
 * Web Worker bootstrap for the in-browser self-hosted dart2wasm compiler.
 *
 * Loads `compiler_worker.wasm` + `compiler_worker.mjs`, populates the in-memory
 * file system with `dart2wasm_platform.dill` and `sources_bundle.json`, and
 * services `compile` and `bundle_for_analysis` requests from the notebook UI.
 */

globalThis.window ??= globalThis;
globalThis.dartUseDateNowForTicks = true;
globalThis.dartPrint = (line) => {
  // Forward compiler diagnostic logs to worker console if needed.
  console.debug('[compiler_worker]', String(line));
};

let initPromise = null;
let compileQueue = Promise.resolve();

async function ensureInitialized() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const baseUrl = new URL('./', import.meta.url).href;
    self.postMessage({
      type: 'status',
      state: 'busy',
      text: 'Loading Wasm Compiler & SDK...',
    });

    const [compilerMjs, wasmResponse, dillResponse, bundleResponse] =
      await Promise.all([
        import(new URL('./compiler_worker.mjs', baseUrl).href),
        fetch(new URL('./compiler_worker.wasm', baseUrl).href),
        fetch(new URL('./dart2wasm_platform.dill', baseUrl).href),
        fetch(new URL('./sources_bundle.json', baseUrl).href),
      ]);

    if (!wasmResponse.ok) {
      throw new Error(`Failed to fetch compiler_worker.wasm: ${wasmResponse.status}`);
    }
    if (!dillResponse.ok) {
      throw new Error(`Failed to fetch dart2wasm_platform.dill: ${dillResponse.status}`);
    }
    if (!bundleResponse.ok) {
      throw new Error(`Failed to fetch sources_bundle.json: ${bundleResponse.status}`);
    }

    const [wasmBuffer, dillBuffer, sourcesBundleText] = await Promise.all([
      wasmResponse.arrayBuffer(),
      dillResponse.arrayBuffer(),
      bundleResponse.text(),
    ]);

    const compiledApp = await compilerMjs.compile(wasmBuffer);
    const appInstance = await compiledApp.instantiate({});
    appInstance.invokeMain();

    if (typeof globalThis.dartCompilerInit !== 'function') {
      throw new Error('compiler_worker.wasm did not register dartCompilerInit');
    }

    const initJson = await globalThis.dartCompilerInit(
      new Uint8Array(dillBuffer),
      sourcesBundleText,
    );
    const initInfo = JSON.parse(String(initJson));
    self.postMessage({
      type: 'ready',
      ...initInfo,
    });
    return initInfo;
  })();
  return initPromise;
}

self.onmessage = (event) => {
  const msg = event.data;
  if (!msg || typeof msg.type !== 'string') return;

  if (msg.type === 'init') {
    ensureInitialized().catch((err) => {
      self.postMessage({
        type: 'init_error',
        error: String(err && err.stack ? err.stack : err),
      });
    });
    return;
  }

  if (msg.type === 'compile') {
    compileQueue = compileQueue.then(async () => {
      try {
        await ensureInitialized();
        const resultJson = await globalThis.dartCompilerCompile(
          JSON.stringify({
            cells: msg.cells,
            targetCellId: msg.targetCellId,
            sourceCode: msg.sourceCode,
          }),
        );
        const parsed = JSON.parse(String(resultJson));
        if (parsed.ok) {
          const rawWasm = globalThis.dartCompilerLastWasmBytes;
          const wasmCopy = new Uint8Array(rawWasm);
          const mjsText = String(globalThis.dartCompilerLastMjsText || '');
          self.postMessage(
            {
              type: 'compile_result',
              id: msg.id,
              ...parsed,
              wasmBytes: wasmCopy,
              mjsText,
            },
            [wasmCopy.buffer],
          );
        } else {
          self.postMessage({
            type: 'compile_result',
            id: msg.id,
            ...parsed,
          });
        }
      } catch (err) {
        self.postMessage({
          type: 'compile_result',
          id: msg.id,
          ok: false,
          error: String(err && err.stack ? err.stack : err),
        });
      }
    });
    return;
  }

  if (msg.type === 'bundle_for_analysis') {
    ensureInitialized()
      .then(() => {
        const resJson = globalThis.dartCompilerBundleForAnalysis(
          JSON.stringify({
            cells: msg.cells,
            activeCellId: msg.activeCellId,
            cursorOffset: msg.cursorOffset,
          }),
        );
        self.postMessage({
          type: 'analysis_bundle_result',
          id: msg.id,
          ...JSON.parse(String(resJson)),
        });
      })
      .catch((err) => {
        self.postMessage({
          type: 'analysis_bundle_result',
          id: msg.id,
          error: String(err),
        });
      });
  }
};

// Automatically begin initialization as soon as the worker is spawned.
ensureInitialized().catch((err) => {
  self.postMessage({
    type: 'init_error',
    error: String(err && err.stack ? err.stack : err),
  });
});
