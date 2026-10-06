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
 * Serverless Browser Runtime for `pkgs/notebook`.
 *
 * Coordinates:
 * 1. The in-browser `dart2wasm` Compiler Web Worker (`compiler_worker_bootstrap.js`).
 * 2. The Wasm32-WASI `native_math.wasm` + `cell.wasm` sandbox executor.
 * 3. Optional DartPad Worker (`dartpad/worker.wasm` + `sdk.tar`) LSP bridge with
 *    instant client-side completion/hover fallback.
 * 4. Client-side `.ipynb` (Jupyter Notebook v4.5) import/export and `localStorage`
 *    session persistence.
 */
(function () {
  const STORAGE_KEY = 'dart_wasm_notebook_session_v1';

  class WasmNotebookRuntime {
    constructor(options = {}) {
      this.onStatusChange = options.onStatusChange || (() => {});
      this.compilerWorker = null;
      this.compilerReady = false;
      this.compilerReadyPromise = null;
      this._compilerReadyResolve = null;
      this._compilerReadyReject = null;
      this.nativeMathModulePromise = null;
      this.pendingRequests = new Map();
      this.reqCounter = 0;
      this.knownVariables = [];
      this.dartpadWorker = null;
      this.dartpadLspReady = false;
      this.dartpadPending = new Map();
      this.dartpadWorkspaceId = null;
    }

    init() {
      if (this.compilerReadyPromise) {
        return this.compilerReadyPromise;
      }

      this.onStatusChange('busy', 'Loading Wasm Compiler...');

      // 1. Start fetching and compiling native_math.wasm in parallel.
      this.nativeMathModulePromise = (async () => {
        const res = await fetch('native_math.wasm');
        if (!res.ok) {
          throw new Error(`Failed to fetch native_math.wasm (${res.status})`);
        }
        if (typeof WebAssembly.compileStreaming === 'function') {
          try {
            return await WebAssembly.compileStreaming(res.clone());
          } catch (_) {
            // Fallback to ArrayBuffer compilation if MIME type is not application/wasm.
          }
        }
        const buf = await res.arrayBuffer();
        return await WebAssembly.compile(buf);
      })();

      // 2. Spawn the dart2wasm compiler Web Worker.
      this.compilerReadyPromise = new Promise((resolve, reject) => {
        this._compilerReadyResolve = resolve;
        this._compilerReadyReject = reject;
      });

      this.compilerWorker = new Worker('compiler_worker_bootstrap.js', {
        type: 'module',
      });

      this.compilerWorker.onmessage = (event) => {
        const msg = event.data;
        if (!msg || typeof msg.type !== 'string') return;

        if (msg.type === 'status') {
          if (!this.compilerReady) {
            this.onStatusChange(msg.state || 'busy', msg.text || 'Loading...');
          }
          return;
        }

        if (msg.type === 'ready') {
          this.nativeMathModulePromise
            .then(() => {
              this.compilerReady = true;
              this.onStatusChange('connected', 'Wasm Ready (Serverless)');
              if (this._compilerReadyResolve) {
                this._compilerReadyResolve(msg);
              }
            })
            .catch((err) => {
              this.onStatusChange('disconnected', 'Wasm Native Load Failed');
              if (this._compilerReadyReject) {
                this._compilerReadyReject(err);
              }
            });
          return;
        }

        if (msg.type === 'init_error') {
          this.onStatusChange('disconnected', 'Wasm Compiler Error');
          if (this._compilerReadyReject) {
            this._compilerReadyReject(new Error(msg.error || 'Compiler init failed'));
          }
          return;
        }

        if (
          msg.type === 'compile_result' ||
          msg.type === 'analysis_bundle_result'
        ) {
          const pending = this.pendingRequests.get(msg.id);
          if (pending) {
            this.pendingRequests.delete(msg.id);
            pending.resolve(msg);
          }
        }
      };

      this.compilerWorker.onerror = (err) => {
        console.error('Compiler worker error:', err);
        if (!this.compilerReady) {
          this.onStatusChange('disconnected', 'Wasm Worker Error');
        }
      };

      // 3. Optionally start DartPad LSP worker in the background if staged.
      this._tryStartDartPadWorker().catch(() => {});

      return this.compilerReadyPromise;
    }

    async compileCells(cells, targetCellId = null) {
      await this.init();
      const id = `compile-${++this.reqCounter}-${Date.now()}`;
      return new Promise((resolve) => {
        this.pendingRequests.set(id, { resolve });
        this.compilerWorker.postMessage({
          type: 'compile',
          id,
          cells,
          targetCellId,
        });
      });
    }

    async executeCells(cells, targetCellId = null) {
      this.onStatusChange('busy', 'Compiling (Wasm)...');
      const compileRes = await this.compileCells(cells, targetCellId);
      if (!compileRes.ok) {
        this.onStatusChange('connected', 'Wasm Ready (Serverless)');
        return {
          ok: false,
          isCompileError: true,
          error: compileRes.error || 'Compilation failed.',
          diagnostics: compileRes.diagnostics || [],
          compileMs: compileRes.compileMs || 0,
        };
      }

      this.onStatusChange(
        'busy',
        compileRes.cached
          ? 'Executing Wasm (cached)...'
          : `Executing Wasm (${compileRes.compileMs} ms compile)...`,
      );

      const runStart = performance.now();
      try {
        const runPayload = await this._runCompiledCellModule(
          compileRes.wasmBytes,
          compileRes.mjsText,
        );
        const runMs = Math.round(performance.now() - runStart);
        if (runPayload && Array.isArray(runPayload.variables)) {
          this.knownVariables = runPayload.variables;
        }
        this.onStatusChange('connected', 'Wasm Ready (Serverless)');
        return {
          ok: true,
          cached: Boolean(compileRes.cached),
          compileMs: compileRes.compileMs || 0,
          runMs,
          cells: (runPayload && runPayload.cells) || [],
          variables: (runPayload && runPayload.variables) || [],
        };
      } catch (err) {
        this.onStatusChange('connected', 'Wasm Ready (Serverless)');
        return {
          ok: false,
          isCompileError: false,
          error: String(err && err.stack ? err.stack : err),
          compileMs: compileRes.compileMs || 0,
        };
      }
    }

    async _runCompiledCellModule(wasmBytes, mjsText) {
      const nativeModule = await this.nativeMathModulePromise;
      const utf8Decoder = new TextDecoder('utf-8');
      let nativeMemory = null;
      const getView = () => new DataView(nativeMemory.buffer);
      const ENOSYS = 52;
      const EBADF = 8;

      const wasi = {
        environ_sizes_get(countPtr, sizePtr) {
          getView().setUint32(countPtr, 0, true);
          getView().setUint32(sizePtr, 0, true);
          return 0;
        },
        environ_get() {
          return 0;
        },
        clock_time_get(id, precision, outPtr) {
          getView().setBigUint64(
            outPtr,
            BigInt(Math.round(Date.now() * 1e6)),
            true,
          );
          return 0;
        },
        random_get(ptr, len) {
          const arr = new Uint8Array(nativeMemory.buffer, ptr, len);
          if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
            crypto.getRandomValues(arr);
          } else {
            for (let i = 0; i < len; i++) {
              arr[i] = (Math.random() * 256) | 0;
            }
          }
          return 0;
        },
        fd_write(fd, iovsPtr, iovsLen, nwrittenPtr) {
          let total = 0;
          let text = '';
          const v = getView();
          for (let i = 0; i < iovsLen; i++) {
            const ptr = v.getUint32(iovsPtr + i * 8, true);
            const len = v.getUint32(iovsPtr + i * 8 + 4, true);
            text += utf8Decoder.decode(
              new Uint8Array(nativeMemory.buffer, ptr, len),
            );
            total += len;
          }
          if (text) {
            console.debug(`[native_math fd${fd}]`, text);
          }
          v.setUint32(nwrittenPtr, total, true);
          return 0;
        },
        proc_exit(code) {
          throw new Error(`native_math proc_exit(${code})`);
        },
        fd_close() {
          return EBADF;
        },
        fd_fdstat_get() {
          return EBADF;
        },
        fd_fdstat_set_flags() {
          return EBADF;
        },
        fd_pread() {
          return EBADF;
        },
        fd_prestat_get() {
          return EBADF;
        },
        fd_prestat_dir_name() {
          return EBADF;
        },
        fd_read() {
          return EBADF;
        },
        fd_seek() {
          return EBADF;
        },
        path_create_directory() {
          return ENOSYS;
        },
        path_filestat_get() {
          return ENOSYS;
        },
        path_open() {
          return ENOSYS;
        },
      };

      const nativeInstance = await WebAssembly.instantiate(nativeModule, {
        wasi_snapshot_preview1: wasi,
      });
      const nativeExports = nativeInstance.exports;
      nativeMemory = nativeExports.memory;
      if (typeof nativeExports._initialize === 'function') {
        nativeExports._initialize();
      }

      const ffiNamespace = new Proxy(nativeExports, {
        get(target, prop) {
          if (prop in target) return target[prop];
          throw new Error(`Unresolved @ffi.Native symbol: ffi.${String(prop)}`);
        },
      });

      const cellModule = await WebAssembly.compile(wasmBytes);
      const additionalImports = {
        memory: {
          memory: nativeExports.memory,
          malloc: nativeExports.malloc,
          calloc: nativeExports.calloc,
          realloc: nativeExports.realloc,
          free: nativeExports.free,
        },
        wasi_snapshot_preview1: wasi,
      };
      for (const imp of WebAssembly.Module.imports(cellModule)) {
        if (imp.module === 'ffi' || imp.module.startsWith('package:')) {
          additionalImports[imp.module] = ffiNamespace;
        }
      }

      const mjsBlob = new Blob([mjsText], { type: 'text/javascript' });
      const mjsUrl = URL.createObjectURL(mjsBlob);
      try {
        const dartSupport = await import(mjsUrl);
        let resolveReport;
        let rejectReport;
        const reportPromise = new Promise((resolve, reject) => {
          resolveReport = resolve;
          rejectReport = reject;
        });

        const prevReport = globalThis.notebookReportRunResult;
        globalThis.notebookReportRunResult = (jsonStr) => {
          try {
            resolveReport(JSON.parse(String(jsonStr)));
          } catch (e) {
            rejectReport(e);
          }
        };

        try {
          const compiledApp = await dartSupport.compile(wasmBytes);
          const dartInstance = await compiledApp.instantiate(additionalImports);
          dartInstance.invokeMain();
          return await reportPromise;
        } finally {
          globalThis.notebookReportRunResult = prevReport;
        }
      } finally {
        URL.revokeObjectURL(mjsUrl);
      }
    }

    async _tryStartDartPadWorker() {
      // Optional enhancement: if dartpad/worker.mjs and dartpad/dart/sdk.tar
      // are staged, we can connect to the DartPad worker for full LSP analysis.
      const probe = await fetch('dartpad/worker.mjs', { method: 'HEAD' }).catch(
        () => null,
      );
      if (!probe || !probe.ok) return;
      this.hasDartPadAssets = true;
    }

    getCompletions(cells, activeCellId, code, cursorOffset) {
      const textBefore = code.substring(
        0,
        Math.min(cursorOffset, code.length),
      );
      const dotMatch = /([\w\d_$]+)\.([\w\d_$]*)$/.exec(textBefore);
      if (dotMatch) {
        const receiver = dotMatch[1];
        const prefix = dotMatch[2].toLowerCase();
        const items = [];
        if (receiver === 'NDArray') {
          items.push(
            {
              label: 'zeros',
              type: 'constructor',
              detail: 'NDArray.zeros(List<int> shape, [DType dtype])',
            },
            {
              label: 'ones',
              type: 'constructor',
              detail: 'NDArray.ones(List<int> shape, [DType dtype])',
            },
            {
              label: 'fromList',
              type: 'constructor',
              detail: 'NDArray.fromList(List data, List<int> shape, [DType dtype])',
            },
            {
              label: 'arange',
              type: 'constructor',
              detail: 'NDArray.arange(num start, [num? stop, num step = 1])',
            },
            {
              label: 'linspace',
              type: 'constructor',
              detail: 'NDArray.linspace(double start, double stop, int num)',
            },
            {
              label: 'eye',
              type: 'constructor',
              detail: 'NDArray.eye(int n, {int? m, int k = 0, DType dtype})',
            },
          );
        } else if (receiver === 'DType') {
          items.push(
            { label: 'float64', type: 'enum-constant', detail: '64-bit float' },
            { label: 'float32', type: 'enum-constant', detail: '32-bit float' },
            { label: 'int64', type: 'enum-constant', detail: '64-bit int' },
            { label: 'int32', type: 'enum-constant', detail: '32-bit int' },
            { label: 'uint8', type: 'enum-constant', detail: '8-bit uint' },
            { label: 'complex128', type: 'enum-constant', detail: '128-bit complex' },
            { label: 'boolean', type: 'enum-constant', detail: 'boolean' },
          );
        } else {
          items.push(
            { label: 'shape', type: 'property', detail: 'List<int> shape' },
            { label: 'dtype', type: 'property', detail: 'DType dtype' },
            { label: 'ndim', type: 'property', detail: 'int ndim' },
            { label: 'size', type: 'property', detail: 'int size' },
            { label: 'T', type: 'property', detail: 'NDArray transpose' },
            {
              label: 'reshape',
              type: 'method',
              detail: 'NDArray reshape(List<int> newShape)',
            },
            { label: 'toList', type: 'method', detail: 'List toList()' },
            { label: 'copy', type: 'method', detail: 'NDArray copy()' },
            { label: 'scalar', type: 'property', detail: 'Element scalar' },
          );
        }
        return Promise.resolve(
          items.filter((i) => i.label.toLowerCase().startsWith(prefix)),
        );
      }

      const wordMatch = /([\w\d_$]+)$/.exec(textBefore);
      const prefix = wordMatch ? wordMatch[1].toLowerCase() : '';
      const seen = new Set();
      const items = [];

      const addItem = (label, type, detail) => {
        if (!seen.has(label)) {
          seen.add(label);
          items.push({ label, type, detail });
        }
      };

      // User variables from previous cells or runtime inspector
      for (const v of this.knownVariables) {
        addItem(v.name, 'variable', v.summary || v.type);
      }
      for (const cell of cells || []) {
        if (cell.type !== 'code') continue;
        const varRegex = /\b(?:var|final|const)\s+([A-Za-z_$][\w$]*)/g;
        let m;
        while ((m = varRegex.exec(cell.code)) !== null) {
          addItem(m[1], 'variable', 'User defined in notebook');
        }
      }

      const builtins = [
        ['NDArray', 'class', 'Multi-dimensional array'],
        ['DType', 'enum', 'Data type specifier'],
        ['Plot', 'class', '2D SVG line plot widget: Plot(x: xArr, y: yArr)'],
        ['Heatmap', 'class', '2D matrix heatmap widget: Heatmap(matrix2D)'],
        ['Histogram', 'class', 'Histogram bar chart widget: Histogram(data)'],
        ['Table', 'class', 'Interactive data table widget: Table(matrix2D)'],
        ['Image', 'class', 'BMP image widget wrapping 2D/3D NDArray'],
        ['Audio', 'class', '16-bit PCM WAV audio widget wrapping NDArray'],
        ['Spectrogram', 'class', 'STFT frequency spectrogram widget'],
        ['LaTeX', 'class', 'KaTeX mathematical equation widget'],
        ['Markdown', 'class', 'Formatted Markdown display widget'],
        ['display', 'function', 'void display(Object? widgetOrValue)'],
        ['sum', 'function', 'NDArray sum(NDArray a, {int? axis})'],
        ['mean', 'function', 'NDArray mean(NDArray a, {int? axis})'],
        ['std', 'function', 'NDArray std(NDArray a, {int? axis})'],
        ['min', 'function', 'NDArray min(NDArray a, {int? axis})'],
        ['max', 'function', 'NDArray max(NDArray a, {int? axis})'],
        ['dot', 'function', 'NDArray dot(NDArray a, NDArray b)'],
        ['matmul', 'function', 'NDArray matmul(NDArray a, NDArray b)'],
        ['linspace', 'function', 'NDArray<Float64> linspace(double start, double stop, int num)'],
        ['arange', 'function', 'NDArray arange(num start, [num? stop, num step = 1])'],
        ['zeros', 'function', 'NDArray zeros(List<int> shape, {DType dtype})'],
        ['ones', 'function', 'NDArray ones(List<int> shape, {DType dtype})'],
        ['eye', 'function', 'NDArray eye(int n, {int? m, int k = 0, DType dtype})'],
        ['sin', 'function', 'NDArray sin(NDArray a)'],
        ['cos', 'function', 'NDArray cos(NDArray a)'],
        ['exp', 'function', 'NDArray exp(NDArray a)'],
        ['log', 'function', 'NDArray log(NDArray a)'],
        ['sqrt', 'function', 'NDArray sqrt(NDArray a)'],
        ['abs', 'function', 'NDArray abs(NDArray a)'],
        ['fft', 'function', 'NDArray<Complex128> fft(NDArray a)'],
        ['rfft', 'function', 'NDArray<Complex128> rfft(NDArray a)'],
      ];
      for (const [label, type, detail] of builtins) {
        addItem(label, type, detail);
      }

      return Promise.resolve(
        items.filter((i) => i.label.toLowerCase().startsWith(prefix)),
      );
    }

    getHover(code, offset) {
      if (offset < 0 || offset >= code.length) {
        return Promise.resolve({ hover: null, hoverHtml: null });
      }
      let start = offset;
      while (start > 0 && /[\w\d_$]/.test(code[start - 1])) start--;
      let end = offset;
      while (end < code.length && /[\w\d_$]/.test(code[end])) end++;
      const token = code.substring(start, end);
      if (!token) return Promise.resolve({ hover: null, hoverHtml: null });

      for (const v of this.knownVariables) {
        if (v.name === token) {
          const text = `variable ${v.name}: ${v.summary}`;
          return Promise.resolve({
            hover: text,
            hoverHtml: `<code>${this._escapeHtml(v.name)}</code> — ${this._escapeHtml(v.summary)}`,
          });
        }
      }

      const docs = {
        NDArray:
          '<strong>class NDArray&lt;T extends DTypeTag&gt;</strong><br>N-dimensional strided array backed by native Wasm linear memory (OpenBLAS + PocketFFT + SIMD128).',
        DType:
          '<strong>enum DType</strong><br>Element data type for <code>NDArray</code> (<code>float64</code>, <code>float32</code>, <code>int64</code>, <code>int32</code>, <code>complex128</code>, <code>boolean</code>, etc.).',
        fromList:
          '<strong>NDArray.fromList(List data, List&lt;int&gt; shape, [DType dtype])</strong><br>Creates an <code>NDArray</code> from a flat Dart list and target shape.',
        Plot:
          '<strong>class Plot extends Displayable</strong><br>Renders a 2D SVG line chart for <code>NDArray</code> coordinate data.',
        Heatmap:
          '<strong>class Heatmap extends Displayable</strong><br>Renders a 2D color-mapped matrix heatmap for a 2D <code>NDArray</code>.',
        display:
          '<strong>void display(dynamic object)</strong><br>Appends a rich HTML widget or value to the current cell output.',
        matmul:
          '<strong>NDArray matmul(NDArray a, NDArray b)</strong><br>Matrix multiplication accelerated by OpenBLAS in WebAssembly.',
      };

      if (docs[token]) {
        return Promise.resolve({
          hover: token,
          hoverHtml: docs[token],
        });
      }
      return Promise.resolve({ hover: null, hoverHtml: null });
    }

    buildIpynbJson(cellsData) {
      const ipynbCells = (cellsData || []).map((c, idx) => {
        const cellType = c.type === 'markdown' ? 'markdown' : 'code';
        const lines = String(c.code || '').split('\n');
        const sourceLines = lines.map((line, i) =>
          i < lines.length - 1 ? `${line}\n` : line,
        );
        const base = {
          id: c.id || `cell-${idx + 1}`,
          cell_type: cellType,
          metadata: {},
          source: sourceLines,
        };
        if (cellType === 'markdown') {
          return base;
        }
        const outputs = [];
        if (c.isError && c.output) {
          outputs.push({
            output_type: 'error',
            ename: 'ExecutionError',
            evalue: String(c.output).split('\n')[0] || 'Error',
            traceback: String(c.output).split('\n'),
          });
        } else if (Array.isArray(c.outputs) && c.outputs.length > 0) {
          for (const item of c.outputs) {
            const mime = item.mimeType || 'text/plain';
            const dataObj = { [mime]: [String(item.data || '')] };
            if (mime !== 'text/plain') {
              dataObj['text/plain'] = [String(item.data || '')];
            }
            outputs.push({
              output_type: 'execute_result',
              execution_count: c.evaluated ? 1 : null,
              data: dataObj,
              metadata: {},
            });
          }
        } else if (c.output && String(c.output).trim()) {
          outputs.push({
            output_type: 'execute_result',
            execution_count: c.evaluated ? 1 : null,
            data: { 'text/plain': [String(c.output)] },
            metadata: {},
          });
        }
        return {
          ...base,
          execution_count: c.evaluated ? 1 : null,
          outputs,
        };
      });

      const doc = {
        nbformat: 4,
        nbformat_minor: 5,
        metadata: {
          kernelspec: {
            display_name: 'Dart (NDArray Wasm)',
            language: 'dart',
            name: 'dart_wasm',
          },
          language_info: {
            name: 'dart',
            file_extension: '.dart',
            mimetype: 'application/dart',
          },
        },
        cells: ipynbCells,
      };
      return JSON.stringify(doc, null, 2);
    }

    exportIpynbClientSide(cellsData) {
      const jsonStr = this.buildIpynbJson(cellsData);
      window.__lastExportedIpynbJson = jsonStr;
      const blob = new Blob([jsonStr], { type: 'application/x-ipynb+json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'notebook_session.ipynb';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      return jsonStr;
    }

    parseIpynbJsonToSessionCells(jsonText) {
      const raw = JSON.parse(jsonText);
      const rawCells = Array.isArray(raw.cells) ? raw.cells : [];
      return rawCells.map((c, idx) => {
        const cellType = c.cell_type === 'markdown' ? 'markdown' : 'code';
        const src = Array.isArray(c.source)
          ? c.source.join('')
          : String(c.source || '');
        const id = c.id || `cell-imported-${idx + 1}`;
        let isError = false;
        let outputText = '';
        const outputsList = [];

        if (Array.isArray(c.outputs)) {
          for (const out of c.outputs) {
            if (out.output_type === 'error') {
              isError = true;
              outputText = Array.isArray(out.traceback)
                ? out.traceback.join('\n')
                : String(out.evalue || 'Error');
            } else if (out.output_type === 'stream') {
              const txt = Array.isArray(out.text)
                ? out.text.join('')
                : String(out.text || '');
              outputsList.push({ mimeType: 'text/plain', data: txt });
            } else if (out.data && typeof out.data === 'object') {
              const joinVal = (v) => (Array.isArray(v) ? v.join('') : String(v || ''));
              if (out.data['text/html']) {
                outputsList.push({
                  mimeType: 'text/html',
                  data: joinVal(out.data['text/html']),
                });
              } else if (out.data['text/plain']) {
                outputsList.push({
                  mimeType: 'text/plain',
                  data: joinVal(out.data['text/plain']),
                });
              }
            }
          }
        }
        if (!outputText && outputsList.length > 0) {
          outputText = outputsList.map((o) => o.data).join('\n');
        }
        return {
          id,
          type: cellType,
          code: src,
          output: outputText,
          outputs: outputsList,
          isError,
          evaluated: Boolean(c.execution_count) || outputsList.length > 0,
        };
      });
    }

    saveSessionToLocalStorage(cellsData) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(cellsData));
      } catch (_) {}
    }

    loadSessionFromLocalStorage() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      } catch (_) {}
      return null;
    }

    _escapeHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }
  }

  window.WasmNotebookRuntime = WasmNotebookRuntime;
})();
