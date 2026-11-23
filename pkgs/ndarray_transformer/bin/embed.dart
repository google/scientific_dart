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
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray_transformer/ndarray_transformer.dart';

const List<String> _defaultDocs = <String>[
  'Fast numerical array computing and linear algebra in Dart.',
  'How to bake sourdough bread with a crispy crust at home.',
  'Transformer self-attention and sentence embeddings for semantic search.',
  'Real-time Fourier transform pitch detection for acoustic guitars.',
  'Hiking trails and alpine lakes in the Swiss mountains.',
];

Future<void> main(List<String> args) async {
  final parser = ArgParser()
    ..addOption(
      'model-dir',
      help:
          'Path to a local Hugging Face BERT/MiniLM directory containing '
          'config.json, tokenizer.json (or vocab.txt), and model.safetensors.',
    )
    ..addFlag(
      'download',
      negatable: false,
      help:
          'Download $defaultMiniLmRepo from Hugging Face into --model-dir '
          '(or default cache directory) if missing.',
    )
    ..addOption(
      'repo',
      defaultsTo: defaultMiniLmRepo,
      help: 'Hugging Face repository ID used with --download.',
    )
    ..addFlag(
      'demo-synthetic',
      negatable: false,
      help:
          'Run an offline end-to-end demo using a deterministic synthetic '
          'BERT .safetensors model (no network or external weights required).',
    )
    ..addOption(
      'query',
      defaultsTo: 'vector similarity search with neural embeddings',
      help: 'Query sentence to rank against indexed documents.',
    )
    ..addMultiOption(
      'doc',
      help: 'Document text to index (repeatable). Uses sample corpus if empty.',
    )
    ..addOption('top-k', defaultsTo: '5', help: 'Number of matches to show.')
    ..addOption(
      'max-length',
      defaultsTo: '128',
      help: 'Maximum token sequence length.',
    )
    ..addFlag(
      'benchmark',
      negatable: false,
      help: 'Measure warmup + steady-state inference latency.',
    )
    ..addOption(
      'bench-iters',
      defaultsTo: '10',
      help: 'Timed iterations when --benchmark is set.',
    )
    ..addFlag('help', abbr: 'h', negatable: false, help: 'Show usage.');

  final ArgResults parsed;
  try {
    parsed = parser.parse(args);
  } on FormatException catch (e) {
    stderr.writeln('Error: ${e.message}\n');
    stderr.writeln(parser.usage);
    exitCode = 2;
    return;
  }

  if (parsed.flag('help')) {
    stdout.writeln(
      'Pure-Dart BERT / all-MiniLM-L6-v2 embedder & semantic search CLI.\n',
    );
    stdout.writeln(parser.usage);
    return;
  }

  final topK = int.tryParse(parsed.option('top-k')!);
  final maxLength = int.tryParse(parsed.option('max-length')!);
  final benchIters = int.tryParse(parsed.option('bench-iters')!);
  if (topK == null ||
      topK <= 0 ||
      maxLength == null ||
      maxLength < 3 ||
      benchIters == null ||
      benchIters <= 0) {
    stderr.writeln(
      'Error: --top-k (>0), --max-length (>=3), and --bench-iters (>0) must be valid integers.',
    );
    exitCode = 2;
    return;
  }

  final docs = parsed.multiOption('doc').isEmpty
      ? _defaultDocs
      : parsed.multiOption('doc');
  final query = parsed.option('query')!;
  final synthetic = parsed.flag('demo-synthetic');
  final shouldDownload = parsed.flag('download');
  final modelDirOpt = parsed.option('model-dir');

  if (!synthetic && modelDirOpt == null && !shouldDownload) {
    stderr.writeln(
      'Provide --model-dir <path>, --download, or --demo-synthetic.\n',
    );
    stderr.writeln(parser.usage);
    exitCode = 2;
    return;
  }

  late final BertEncoder encoder;
  late final WordPieceTokenizer tokenizer;
  final loadSw = Stopwatch()..start();

  if (synthetic) {
    stdout.writeln(
      'Building deterministic synthetic 2-layer BERT .safetensors model (offline demo)...',
    );
    final synth = _buildSyntheticModel();
    encoder = synth.encoder;
    tokenizer = synth.tokenizer;
  } else {
    final repo = parsed.option('repo')!;
    final safeRepo = repo.replaceAll(RegExp(r'[^a-zA-Z0-9._-]'), '--');
    final home =
        Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'];
    final dirPath =
        modelDirOpt ??
        (home != null
            ? '$home/.cache/scientific_dart/$safeRepo'
            : '${Directory.systemTemp.path}/scientific_dart_$safeRepo');
    final modelDir = Directory(dirPath);
    if (shouldDownload) {
      await downloadHuggingFaceModel(
        targetDir: modelDir,
        repoId: repo,
        onProgress: stdout.writeln,
      );
    }
    final loaded = BertEncoder.fromDirectory(modelDir);
    encoder = loaded.encoder;
    tokenizer = loaded.tokenizer;
  }
  loadSw.stop();

  try {
    final cfg = encoder.config;
    final effectiveMaxLength = math.min(maxLength, cfg.maxPositionEmbeddings);
    if (effectiveMaxLength < maxLength) {
      stderr.writeln(
        'Note: clamping --max-length from $maxLength to model maxPositionEmbeddings ($effectiveMaxLength).',
      );
    }
    stdout.writeln(
      'Loaded BERT encoder (L=${cfg.numHiddenLayers}, H=${cfg.hiddenSize}, '
      'heads=${cfg.numAttentionHeads}, vocab=${cfg.vocabSize}) in '
      '${loadSw.elapsedMilliseconds} ms.',
    );

    final indexSw = Stopwatch()..start();
    final index = SemanticSearchIndex.fromDocuments(
      encoder: encoder,
      tokenizer: tokenizer,
      documents: docs,
      maxLength: effectiveMaxLength,
    );
    indexSw.stop();

    try {
      stdout.writeln(
        'Indexed ${index.length} documents in '
        '${(indexSw.elapsedMicroseconds / 1000.0).toStringAsFixed(2)} ms.\n',
      );

      final querySw = Stopwatch()..start();
      final hits = index.searchText(
        encoder: encoder,
        tokenizer: tokenizer,
        query: query,
        topK: topK,
        maxLength: effectiveMaxLength,
      );
      querySw.stop();

      stdout.writeln('Query: "$query"');
      stdout.writeln(
        'Search latency: ${(querySw.elapsedMicroseconds / 1000.0).toStringAsFixed(2)} ms',
      );
      for (var i = 0; i < hits.length; i++) {
        final hit = hits[i];
        stdout.writeln(
          '  ${i + 1}. [${hit.score.toStringAsFixed(4)}] ${hit.text}',
        );
      }

      if (parsed.flag('benchmark')) {
        stdout.writeln(
          '\nRunning latency benchmark ($benchIters iterations)...',
        );
        final singleBatch = tokenizer.encodeBatch([
          query,
        ], maxLength: effectiveMaxLength);
        final multiBatch = tokenizer.encodeBatch(
          docs,
          maxLength: effectiveMaxLength,
        );

        // Warmup.
        NDArray.scope(() {
          encoder.embed(singleBatch);
          encoder.embed(multiBatch);
        });

        final singleSamples = <double>[];
        for (var i = 0; i < benchIters; i++) {
          final sw = Stopwatch()..start();
          NDArray.scope(() {
            encoder.embed(singleBatch);
          });
          sw.stop();
          singleSamples.add(sw.elapsedMicroseconds / 1000.0);
        }
        singleSamples.sort();

        final multiSamples = <double>[];
        for (var i = 0; i < benchIters; i++) {
          final sw = Stopwatch()..start();
          NDArray.scope(() {
            encoder.embed(multiBatch);
          });
          sw.stop();
          multiSamples.add(sw.elapsedMicroseconds / 1000.0);
        }
        multiSamples.sort();

        final singleMed = singleSamples[singleSamples.length ~/ 2];
        final multiMed = multiSamples[multiSamples.length ~/ 2];
        stdout.writeln(
          '  Single query (B=1, S=${singleBatch.sequenceLength}): '
          'median ${singleMed.toStringAsFixed(2)} ms',
        );
        stdout.writeln(
          '  Batch corpus (B=${multiBatch.batchSize}, S=${multiBatch.sequenceLength}): '
          'median ${multiMed.toStringAsFixed(2)} ms '
          '(${(multiMed / multiBatch.batchSize).toStringAsFixed(2)} ms/doc)',
        );
      }
    } finally {
      index.dispose();
    }
  } finally {
    encoder.dispose();
  }
}

({BertEncoder encoder, WordPieceTokenizer tokenizer}) _buildSyntheticModel() {
  final tokens = <String>[
    '[PAD]',
    '[UNK]',
    '[CLS]',
    '[SEP]',
    'fast',
    'numerical',
    'array',
    'computing',
    'and',
    'linear',
    'algebra',
    'in',
    'dart',
    'how',
    'to',
    'bake',
    'sourdough',
    'bread',
    'with',
    'a',
    'crispy',
    'crust',
    'at',
    'home',
    'transformer',
    'self',
    '-',
    'attention',
    'sentence',
    'embed',
    '##ding',
    '##dings',
    'for',
    'semantic',
    'search',
    'real',
    'time',
    'fourier',
    'pitch',
    'detection',
    'acoustic',
    'guitar',
    '##s',
    'hiking',
    'trails',
    'alpine',
    'lakes',
    'swiss',
    'mountains',
    'vector',
    'similarity',
    'neural',
    '.',
  ];
  final vocab = <String, int>{
    for (var i = 0; i < tokens.length; i++) tokens[i]: i,
  };
  final tokenizer = WordPieceTokenizer.fromVocab(vocab);
  final config = BertConfig(
    vocabSize: vocab.length,
    hiddenSize: 64,
    numHiddenLayers: 2,
    numAttentionHeads: 4,
    intermediateSize: 128,
    maxPositionEmbeddings: 64,
  );

  final rng = math.Random(42);
  NDArray<Float32> randWeight(List<int> shape, double scale) {
    final size = shape.fold<int>(1, (a, b) => a * b);
    final data = Float32List(size);
    for (var i = 0; i < size; i++) {
      data[i] = (rng.nextDouble() * 2.0 - 1.0) * scale;
    }
    return NDArray<Float32>.fromTypedData(data, shape, DType.float32);
  }

  final bytes = NDArray.scope(() {
    final h = config.hiddenSize;
    final inter = config.intermediateSize;
    final tensors = <String, NDArray<DTypeTag>>{
      'embeddings.word_embeddings.weight': randWeight([
        config.vocabSize,
        h,
      ], 0.2),
      'embeddings.position_embeddings.weight': randWeight([
        config.maxPositionEmbeddings,
        h,
      ], 0.05),
      'embeddings.token_type_embeddings.weight': randWeight([
        config.typeVocabSize,
        h,
      ], 0.02),
      'embeddings.LayerNorm.weight': NDArray<Float32>.ones([h], DType.float32),
      'embeddings.LayerNorm.bias': NDArray<Float32>.zeros([h], DType.float32),
    };
    for (var i = 0; i < config.numHiddenLayers; i++) {
      final p = 'encoder.layer.$i';
      tensors['$p.attention.self.query.weight'] = randWeight([h, h], 0.08);
      tensors['$p.attention.self.query.bias'] = NDArray<Float32>.zeros([
        h,
      ], DType.float32);
      tensors['$p.attention.self.key.weight'] = randWeight([h, h], 0.08);
      tensors['$p.attention.self.key.bias'] = NDArray<Float32>.zeros([
        h,
      ], DType.float32);
      tensors['$p.attention.self.value.weight'] = randWeight([h, h], 0.08);
      tensors['$p.attention.self.value.bias'] = NDArray<Float32>.zeros([
        h,
      ], DType.float32);
      tensors['$p.attention.output.dense.weight'] = randWeight([h, h], 0.08);
      tensors['$p.attention.output.dense.bias'] = NDArray<Float32>.zeros([
        h,
      ], DType.float32);
      tensors['$p.attention.output.LayerNorm.weight'] = NDArray<Float32>.ones([
        h,
      ], DType.float32);
      tensors['$p.attention.output.LayerNorm.bias'] = NDArray<Float32>.zeros([
        h,
      ], DType.float32);
      tensors['$p.intermediate.dense.weight'] = randWeight([inter, h], 0.08);
      tensors['$p.intermediate.dense.bias'] = NDArray<Float32>.zeros([
        inter,
      ], DType.float32);
      tensors['$p.output.dense.weight'] = randWeight([h, inter], 0.08);
      tensors['$p.output.dense.bias'] = NDArray<Float32>.zeros([
        h,
      ], DType.float32);
      tensors['$p.output.LayerNorm.weight'] = NDArray<Float32>.ones([
        h,
      ], DType.float32);
      tensors['$p.output.LayerNorm.bias'] = NDArray<Float32>.zeros([
        h,
      ], DType.float32);
    }
    return SafetensorsFile.encode(tensors, metadata: {'format': 'pt'});
  });

  final st = SafetensorsFile.fromBytes(bytes);
  final weights = BertWeights.fromSafetensors(st, config);
  return (
    encoder: BertEncoder(config: config, weights: weights),
    tokenizer: tokenizer,
  );
}
