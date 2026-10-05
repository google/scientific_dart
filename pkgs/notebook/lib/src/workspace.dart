// ignore_for_file: unused_import, unused_element, non_constant_identifier_names
// Auto-generated workspace. Do not edit.
import 'dart:math' as math;
import 'package:notebook/src/kernel_helper.dart';
import 'package:ndarray/ndarray.dart';
import 'package:symbolic_dart/symbolic_dart.dart'
    hide sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, abs;
import 'package:resource_scope/resource_scope.dart';
import 'package:gpuarray/gpuarray.dart'
    show
        GpuArray,
        GpuDevice,
        GpuBuffer,
        GpuBackend,
        GpuDeviceType,
        GpuMemoryPool,
        NDArrayGpuInterop,
        GpuArrayNDArrayInterop;

late NDArray<Float64> xString;
T __set_xString<T extends NDArray<Float64>>(T v) {
  xString = v;
  return v;
}

late NDArray<Float64> mode1;
T __set_mode1<T extends NDArray<Float64>>(T v) {
  mode1 = v;
  return v;
}

late NDArray<Float64> mode2;
T __set_mode2<T extends NDArray<Float64>>(T v) {
  mode2 = v;
  return v;
}

late int sampleRate;
T __set_sampleRate<T extends int>(T v) {
  sampleRate = v;
  return v;
}

late NDArray<Float64> tNote;
T __set_tNote<T extends NDArray<Float64>>(T v) {
  tNote = v;
  return v;
}

late double f0;
T __set_f0<T extends double>(T v) {
  f0 = v;
  return v;
}

dynamic fadeOut;
T __set_fadeOut<T>(T v) {
  fadeOut = v;
  return v;
}

dynamic guitarChord;
T __set_guitarChord<T>(T v) {
  guitarChord = v;
  return v;
}

dynamic uBeads;
T __set_uBeads<T>(T v) {
  uBeads = v;
  return v;
}

dynamic nBeads;
T __set_nBeads<T>(T v) {
  nBeads = v;
  return v;
}

dynamic left;
T __set_left<T>(T v) {
  left = v;
  return v;
}

dynamic center;
T __set_center<T>(T v) {
  center = v;
  return v;
}

dynamic right;
T __set_right<T>(T v) {
  right = v;
  return v;
}

dynamic pull;
T __set_pull<T>(T v) {
  pull = v;
  return v;
}

late int nTiny;
T __set_nTiny<T extends int>(T v) {
  nTiny = v;
  return v;
}

dynamic mainDiag5;
T __set_mainDiag5<T>(T v) {
  mainDiag5 = v;
  return v;
}

dynamic offDiag5;
T __set_offDiag5<T>(T v) {
  offDiag5 = v;
  return v;
}

late NDArray<Float64> l1dTiny;
T __set_l1dTiny<T extends NDArray<Float64>>(T v) {
  l1dTiny = v;
  return v;
}

dynamic interiorBeads;
T __set_interiorBeads<T>(T v) {
  interiorBeads = v;
  return v;
}

dynamic matrixPull;
T __set_matrixPull<T>(T v) {
  matrixPull = v;
  return v;
}

late int nString;
T __set_nString<T extends int>(T v) {
  nString = v;
  return v;
}

late double hString;
T __set_hString<T extends double>(T v) {
  hString = v;
  return v;
}

late NDArray<Float64> l1dString;
T __set_l1dString<T extends NDArray<Float64>>(T v) {
  l1dString = v;
  return v;
}

dynamic stringEig;
T __set_stringEig<T>(T v) {
  stringEig = v;
  return v;
}

late NDArray<Float64> stringLambdas;
T __set_stringLambdas<T extends NDArray<Float64>>(T v) {
  stringLambdas = v;
  return v;
}

late NDArray<Float64> stringModes;
T __set_stringModes<T extends NDArray<Float64>>(T v) {
  stringModes = v;
  return v;
}

late double lambda1;
T __set_lambda1<T extends double>(T v) {
  lambda1 = v;
  return v;
}

dynamic xGrid30;
T __set_xGrid30<T>(T v) {
  xGrid30 = v;
  return v;
}

late int nSmall;
T __set_nSmall<T extends int>(T v) {
  nSmall = v;
  return v;
}

dynamic l1dSmall;
T __set_l1dSmall<T>(T v) {
  l1dSmall = v;
  return v;
}

dynamic eyeSmall;
T __set_eyeSmall<T>(T v) {
  eyeSmall = v;
  return v;
}

dynamic l2dSmall;
T __set_l2dSmall<T>(T v) {
  l2dSmall = v;
  return v;
}

late int nSquare;
T __set_nSquare<T extends int>(T v) {
  nSquare = v;
  return v;
}

late double hSquare;
T __set_hSquare<T extends double>(T v) {
  hSquare = v;
  return v;
}

late NDArray<Float64> l2dSquare;
T __set_l2dSquare<T extends NDArray<Float64>>(T v) {
  l2dSquare = v;
  return v;
}

dynamic squareEig;
T __set_squareEig<T>(T v) {
  squareEig = v;
  return v;
}

late NDArray<Float64> squareLambdas;
T __set_squareLambdas<T extends NDArray<Float64>>(T v) {
  squareLambdas = v;
  return v;
}

late NDArray<Float64> squareModes;
T __set_squareModes<T extends NDArray<Float64>>(T v) {
  squareModes = v;
  return v;
}

late double sqLambda1;
T __set_sqLambda1<T extends double>(T v) {
  sqLambda1 = v;
  return v;
}

dynamic mode1Grid;
T __set_mode1Grid<T>(T v) {
  mode1Grid = v;
  return v;
}

dynamic mode4Grid;
T __set_mode4Grid<T>(T v) {
  mode4Grid = v;
  return v;
}

NDArray<Float64> colorizeModeTile(NDArray<Float64> mode2D) {
  return NDArray.scope(() {
    final maxVal = max(abs(mode2D)).scalar;
    final u = mode2D * (1.0 / (maxVal > 1e-12 ? maxVal : 1.0));
    final nodal = exp((u * u) * -35.0);
    final pos = (u + abs(u)) * 0.5;
    final neg = (abs(u) - u) * 0.5;
    final red = pad(
      pos * 0.82 + nodal * 0.88 + 0.10,
      PadWidth.all(2),
      constantValues: PadValues<Float64>.all(0.06),
    );
    final green = pad(
      pos * 0.25 + neg * 0.45 + nodal * 0.78 + 0.10,
      PadWidth.all(2),
      constantValues: PadValues<Float64>.all(0.06),
    );
    final blue = pad(
      neg * 0.85 + nodal * 0.45 + 0.14,
      PadWidth.all(2),
      constantValues: PadValues<Float64>.all(0.09),
    );
    return stack([red, green, blue], axis: 2).detachToParentScope();
  });
}

dynamic squareGallery;
T __set_squareGallery<T>(T v) {
  squareGallery = v;
  return v;
}

late int nGeom;
T __set_nGeom<T extends int>(T v) {
  nGeom = v;
  return v;
}

late double radius;
T __set_radius<T extends double>(T v) {
  radius = v;
  return v;
}

late double hGeom;
T __set_hGeom<T extends double>(T v) {
  hGeom = v;
  return v;
}

dynamic geomCoords;
T __set_geomCoords<T>(T v) {
  geomCoords = v;
  return v;
}

late NDArray<Float64> yCoord;
T __set_yCoord<T extends NDArray<Float64>>(T v) {
  yCoord = v;
  return v;
}

late NDArray<Float64> xCoord;
T __set_xCoord<T extends NDArray<Float64>>(T v) {
  xCoord = v;
  return v;
}

dynamic distSquared;
T __set_distSquared<T>(T v) {
  distSquared = v;
  return v;
}

late NDArray<Boolean> circleMask;
T __set_circleMask<T extends NDArray<Boolean>>(T v) {
  circleMask = v;
  return v;
}

late NDArray<Boolean> ellipseMask;
T __set_ellipseMask<T extends NDArray<Boolean>>(T v) {
  ellipseMask = v;
  return v;
}

late NDArray<Boolean> ringMask;
T __set_ringMask<T extends NDArray<Boolean>>(T v) {
  ringMask = v;
  return v;
}

dynamic inBox;
T __set_inBox<T>(T v) {
  inBox = v;
  return v;
}

late NDArray<Boolean> lShapeMask;
T __set_lShapeMask<T extends NDArray<Boolean>>(T v) {
  lShapeMask = v;
  return v;
}

dynamic stadiumRect;
T __set_stadiumRect<T>(T v) {
  stadiumRect = v;
  return v;
}

dynamic stadiumLeftCap;
T __set_stadiumLeftCap<T>(T v) {
  stadiumLeftCap = v;
  return v;
}

dynamic stadiumRightCap;
T __set_stadiumRightCap<T>(T v) {
  stadiumRightCap = v;
  return v;
}

late NDArray<Boolean> stadiumMask;
T __set_stadiumMask<T extends NDArray<Boolean>>(T v) {
  stadiumMask = v;
  return v;
}

dynamic leftBell;
T __set_leftBell<T>(T v) {
  leftBell = v;
  return v;
}

dynamic rightBell;
T __set_rightBell<T>(T v) {
  rightBell = v;
  return v;
}

dynamic narrowNeck;
T __set_narrowNeck<T>(T v) {
  narrowNeck = v;
  return v;
}

late NDArray<Boolean> peanutMask;
T __set_peanutMask<T extends NDArray<Boolean>>(T v) {
  peanutMask = v;
  return v;
}

dynamic vSep;
T __set_vSep<T>(T v) {
  vSep = v;
  return v;
}

dynamic topRowMasks;
T __set_topRowMasks<T>(T v) {
  topRowMasks = v;
  return v;
}

dynamic botRowMasks;
T __set_botRowMasks<T>(T v) {
  botRowMasks = v;
  return v;
}

dynamic hSep;
T __set_hSep<T>(T v) {
  hSep = v;
  return v;
}

dynamic allMasksGrid;
T __set_allMasksGrid<T>(T v) {
  allMasksGrid = v;
  return v;
}

late NDArray<Float64> l2dGeom;
T __set_l2dGeom<T extends NDArray<Float64>>(T v) {
  l2dGeom = v;
  return v;
}

({NDArray<Float64> eigenvalues, NDArray<Float64> eigenvectors}) solveMaskedDrum(
  NDArray<Boolean> mask,
) {
  final flat = mask.flatten();
  final subMatrix = l2dGeom.slice([
    Mask(BooleanMask(flat)),
    Mask(BooleanMask(flat)),
  ]);
  return eigh(subMatrix);
}

dynamic circleEig;
T __set_circleEig<T>(T v) {
  circleEig = v;
  return v;
}

dynamic ellipseEig;
T __set_ellipseEig<T>(T v) {
  ellipseEig = v;
  return v;
}

dynamic ringEig;
T __set_ringEig<T>(T v) {
  ringEig = v;
  return v;
}

dynamic lShapeEig;
T __set_lShapeEig<T>(T v) {
  lShapeEig = v;
  return v;
}

dynamic stadiumEig;
T __set_stadiumEig<T>(T v) {
  stadiumEig = v;
  return v;
}

dynamic peanutEig;
T __set_peanutEig<T>(T v) {
  peanutEig = v;
  return v;
}

late NDArray<Float64> circleLambdas;
T __set_circleLambdas<T extends NDArray<Float64>>(T v) {
  circleLambdas = v;
  return v;
}

late NDArray<Float64> circleModes;
T __set_circleModes<T extends NDArray<Float64>>(T v) {
  circleModes = v;
  return v;
}

late NDArray<Float64> ellipseLambdas;
T __set_ellipseLambdas<T extends NDArray<Float64>>(T v) {
  ellipseLambdas = v;
  return v;
}

late NDArray<Float64> ellipseModes;
T __set_ellipseModes<T extends NDArray<Float64>>(T v) {
  ellipseModes = v;
  return v;
}

late NDArray<Float64> ringLambdas;
T __set_ringLambdas<T extends NDArray<Float64>>(T v) {
  ringLambdas = v;
  return v;
}

late NDArray<Float64> ringModes;
T __set_ringModes<T extends NDArray<Float64>>(T v) {
  ringModes = v;
  return v;
}

late NDArray<Float64> lShapeLambdas;
T __set_lShapeLambdas<T extends NDArray<Float64>>(T v) {
  lShapeLambdas = v;
  return v;
}

late NDArray<Float64> lShapeModes;
T __set_lShapeModes<T extends NDArray<Float64>>(T v) {
  lShapeModes = v;
  return v;
}

late NDArray<Float64> stadiumLambdas;
T __set_stadiumLambdas<T extends NDArray<Float64>>(T v) {
  stadiumLambdas = v;
  return v;
}

late NDArray<Float64> stadiumModes;
T __set_stadiumModes<T extends NDArray<Float64>>(T v) {
  stadiumModes = v;
  return v;
}

late NDArray<Float64> peanutLambdas;
T __set_peanutLambdas<T extends NDArray<Float64>>(T v) {
  peanutLambdas = v;
  return v;
}

late NDArray<Float64> peanutModes;
T __set_peanutModes<T extends NDArray<Float64>>(T v) {
  peanutModes = v;
  return v;
}

dynamic peanutMode2;
T __set_peanutMode2<T>(T v) {
  peanutMode2 = v;
  return v;
}

NDArray<Float64> renderMaskedDrumGallery(
  NDArray<Float64> eigVecs,
  NDArray<Boolean> mask,
) {
  return NDArray.scope(() {
    final maskF = mask.astype(DType.float64);
    final outsideF = (maskF * -1.0) + 1.0;
    final rows = <NDArray<Float64>>[];
    for (var r = 0; r < 2; r++) {
      final rowTiles = <NDArray<Float64>>[];
      for (var c = 0; c < 3; c++) {
        final k = r * 3 + c;
        final mode2D = NDArray.zeros([nGeom, nGeom], DType.float64);
        mode2D[mask] = eigVecs.slice([Slice.all(), Index(k)]);
        final maxVal = max(abs(mode2D)).scalar;
        final u = mode2D * (1.0 / (maxVal > 1e-12 ? maxVal : 1.0));
        final nodal = exp((u * u) * -35.0) * maskF;
        final pos = (u + abs(u)) * 0.5;
        final neg = (abs(u) - u) * 0.5;
        final red = pad(
          (pos * 0.82 + nodal * 0.88 + 0.12) * maskF + outsideF * 0.05,
          PadWidth.all(2),
          constantValues: PadValues<Float64>.all(0.03),
        );
        final green = pad(
          (pos * 0.25 + neg * 0.45 + nodal * 0.78 + 0.12) * maskF +
              outsideF * 0.05,
          PadWidth.all(2),
          constantValues: PadValues<Float64>.all(0.03),
        );
        final blue = pad(
          (neg * 0.85 + nodal * 0.45 + 0.16) * maskF + outsideF * 0.08,
          PadWidth.all(2),
          constantValues: PadValues<Float64>.all(0.05),
        );
        rowTiles.add(stack([red, green, blue], axis: 2));
      }
      rows.add(concatenate(rowTiles, axis: 1));
    }
    final grid = concatenate(rows, axis: 0);
    return repeat(repeat(grid, 3, axis: 0), 3, axis: 1).detachToParentScope();
  });
}

dynamic freqRows;
T __set_freqRows<T>(T v) {
  freqRows = v;
  return v;
}

dynamic allLambdaLists;
T __set_allLambdaLists<T>(T v) {
  allLambdaLists = v;
  return v;
}

late int nWeyl;
T __set_nWeyl<T extends int>(T v) {
  nWeyl = v;
  return v;
}

late NDArray<Float64> kAll;
T __set_kAll<T extends NDArray<Float64>>(T v) {
  kAll = v;
  return v;
}

late NDArray<Float64> kTail;
T __set_kTail<T extends NDArray<Float64>>(T v) {
  kTail = v;
  return v;
}

double estimateAreaFromSpectrum(NDArray<Float64> lambdas) {
  return mean(
    (kTail * (4 * math.pi)) / lambdas.slice([Slice(start: 14, stop: nWeyl)]),
  ).scalar;
}

dynamic drumAreaData;
T __set_drumAreaData<T>(T v) {
  drumAreaData = v;
  return v;
}

dynamic weylRows;
T __set_weylRows<T>(T v) {
  weylRows = v;
  return v;
}

dynamic trueAreaRing;
T __set_trueAreaRing<T>(T v) {
  trueAreaRing = v;
  return v;
}

dynamic trueAreaPeanut;
T __set_trueAreaPeanut<T>(T v) {
  trueAreaPeanut = v;
  return v;
}

dynamic waveAnimation;
T __set_waveAnimation<T>(T v) {
  waveAnimation = v;
  return v;
}

late int numSynthModes;
T __set_numSynthModes<T extends int>(T v) {
  numSynthModes = v;
  return v;
}

late double baseFreqHz;
T __set_baseFreqHz<T extends double>(T v) {
  baseFreqHz = v;
  return v;
}

late double refLambda0;
T __set_refLambda0<T extends double>(T v) {
  refLambda0 = v;
  return v;
}

NDArray<Float64> synthesizeDrumStrike({
  required NDArray<Float64> eigenvalues,
  required NDArray<Float64> eigenvectors,
  required NDArray<Boolean> mask,
  required double xStrike,
  required double yStrike,
  required double durationSec,
  required bool sameMembraneTension,
  double malletSharpness = 55.0,
}) {
  return NDArray.scope(() {
    final int nSamp = (sampleRate * durationSec).round();
    final tVec = linspace(
      0.0,
      durationSec,
      nSamp,
      endpoint: false,
      dtype: DType.float64,
    );
    final distSqStrike =
        (xCoord - xStrike) * (xCoord - xStrike) +
        (yCoord - yStrike) * (yCoord - yStrike);
    final mallet2D = exp(distSqStrike * (-malletSharpness));
    final malletInterior = mallet2D[mask] as NDArray<Float64>;
    final nDof = malletInterior.shape[0];
    final nModes = math.min(numSynthModes, nDof);
    final eigVecsM = eigenvectors.slice([Slice.all(), Slice(stop: nModes)]);
    final rawCoupling = abs(
      matmul(malletInterior.reshape([1, nDof]), eigVecsM).flatten(),
    );
    final maxCoupling = max(rawCoupling).scalar;
    final normCoupling =
        rawCoupling * (1.0 / (maxCoupling > 1e-12 ? maxCoupling : 1.0));
    final modeIndex = linspace(
      0.0,
      (nModes - 1).toDouble(),
      nModes,
      dtype: DType.float64,
    );
    final overtoneBoost = ((exp(modeIndex * -0.8) * -1.0) + 1.0) * 0.95 + 0.45;
    final weights = (normCoupling * overtoneBoost).reshape([1, nModes]);
    final lambdasM = eigenvalues.slice([Slice(stop: nModes)]);
    final freqRatios = sqrt(lambdasM * (1.0 / lambdasM.getCell([0])));
    final freqsHz = sameMembraneTension
        ? sqrt(lambdasM * (1.0 / refLambda0)) * 140.0
        : freqRatios * baseFreqHz;
    final omegaCol = (freqsHz * (2.0 * math.pi)).reshape([nModes, 1]);
    final dampCol = ((freqRatios - 1.0) * 0.38 + 0.85).reshape([nModes, 1]);
    final tRow = tVec.reshape([1, nSamp]);
    final attackEnv = (exp(tRow * -350.0) * -1.0) + 1.0;
    final modeWaves =
        sin(omegaCol * tRow) * exp((dampCol * -1.0) * tRow) * attackEnv;
    final rawSignal = matmul(weights, modeWaves).flatten();
    final maxAmp = max(abs(rawSignal)).scalar;
    return (rawSignal * (0.88 / (maxAmp > 1e-12 ? maxAmp : 1.0)))
        .detachToParentScope();
  });
}

dynamic tourGap;
T __set_tourGap<T>(T v) {
  tourGap = v;
  return v;
}

dynamic sameMembraneTour;
T __set_sameMembraneTour<T>(T v) {
  sameMembraneTour = v;
  return v;
}

late double clipDur;
T __set_clipDur<T extends double>(T v) {
  clipDur = v;
  return v;
}

late int numSamples;
T __set_numSamples<T extends int>(T v) {
  numSamples = v;
  return v;
}

late NDArray<Float64> circleAudio;
T __set_circleAudio<T extends NDArray<Float64>>(T v) {
  circleAudio = v;
  return v;
}

late NDArray<Float64> ellipseAudio;
T __set_ellipseAudio<T extends NDArray<Float64>>(T v) {
  ellipseAudio = v;
  return v;
}

late NDArray<Float64> ringAudio;
T __set_ringAudio<T extends NDArray<Float64>>(T v) {
  ringAudio = v;
  return v;
}

late NDArray<Float64> lShapeAudio;
T __set_lShapeAudio<T extends NDArray<Float64>>(T v) {
  lShapeAudio = v;
  return v;
}

late NDArray<Float64> stadiumAudio;
T __set_stadiumAudio<T extends NDArray<Float64>>(T v) {
  stadiumAudio = v;
  return v;
}

late NDArray<Float64> peanutAudio;
T __set_peanutAudio<T extends NDArray<Float64>>(T v) {
  peanutAudio = v;
  return v;
}

dynamic equalPitchTour;
T __set_equalPitchTour<T>(T v) {
  equalPitchTour = v;
  return v;
}

dynamic fftPlot;
T __set_fftPlot<T>(T v) {
  fftPlot = v;
  return v;
}
