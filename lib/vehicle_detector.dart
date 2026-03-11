//vehicle_detector.dart

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class VehicleDetector {
  late Interpreter interpreter;
  /// Class labels for the YOLO model (2 classes: vehicle and plate)
  List<String> labels = ['vehicle', 'plate number'];
  
  /// Vehicle class ID
  static const int vehicleClassId = 0;
  /// Plate class ID
  static const int plateClassId = 1;

  Future<void> loadModel() async {
  interpreter = await Interpreter.fromAsset('assets/models/detectlatest.tflite');
}


  /// Run detection on an image file path. Returns a list of detections where each
  /// detection is a map: {"xmin","ymin","xmax","ymax","score","classId","label"}
  /// Coordinates are normalized in [0,1].
  Future<List<Map<String, dynamic>>> detect(String imagePath, {double confThreshold = 0.25}) async {
    print('VehicleDetector.detect: starting for $imagePath');
    final bytes = await File(imagePath).readAsBytes();
    final original = img.decodeImage(bytes);
    if (original == null) return [];

    final inputShape = interpreter.getInputTensor(0).shape; // [1,h,w,3]
    final inputH = inputShape.length > 1 ? inputShape[1] : 0;
    final inputW = inputShape.length > 2 ? inputShape[2] : 0;
    if (inputH == 0 || inputW == 0) return [];
    print('VehicleDetector: input shape [1, $inputH, $inputW, 3]');
    // Letterbox resize: preserve aspect ratio, pad to input size and remember scale/pad
    final origW = original.width.toDouble();
    final origH = original.height.toDouble();
    final scale = min(inputW / origW, inputH / origH);
    final resizedW = (origW * scale).round();
    final resizedH = (origH * scale).round();
    final resized = img.copyResize(original, width: resizedW, height: resizedH);

    // create padded regions (letterbox) and compute per-pixel input values
    final padX = ((inputW - resizedW) / 2).round();
    final padY = ((inputH - resizedH) / 2).round();
    // pad color 114 (mid-gray)
    const padR = 114;
    const padG = 114;
    const padB = 114;

    // IMPORTANT: Avoid building a deeply nested Dart List for input.
    // A [1,H,W,3] nested list causes massive allocations and can freeze the UI.
    final inputBuffer = Float32List(inputH * inputW * 3);
    var idx = 0;
    for (int y = 0; y < inputH; y++) {
      for (int x = 0; x < inputW; x++) {
        // check if this pixel falls inside the pasted resized image
        final sx = x - padX;
        final sy = y - padY;
        int r, g, b;
        if (sx >= 0 && sx < resizedW && sy >= 0 && sy < resizedH) {
          final px = resized.getPixel(sx, sy);
          r = px.r.toInt();
          g = px.g.toInt();
          b = px.b.toInt();
        } else {
          r = padR;
          g = padG;
          b = padB;
        }

        inputBuffer[idx++] = r / 255.0;
        inputBuffer[idx++] = g / 255.0;
        inputBuffer[idx++] = b / 255.0;
      }
    }
    final input = inputBuffer.reshape([1, inputH, inputW, 3]);

    // Output often looks like [1, C, N] (channels-first) for YOLO exports.
    // Some exports use [1, N, C] (channels-last). Mis-detecting this causes
    // nonsense boxes/scores and frequent false positives.
    final outShape = interpreter.getOutputTensor(0).shape;
    final outCount = outShape.reduce((a, b) => a * b);
    final output = List.filled(outCount, 0.0).reshape(outShape);
    interpreter.run(input, output);

    if (outShape.length != 3) {
      print('VehicleDetector: unsupported output shape $outShape');
      return [];
    }

    final detections = <Map<String, dynamic>>[];

    final dim1 = outShape[1];
    final dim2 = outShape[2];
    bool looksLikeChannels(int d) => d > 0 && d <= 64;

    final channelsFirst = (looksLikeChannels(dim1) && !looksLikeChannels(dim2))
        ? true
        : (looksLikeChannels(dim2) && !looksLikeChannels(dim1))
            ? false
            : dim1 < dim2; // fallback: smaller dimension is usually channels

    final channelCount = channelsFirst ? dim1 : dim2;
    final count = channelsFirst ? dim2 : dim1;

    print(
      'VehicleDetector: output shape $outShape, channelsFirst=$channelsFirst, channels=$channelCount, detections=$count',
    );

    final numClasses = labels.length;
    if (channelCount < 4 + numClasses) {
      print(
        'VehicleDetector: output has too few channels ($channelCount) for $numClasses classes',
      );
      return [];
    }

    // Generic YOLO-like decoding:
    // - first 4 channels are box (cx, cy, w, h)
    // - last `numClasses` channels are class logits/probs
    // - optional objectness channel sits right before the class channels
    final classStart = channelCount - numClasses;
    final objChannel = classStart - 1;
    final hasObj = objChannel >= 4;

    double sigmoid(double x) => 1.0 / (1.0 + exp(-x));

    double normOrSigmoid(double v) {
      // Many TFLite exports already output normalized values in [0, 1].
      // If it's outside that range, treat it as a logit.
      if (v >= 0.0 && v <= 1.0) return v;
      return sigmoid(v);
    }

    double getVal(int c, int i) {
      if (channelsFirst) {
        final v = output[0][c][i];
        return (v is num) ? v.toDouble() : double.tryParse(v.toString()) ?? 0.0;
      } else {
        final v = output[0][i][c];
        return (v is num) ? v.toDouble() : double.tryParse(v.toString()) ?? 0.0;
      }
    }

    // collect raw boxes & scores
    for (int i = 0; i < count; i++) {
      double rawCx = getVal(0, i);
      double rawCy = getVal(1, i);
      double rawW = getVal(2, i);
      double rawH = getVal(3, i);

      final cx = normOrSigmoid(rawCx);
      final cy = normOrSigmoid(rawCy);

      final w = normOrSigmoid(rawW.abs());
      final h = normOrSigmoid(rawH.abs());

      final obj = hasObj ? normOrSigmoid(getVal(objChannel, i)) : 1.0;

      int bestClass = -1;
      double bestScore = -double.infinity;
      double bestClassProb = 0.0;

      for (int c = 0; c < numClasses; c++) {
        final clsProb = normOrSigmoid(getVal(classStart + c, i));
        final score = obj * clsProb;
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
          bestClassProb = clsProb;
        }
      }

      if (bestScore < confThreshold) continue;
      
      // Debug: log detections with valid class IDs
      print(
        'Detection: classId=$bestClass, score=${bestScore.toStringAsFixed(3)}, obj=${obj.toStringAsFixed(3)}, classProb=${bestClassProb.toStringAsFixed(3)}',
      );

      // Map box from letterboxed input coordinates back to original image normalized coords
      // Model likely outputs normalized coords relative to the input size (including padding)
      final xCenterInput = cx * inputW;
      final yCenterInput = cy * inputH;
      final wInput = w * inputW;
      final hInput = h * inputH;

      final xCenterUnpad = xCenterInput - padX;
      final yCenterUnpad = yCenterInput - padY;
      final xCenterOrig = xCenterUnpad / scale;
      final yCenterOrig = yCenterUnpad / scale;
      final wOrig = wInput / scale;
      final hOrig = hInput / scale;

      double xmin = (xCenterOrig - wOrig / 2) / origW;
      double ymin = (yCenterOrig - hOrig / 2) / origH;
      double xmax = (xCenterOrig + wOrig / 2) / origW;
      double ymax = (yCenterOrig + hOrig / 2) / origH;

      xmin = xmin.clamp(0.0, 1.0);
      ymin = ymin.clamp(0.0, 1.0);
      xmax = xmax.clamp(0.0, 1.0);
      ymax = ymax.clamp(0.0, 1.0);

      final label = (bestClass >= 0 && bestClass < labels.length) ? labels[bestClass] : 'cls_$bestClass';

      detections.add({
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        'score': bestScore,
        'classId': bestClass,
        'label': label,
        // helpful metadata
        'raw_cx': rawCx,
        'raw_cy': rawCy,
        'raw_w': rawW,
        'raw_h': rawH,
      });
    }

    // Apply Non-Maximum Suppression (NMS)
    List<Map<String, dynamic>> nms(List<Map<String, dynamic>> dets, {double iouThresh = 0.45}) {
      dets.sort((a, b) => (b['score'] as double).compareTo(a['score'] as double));
      final picked = <Map<String, dynamic>>[];
      bool iouThreshFunc(Map<String, dynamic> a, Map<String, dynamic> b) {
        final double ax1 = a['xmin'];
        final double ay1 = a['ymin'];
        final double ax2 = a['xmax'];
        final double ay2 = a['ymax'];
        final double bx1 = b['xmin'];
        final double by1 = b['ymin'];
        final double bx2 = b['xmax'];
        final double by2 = b['ymax'];
        final double interW = max(0.0, min(ax2, bx2) - max(ax1, bx1));
        final double interH = max(0.0, min(ay2, by2) - max(ay1, by1));
        final double inter = interW * interH;
        final double areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1);
        final double areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1);
        final double union = areaA + areaB - inter;
        final double iou = union > 0 ? inter / union : 0.0;
        return iou > iouThresh;
      }

      for (final d in dets) {
        bool keep = true;
        for (final p in picked) {
          if (iouThreshFunc(d, p)) {
            keep = false;
            break;
          }
        }
        if (keep) picked.add(d);
      }
      return picked;
    }

    final finalDetections = nms(detections, iouThresh: 0.45);
    finalDetections.sort((a, b) => (b['score'] as double).compareTo(a['score'] as double));
    
    // Print detection summary
    _printDetectionSummary(finalDetections);
    
    return finalDetections;
  }

  /// Print a summary of detections to console with class breakdown
  void _printDetectionSummary(List<Map<String, dynamic>> detections) {
    print('\n========== DETECTION SUMMARY ==========');
    
    if (detections.isEmpty) {
      print('NO DETECTIONS FOUND');
       return;
    }
    
    print('DETECTIONS FOUND: ${detections.length}');
    
    // Group by class
    final byClass = <int, List<Map<String, dynamic>>>{};
    for (final det in detections) {
      final classId = det['classId'] as int?;
      if (classId != null) {
        byClass.putIfAbsent(classId, () => []).add(det);
      }
    }
    
    // Print breakdown by class
    print('\nDETAILS BY CLASS:');
    for (final classId in byClass.keys) {
      final label = (classId >= 0 && classId < labels.length) ? labels[classId] : 'unknown_$classId';
      final dets = byClass[classId]!;
      final isVehicle = (classId == vehicleClassId);
      final type = isVehicle ? '' : '';
      
      print('  $type $label (ID: $classId): ${dets.length} detections');
      
      // Show top 3 by confidence
      final top3 = dets.take(3);
      for (int i = 0; i < top3.length; i++) {
        final det = top3.elementAt(i);
        final score = det['score'] as double;
        final confidence = (score * 100).toStringAsFixed(1);
        final box = 'box=[${(det['xmin'] as double).toStringAsFixed(2)}, ${(det['ymin'] as double).toStringAsFixed(2)}, ${(det['xmax'] as double).toStringAsFixed(2)}, ${(det['ymax'] as double).toStringAsFixed(2)}]';
        print('     #${i+1}: $confidence% confidence, $box');
      }
    }
    
  }

  /// Return debug info for each output channel: min/max/mean and top-k indices/values.
  Future<List<Map<String, dynamic>>> debugOutputs(String imagePath, {int topK = 5}) async {
    final bytes = await File(imagePath).readAsBytes();
    final original = img.decodeImage(bytes);
    if (original == null) return [];

    final inputShape = interpreter.getInputTensor(0).shape;
    final inputHeight = inputShape.length > 1 ? inputShape[1] : 0;
    final inputWidth = inputShape.length > 2 ? inputShape[2] : 0;
    if (inputHeight == 0 || inputWidth == 0) return [];

    final resized = img.copyResize(original, width: inputWidth, height: inputHeight);
    final input = List.generate(1, (_) {
      return List.generate(inputHeight, (y) {
        return List.generate(inputWidth, (x) {
          final px = resized.getPixel(x, y);
          final r = (px.r) / 255.0;
          final g = (px.g) / 255.0;
          final b = (px.b) / 255.0;
          return [r, g, b];
        });
      });
    });

    final outShape = interpreter.getOutputTensor(0).shape; // e.g. [1, 6, 8400]
    final outCount = outShape.reduce((a, b) => a * b);
    final output = List.filled(outCount, 0.0).reshape(outShape);

    try {
      interpreter.run(input, output);
    } catch (e) {
      return [];
    }

    if (outShape.length != 3) return [];

    final dim1 = outShape[1];
    final dim2 = outShape[2];
    bool looksLikeChannels(int d) => d > 0 && d <= 64;

    final channelsFirst = (looksLikeChannels(dim1) && !looksLikeChannels(dim2))
        ? true
        : (looksLikeChannels(dim2) && !looksLikeChannels(dim1))
            ? false
            : dim1 < dim2;

    final channelCount = channelsFirst ? dim1 : dim2;
    final channelLen = channelsFirst ? dim2 : dim1;

    double getVal(int c, int i) {
      if (channelsFirst) {
        final v = output[0][c][i];
        return (v is num) ? v.toDouble() : double.tryParse(v.toString()) ?? 0.0;
      } else {
        final v = output[0][i][c];
        return (v is num) ? v.toDouble() : double.tryParse(v.toString()) ?? 0.0;
      }
    }
    final List<Map<String, dynamic>> info = [];
    for (var c = 0; c < channelCount; c++) {
      // flatten channel values
      final List<double> vals = List.generate(channelLen, (i) => getVal(c, i));

      double minv = double.infinity;
      double maxv = -double.infinity;
      double sum = 0.0;
      for (var v in vals) {
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += v;
      }
      final mean = vals.isNotEmpty ? sum / vals.length : 0.0;

      // top-k indices
      final indexed = List<int>.generate(vals.length, (i) => i);
      indexed.sort((a, b) => vals[b].compareTo(vals[a]));
      final top = indexed.take(topK).map((i) => {'idx': i, 'value': vals[i]}).toList();

      info.add({
        'channel': c,
        'min': minv,
        'max': maxv,
        'mean': mean,
        'top': top,
        'length': vals.length,
      });
    }

    return info;
  }

  /// Crop a detection region from the original image and write it to a temporary file.
  /// `box` should contain normalized coordinates from detect(). Returns the temp file path.
  Future<String?> cropToFile(String imagePath, Map<String, dynamic> box) async {
    final bytes = await File(imagePath).readAsBytes();
    final original = img.decodeImage(bytes);
    if (original == null) return null;

    final w = original.width;
    final h = original.height;

    final xmin = ((box['xmin'] ?? 0) as num).toDouble();
    final ymin = ((box['ymin'] ?? 0) as num).toDouble();
    final xmax = ((box['xmax'] ?? 0) as num).toDouble();
    final ymax = ((box['ymax'] ?? 0) as num).toDouble();

    final ixmin = (xmin * w).clamp(0, w).toInt();
    final iymin = (ymin * h).clamp(0, h).toInt();
    final ixmax = (xmax * w).clamp(0, w).toInt();
    final iymax = (ymax * h).clamp(0, h).toInt();

    final cropW = max(1, ixmax - ixmin);
    final cropH = max(1, iymax - iymin);

    final cropped = img.copyCrop(original, x: ixmin, y: iymin, width: cropW, height: cropH);

    final jpg = img.encodeJpg(cropped);
    final tmp = await getTemporaryDirectory();
    final out = File('${tmp.path}/crop_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await out.writeAsBytes(jpg);
    return out.path;
  }

  /// Draw detection boxes and optional labels onto the original image and
  /// write to a temporary file. Returns the temp file path.
  Future<String?> drawDetectionsOnImage(String imagePath, List<Map<String, dynamic>> detections) async {
    // For now, don't modify the image; just return the original path.
    // This avoids incompatibilities with the `image` package draw APIs.
    return imagePath;
  }

  /// Generate multiple OCR-friendly variants.
  /// Returns list of file paths (original upscale/contrast, thresholded, inverted+thresholded)
  Future<List<String>> enhanceForOcrVariants(String cropPath) async {
    final outPaths = <String>[];

    try {
      final bytes = await File(cropPath).readAsBytes();
      final im = img.decodeImage(bytes);
      if (im == null) return outPaths;

      // 4x upscale (more aggressive for tiny digits)
      final up = img.copyResize(
        im,
        width: im.width * 4,
        height: im.height * 4,
        interpolation: img.Interpolation.cubic,
      );

      // grayscale
      final gray = img.grayscale(up);

      // high contrast baseline
      final contrasted = img.adjustColor(gray, contrast: 2.0);

      // ---- helper: hard threshold (binary) ----
      img.Image binarize(img.Image src, {int thresh = 140, bool invert = false}) {
        final out = img.Image.from(src);
        for (int y = 0; y < out.height; y++) {
          for (int x = 0; x < out.width; x++) {
            final p = out.getPixel(x, y);
            final lum = (p.r + p.g + p.b) ~/ 3;
            int v = lum > thresh ? 255 : 0;
            if (invert) v = 255 - v;
            out.setPixelRgb(x, y, v, v, v);
          }
        }
        return out;
      }

      final thresh1 = binarize(contrasted, thresh: 135, invert: false);
      final thresh2 = binarize(contrasted, thresh: 135, invert: true);

      // optional sharpen after threshold
      img.Image sharpen(img.Image src) {
        return img.convolution(src, filter: [
          0, -1, 0,
          -1, 5, -1,
          0, -1, 0
        ]);
      }

      final variantImages = [
        sharpen(contrasted),
        sharpen(thresh1),
        sharpen(thresh2),
      ];

      final tmp = await getTemporaryDirectory();
      for (final v in variantImages) {
        final outJpg = img.encodeJpg(v, quality: 98);
        final outFile = File(
          '${tmp.path}/ocr_var_${DateTime.now().millisecondsSinceEpoch}_${outPaths.length}.jpg',
        );
        await outFile.writeAsBytes(outJpg);
        outPaths.add(outFile.path);
      }
    } catch (e) {
      print('enhanceForOcrVariants error: $e');
    }

    return outPaths;
  }

  /// Enhance a crop for better OCR by upscaling, adjusting contrast, and sharpening
  Future<String?> enhanceForOcr(String cropPath) async {
    try {
      final bytes = await File(cropPath).readAsBytes();
      final im = img.decodeImage(bytes);
      if (im == null) return null;

      // upscale 3x
      final up = img.copyResize(
        im,
        width: im.width * 3,
        height: im.height * 3,
        interpolation: img.Interpolation.cubic,
      );

      // grayscale + contrast + sharpen
      final gray = img.grayscale(up);
      final contrasted = img.adjustColor(gray, contrast: 1.7);
      final sharpened = img.convolution(contrasted, filter: [
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
      ]);

      final outJpg = img.encodeJpg(sharpened, quality: 95);
      final tmp = await getTemporaryDirectory();
      final out = File('${tmp.path}/ocr_up_${DateTime.now().millisecondsSinceEpoch}.jpg');
      await out.writeAsBytes(outJpg);
      return out.path;
    } catch (e) {
      print('VehicleDetector.enhanceForOcr: failed to enhance $cropPath: $e');
      return null;
    }
  }
}
