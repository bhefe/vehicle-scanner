//vehicle_detector.dart

import 'dart:io';
import 'dart:math';

import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class VehicleDetector {
  late Interpreter interpreter;
  /// Optional class labels for the model. Adjust to match your model's classes.
  List<String> labels = ['bus', 'car', 'jeep', 'plate no-', 'truck'];
  
  /// Vehicle class IDs (excluding plate)
  static const Set<int> vehicleClassIds = {0, 1}; // bus, car, jeep, truck
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

    final input = List.generate(1, (_) {
      return List.generate(inputH, (y) {
        return List.generate(inputW, (x) {
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

          final rf = r / 255.0;
          final gf = g / 255.0;
          final bf = b / 255.0;
          return [rf, gf, bf];
        });
      });
    });

    // Output currently observed as [1, 9, 8400]
    final outShape = interpreter.getOutputTensor(0).shape;
    final outCount = outShape.reduce((a, b) => a * b);
    final output = List.filled(outCount, 0.0).reshape(outShape);
    interpreter.run(input, output);

    // Support both channel-first [1,C,N] and channel-last [1,N,C]
    final detections = <Map<String, dynamic>>[];
    final channelDim = outShape[1];
    final otherDim = outShape[2];
    final channelsFirst = (channelDim > otherDim);
    final channelCount = channelsFirst ? outShape[1] : outShape[2];
    final count = channelsFirst ? outShape[2] : outShape[1];

    // heuristic: bbox channels 0..3. check for objectness channel (channel 4)
    final hasObj = (channelCount - 5) >= 1;
    final numClasses = hasObj ? channelCount - 5 : max(0, channelCount - 4);

    double sigmoid(double x) => 1.0 / (1.0 + exp(-x));

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

      // apply sigmoid to centers (common in many YOLO-like exports)
      final cx = sigmoid(rawCx);
      final cy = sigmoid(rawCy);

      // w,h may be logits; assume they are normalized already, but if too large, apply sigmoid
      final wRaw = rawW;
      final hRaw = rawH;
      final w = (wRaw.abs() > 1.0) ? sigmoid(wRaw) : wRaw.abs();
      final h = (hRaw.abs() > 1.0) ? sigmoid(hRaw) : hRaw.abs();

      double obj = 1.0;
      if (hasObj) {
        obj = sigmoid(getVal(4, i));
      }

      int bestClass = -1;
      double bestScore = -double.infinity;
      for (int c = 0; c < numClasses; c++) {
        final rawCls = getVal(4 + (hasObj ? 1 : 0) + c, i);
        final clsProb = sigmoid(rawCls);
        final score = obj * clsProb;
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < confThreshold) continue;

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
      final isVehicle = vehicleClassIds.contains(classId);
      final type = isVehicle ? '🚗' : '📍';
      
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
    
    // Summary line
    final vehicleCount = detections.where((d) => vehicleClassIds.contains(d['classId'] as int?)).length;
    final plateCount = detections.where((d) => (d['classId'] as int?) == plateClassId).length;
    
    print('\nSUMMARY: $vehicleCount vehicle(s) + $plateCount plate(s) detected');
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

    final outShape = interpreter.getOutputTensor(0).shape; // e.g. [1, 9, 8400]
    final outCount = outShape.reduce((a, b) => a * b);
    final output = List.filled(outCount, 0).reshape(outShape);

    try {
      interpreter.run(input, output);
    } catch (e) {
      return [];
    }

    final channelCount = outShape[1];
    final channelLen = outShape[2];
    final List<Map<String, dynamic>> info = [];
    for (var c = 0; c < channelCount; c++) {
      // flatten channel values
      final List<double> vals = List.generate(channelLen, (i) {
        final v = output[0][c][i];
        return (v is num) ? v.toDouble() : double.tryParse(v.toString()) ?? 0.0;
      });

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
