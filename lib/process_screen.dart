import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'vehicle_detector.dart';
import 'ocr.dart';

class ProcessScreen extends StatefulWidget {
  final String imagePath;
  const ProcessScreen({required this.imagePath, super.key});

  @override
  State<ProcessScreen> createState() => _ProcessScreenState();
}

class _ProcessScreenState extends State<ProcessScreen> {
  static const bool collectModelDebugInfo = false; // set true only when diagnosing

  String result = "";
  bool isProcessing = true;
  final detector = VehicleDetector();
  final ocr = PlateOCR();
  String? cropPath;
  Map<String, dynamic>? topDetection;
  String modelStatus = 'Model not loaded';
  int detectionCount = 0;
  int vehicleCount = 0;
  int plateCount = 0;
  String detectionStatus = ''; // Status message about what was detected
  List<Map<String, dynamic>>? debugInfo;
  String? annotatedPath;

  @override
  void initState() {
    super.initState();
    // Show loading spinner, start processing
    WidgetsBinding.instance.addPostFrameCallback((_) {
      runPipeline();
    });
  }

  Future<void> runPipeline() async {
    final totalSw = Stopwatch()..start();
    // Allow UI to render the loading spinner first
    await Future.delayed(const Duration(milliseconds: 100));
    
    try {
      final loadSw = Stopwatch()..start();
      await detector.loadModel();
      setState(() {
        modelStatus = 'Model loaded';
      });
      loadSw.stop();
      if (kDebugMode) {
        print('Timing: model load ${loadSw.elapsedMilliseconds}ms');
      }
    } catch (e, st) {
      setState(() {
        modelStatus = 'Model load failed: $e';
        result = 'Model load failed';
        isProcessing = false;
      });
      print('ProcessScreen.runPipeline: model load failed: $e\n$st');
      return;
    }
    // Run detector on the full image and get detections (normalized coords)
    final detectSw = Stopwatch()..start();
    final detections = await detector.detect(widget.imagePath, confThreshold: 0.35);
    detectSw.stop();
    if (kDebugMode) {
      print('Timing: detect ${detectSw.elapsedMilliseconds}ms (detections=${detections.length})');
    }
    
    // collect debug info for the outputs so user can inspect channels
    if (kDebugMode && collectModelDebugInfo) {
      final dbgSw = Stopwatch()..start();
      debugInfo = await detector.debugOutputs(widget.imagePath, topK: 3);
      dbgSw.stop();
      print('Timing: debugOutputs ${dbgSw.elapsedMilliseconds}ms');
    }
    detectionCount = detections.length;
    print('ProcessScreen.runPipeline: Total detections found: $detectionCount');

    // Always produce an annotated image so user can see where the model fired
    // even if detections are below the confidence threshold
    try {
      final annSw = Stopwatch()..start();
      annotatedPath = await detector.drawDetectionsOnImage(widget.imagePath, detections);
      annSw.stop();
      if (kDebugMode) {
        print('Timing: annotate ${annSw.elapsedMilliseconds}ms');
      }
    } catch (e) {
      annotatedPath = null;
    }

    // Keep only plate detections (classId == 1)
    final plates = detections.where((d) => (d['classId'] ?? -1) == VehicleDetector.plateClassId).toList();
    
    // Get vehicle detections (classId == 0)
    final vehicles = detections.where((d) {
      final classId = (d['classId'] ?? -1);
      return classId == VehicleDetector.vehicleClassId;
    }).toList();
    
    // Update vehicle and plate counts
    vehicleCount = vehicles.length;
    plateCount = plates.length;
    
    // Set detection status message
    if (detectionCount == 0) {
      detectionStatus = 'NO VEHICLE DETECTED';
      print('Model did not detect any vehicles in the image');
    } else if (vehicleCount > 0 && plateCount > 0) {
      detectionStatus = 'VEHICLE DETECTED ';
    }

    // Fast path: only OCR the top plate detections, and keep OCR attempts minimal.
    // This avoids ANRs caused by doing many crops + many OCR passes.
    String? foundPlate;
    String? foundCropPath;
    Map<String, dynamic>? foundDetection;

    if (plates.isNotEmpty) {
      plates.sort((a, b) => ((b['score'] ?? 0) as num).compareTo((a['score'] ?? 0) as num));
      final candidates = plates.take(2).toList();

      // Allow a frame to render before doing OCR work.
      await Future<void>.delayed(Duration.zero);

      for (final cand in candidates) {
        // Try a slightly padded crop first.
        final padded = Map<String, dynamic>.from(cand);
        const pad = 0.25;
        final xmin = ((cand['xmin'] as num?)?.toDouble() ?? 0.0);
        final ymin = ((cand['ymin'] as num?)?.toDouble() ?? 0.0);
        final xmax = ((cand['xmax'] as num?)?.toDouble() ?? 1.0);
        final ymax = ((cand['ymax'] as num?)?.toDouble() ?? 1.0);
        final w = (xmax - xmin).clamp(0.0, 1.0);
        final h = (ymax - ymin).clamp(0.0, 1.0);
        padded['xmin'] = (xmin - w * pad).clamp(0.0, 1.0);
        padded['ymin'] = (ymin - h * pad).clamp(0.0, 1.0);
        padded['xmax'] = (xmax + w * pad).clamp(0.0, 1.0);
        padded['ymax'] = (ymax + h * pad).clamp(0.0, 1.0);

        for (final box in [padded, cand]) {
          final crop = await detector.cropToFile(widget.imagePath, box);
          if (crop == null) continue;

          try {
            final ocrSw = Stopwatch()..start();
            final text1 = await ocr.extractText(crop);
            final plate1 = _extractPlate(text1);
            if (kDebugMode) {
              ocrSw.stop();
              print('Timing: OCR plate-crop ${ocrSw.elapsedMilliseconds}ms');
            }

            if (plate1 != null) {
              foundPlate = plate1;
              foundCropPath = crop;
              foundDetection = box;
              break;
            }

            // If first pass fails, try one enhanced variant (not multiple variants).
            final enhanced = await detector.enhanceForOcr(crop);
            if (enhanced != null) {
              final text2 = await ocr.extractText(enhanced);
              final plate2 = _extractPlate(text2);
              if (plate2 != null) {
                foundPlate = plate2;
                foundCropPath = crop;
                foundDetection = box;
                break;
              }
            }
          } catch (e) {
            // Ignore small crops (ML Kit requires >= 32x32) and move on.
            if (kDebugMode) {
              print('ProcessScreen: OCR failed for plate crop: $e');
            }
          }
        }

        if (foundPlate != null) break;
      }
    }

    if (foundPlate != null) {
      cropPath = foundCropPath;
      if (kDebugMode) {
        final score = ((foundDetection?['score'] ?? 0) as num).toDouble();
        final xmin = ((foundDetection?['xmin'] ?? 0) as num).toDouble();
        final ymin = ((foundDetection?['ymin'] ?? 0) as num).toDouble();
        final xmax = ((foundDetection?['xmax'] ?? 0) as num).toDouble();
        final ymax = ((foundDetection?['ymax'] ?? 0) as num).toDouble();
        debugPrint('Selected plate crop: $cropPath');
        debugPrint('Detection score: ${score.toStringAsFixed(2)}');
        debugPrint(
          'Box (normalized): xmin=${xmin.toStringAsFixed(3)}, ymin=${ymin.toStringAsFixed(3)}, xmax=${xmax.toStringAsFixed(3)}, ymax=${ymax.toStringAsFixed(3)}',
        );
        debugPrint('Class: ${foundDetection?['label']} (id: ${foundDetection?['classId']})');
      }
      setState(() {
        topDetection = foundDetection;
        result = foundPlate!;
        modelStatus = 'Model loaded (plate detected)';
        isProcessing = false;
      });
      totalSw.stop();
      if (kDebugMode) {
        print('Timing: pipeline total ${totalSw.elapsedMilliseconds}ms');
      }
      return;
    }
      
    // Final fallback: no plate or vehicle detections, so run ocr on  whole image
    final text = await ocr.extractText(widget.imagePath);
    final plate = _extractPlate(text);
    setState(() {
      result = plate ?? 'No plate found';
      modelStatus = 'Model loaded (no detections worked, used full image)';
      isProcessing = false;
    });

    totalSw.stop();
    if (kDebugMode) {
      print('Timing: pipeline total ${totalSw.elapsedMilliseconds}ms');
    }
  }

  String? _extractPlate(String? ocrText) {
    if (ocrText == null || ocrText.trim().isEmpty) return null;
    final text = ocrText.toUpperCase();
    print('ProcessScreen: _extractPlate input: "$text"'); // Debug: input to extraction

    // Clean helper
    String clean(String s) => s.replaceAll(RegExp(r'[^A-Z0-9\-]'), '');

    // Enhanced plate patterns 
    final patterns = <RegExp>[
      RegExp(r'\bBMP\s?[.-]?\d{3,4}\b'),                    
      RegExp(r'\bBLW\s?[.-]?\d{3,4}\b'),                    
      RegExp(r'\bWLW\s?[.-]?\d{3,4}\b'),                    
      RegExp(r'\bEV[A-Z]?\s?[.-]?\d{1,4}\b'),              // EV plates  
      RegExp(r'\bZ[A-Z]?\s?[.-]?\d{1,4}\b'),               // Military
      RegExp(r'\bS[A-Z]{0,2}\s?[.-]?\d{1,4}\b'),           // Sabah
      RegExp(r'\bQ[A-Z]{0,2}\s?[.-]?\d{1,4}\b'),           // Sarawak
      RegExp(r'\b\d{1,3}[-\s]\d{1,3}[-\s][A-Z]{2,3}\b'),  // Diplomatic
      RegExp(r'\b[A-Z]{2,3}\s?[.-]?\d{3,4}[A-Z]?\b'),     // Standard 3-letter prefix plates
      RegExp(r'\b[A-Z]{1,3}\s?[.-]?\d{1,4}[A-Z]?\b'),     // General MY plates
    ];

    final candidates = <String>[];

    for (int i = 0; i < patterns.length; i++) {
      final re = patterns[i];
      for (final m in re.allMatches(text)) {
        final raw = m.group(0) ?? '';
        var c = clean(raw).replaceAll(' ', '').replaceAll('-', '').replaceAll('.', '');
        if (c.isNotEmpty) {
          candidates.add(c);
          print('ProcessScreen: Pattern $i matched: "$raw" -> cleaned: "$c"');
        }
      }
    }

    print('ProcessScreen: All pattern candidates: $candidates'); // Debug: found candidates

    if (candidates.isEmpty) {
      print('ProcessScreen: No candidates found with patterns, trying fallback...');
      // Enhanced fallback: look for any letter+digit combinations with flexible separators
      final fallbackPattern = RegExp(r'[A-Z]{2,4}[.\s-]?\d{2,4}');
      for (final m in fallbackPattern.allMatches(text)) {
        final raw = m.group(0) ?? '';
        var c = clean(raw).replaceAll(' ', '').replaceAll('.', '').replaceAll('-', '');
        if (c.isNotEmpty && c.length >= 5) {
          candidates.add(c);
          print('ProcessScreen: Fallback matched: "$raw" -> "$c"');
        }
      }
    }

    if (candidates.isEmpty) return null;

    double score(String c) {
      final hasLetters = RegExp(r'[A-Z]').hasMatch(c);
      final hasDigits  = RegExp(r'\d').hasMatch(c);
      if (!hasLetters || !hasDigits) return 0;

      final len = c.length;
      final lenScore = (len >= 5 && len <= 7) ? 1.0 : 0.6; // Standard plate length
      final endsWithDigits = RegExp(r'\d+$').hasMatch(c) ? 1.0 : 0.7;
      
      // Enhanced prefix scoring for various Malaysian plates
      double prefixBonus = 1.0;
      if (c.startsWith('EV')) prefixBonus = 1.2;   // Electric vehicle
      else if (RegExp(r'^[A-Z]{2,3}\d{3,4}$').hasMatch(c)) prefixBonus = 1.15; // Standard format
      
      // Bonus for 4-digit numbers (common in Malaysian plates)
      final digitCount = c.replaceAll(RegExp(r'[^\d]'), '').length;
      final digitBonus = digitCount == 4 ? 1.1 : 1.0;

      return lenScore * endsWithDigits * prefixBonus * digitBonus;
    }

    candidates.sort((a, b) => score(b).compareTo(score(a)));
    print('ProcessScreen: Candidates after scoring: ${candidates.map((c) => '$c (${score(c).toStringAsFixed(2)})').toList()}');
    print('ProcessScreen: Top candidate selected: "${candidates.first}" with score ${score(candidates.first).toStringAsFixed(2)}');
    final best = candidates.first;

    // letters + space + digits (+ optional suffix)
    final match = RegExp(r'^([A-Z]+)(\d+)([A-Z]?)$').firstMatch(best);
    if (match != null) {
      final prefix = match.group(1)!;
      final digits = match.group(2)!;
      final suffix = match.group(3) ?? '';
      return '$prefix $digits$suffix';
    }
    return best;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Result")),
      body: isProcessing
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Processing image...'),
                ],
              ),
            )
          : SizedBox(
              width: double.infinity,
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 10), 
            child: Text(
              detectionStatus,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: detectionStatus.contains('NO') ? Colors.red[900] : Colors.green[900],
              ),
            ),
          ),

          Builder(builder: (context) {
            final maxH = MediaQuery.of(context).size.height * 0.55;
            return Container(
              width: double.infinity,
              height: maxH,
              child: Image.file(File(widget.imagePath), fit: BoxFit.contain),
            );
          }),

          const SizedBox(height: 18),
          const Text('Plate Number:'),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text(result, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          ),
                  ],
                ),
              ),
            ),
    );
  }
}
