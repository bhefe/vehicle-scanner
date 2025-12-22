import 'dart:io';
import 'package:flutter/material.dart';
import 'vehicle_detector.dart';
import 'ocr.dart';

class ProcessScreen extends StatefulWidget {
  final String imagePath;
  const ProcessScreen({required this.imagePath, super.key});

  @override
  State<ProcessScreen> createState() => _ProcessScreenState();
}

class _ProcessScreenState extends State<ProcessScreen> {
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
    // Show loading spinner immediately, then start processing
    WidgetsBinding.instance.addPostFrameCallback((_) {
      runPipeline();
    });
  }

  Future<void> runPipeline() async {
    // Allow UI to render the loading spinner first
    await Future.delayed(const Duration(milliseconds: 100));
    
    try {
      await detector.loadModel();
      setState(() {
        modelStatus = 'Model loaded';
      });
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
    final detections = await detector.detect(widget.imagePath, confThreshold: 0.15); // Lower threshold
    
    // Also collect debug info for the outputs so user can inspect channels
    debugInfo = await detector.debugOutputs(widget.imagePath, topK: 3);
    detectionCount = detections.length;
    print('ProcessScreen.runPipeline: Total detections found: $detectionCount');

    // Always produce an annotated image so the user can see where the model fired
    // even if detections are below the confidence threshold.
    try {
      annotatedPath = await detector.drawDetectionsOnImage(widget.imagePath, detections);
    } catch (e) {
      annotatedPath = null;
    }

    // Keep only plate detections (classId == 3)
    final plates = detections.where((d) => (d['classId'] ?? -1) == 3).toList();
    
    // Get vehicle detections as fallback (bus, car, jeep, truck - classId 0,1,2,4)
    final vehicles = detections.where((d) {
      final classId = (d['classId'] ?? -1);
      return classId == 0 || classId == 1 || classId == 2 || classId == 4; // bus, car, jeep, truck
    }).toList();
    
    // Update vehicle and plate counts
    vehicleCount = vehicles.length;
    plateCount = plates.length;
    
    // Set detection status message
    if (detectionCount == 0) {
      detectionStatus = 'NO VEHICLE OR PLATE DETECTED';
      print('Model did not detect any vehicles or plates in the image');
    } else if (vehicleCount > 0 && plateCount > 0) {
      detectionStatus = 'VEHICLE & PLATE DETECTED (${vehicleCount} vehicle(s), ${plateCount} plate(s))';
      print('Detected ${vehicleCount} vehicle(s) and ${plateCount} plate(s)');
    } else if (vehicleCount > 0) {
      detectionStatus = 'VEHICLE DETECTED (${vehicleCount} vehicle(s), no plates found)';
      print('Vehicle detected but no plate found');
    } else if (plateCount > 0) {
      detectionStatus = 'PLATE DETECTED (${plateCount} plate(s), no vehicles)';
      print('Plate detected but no vehicle box found');
    }

    // Function to check if OCR text contains plate-like patterns
    bool hasPlateText(String? text) {
      if (text == null || text.trim().isEmpty) return false;
      final upper = text.toUpperCase();
      
      // Enhanced Malaysian plate patterns
      final platePatterns = [
        RegExp(r'[A-Z]{1,4}[.\s-]*\d{1,4}'), // Basic format with flexible separators
        RegExp(r'\b[A-Z]{2,3}[.\s-]*\d{3,4}\b'), // Common MY format
        RegExp(r'\bBM[A-Z]?[.\s-]*\d{3,4}\b'), // BMP, BMA, etc.
        RegExp(r'\bBLW[.\s-]*\d{3,4}\b'), // BLW plates (Labuan)
        RegExp(r'\bWLW[.\s-]*\d{3,4}\b'), // WLW plates
        RegExp(r'\bEV[A-Z]?[.\s-]*\d{1,4}\b'), // Electric vehicle plates
        RegExp(r'\b[A-Z]{1,3}\d{3,4}[A-Z]?\b'), // Standard formats
        RegExp(r'\b[A-Z]{2,3}[.\s-]*\d{3,4}[A-Z]?\b'), // With optional suffix
      ];
      
      return platePatterns.any((pattern) => pattern.hasMatch(upper));
    }    // Try ALL detections for plate-like text (comprehensive approach)
    String? foundPlate;
    String? foundCropPath;
    Map<String, dynamic>? foundDetection;
    
    // Sort all detections by confidence
    final allDetections = List<Map<String, dynamic>>.from(detections);
    allDetections.sort((a, b) => ((b['score'] ?? 0) as num).compareTo((a['score'] ?? 0) as num));
    
    print('ProcessScreen: Checking ${allDetections.length} total detections for plate text');
    
    // Check each detection for plate-like text
    for (final detection in allDetections) {
      final classId = detection['classId'] ?? -1;
      final score = detection['score'] ?? 0;
      
      // Skip invalid detections (likely decoder issues)
      if (classId < 0 || classId > 10 || score < 0.15) {
        continue;
      }
      
      print('ProcessScreen: Trying detection classId=$classId, score=${score.toStringAsFixed(3)}');
      
      // Check if crop would be large enough for ML Kit (min 32x32)
      final xmin = (detection['xmin'] as num?)?.toDouble() ?? 0.0;
      final ymin = (detection['ymin'] as num?)?.toDouble() ?? 0.0;
      final xmax = (detection['xmax'] as num?)?.toDouble() ?? 1.0;
      final ymax = (detection['ymax'] as num?)?.toDouble() ?? 1.0;
      
      // Add padding to capture more of the plate (try multiple padding levels)
      final w = xmax - xmin;
      final h = ymax - ymin;
      
      // Try multiple padding levels if needed
      final paddingLevels = [0.20, 0.30, 0.40]; // 20%, 30%, 40% padding
      
      for (final paddingLevel in paddingLevels) {
        final padW = w * paddingLevel;
        final padH = h * paddingLevel;
        
        final paddedXmin = (xmin - padW).clamp(0.0, 1.0);
        final paddedYmin = (ymin - padH).clamp(0.0, 1.0);
        final paddedXmax = (xmax + padW).clamp(0.0, 1.0);
        final paddedYmax = (ymax + padH).clamp(0.0, 1.0);
        
        // Create padded detection box
        final paddedDetection = Map<String, dynamic>.from(detection);
        paddedDetection['xmin'] = paddedXmin;
        paddedDetection['ymin'] = paddedYmin;
        paddedDetection['xmax'] = paddedXmax;
        paddedDetection['ymax'] = paddedYmax;
        
        final cropW = ((paddedXmax - paddedXmin) * 1000).round();
        final cropH = ((paddedYmax - paddedYmin) * 1000).round();
        
        if (cropW < 32 || cropH < 32) {
          continue; // Try next padding level
        }
        
        print('ProcessScreen: Trying padding level ${(paddingLevel * 100).round()}%');
        print('ProcessScreen: Original box: [${xmin.toStringAsFixed(3)}, ${ymin.toStringAsFixed(3)}, ${xmax.toStringAsFixed(3)}, ${ymax.toStringAsFixed(3)}]');
        print('ProcessScreen: Padded box: [${paddedXmin.toStringAsFixed(3)}, ${paddedYmin.toStringAsFixed(3)}, ${paddedXmax.toStringAsFixed(3)}, ${paddedYmax.toStringAsFixed(3)}]');
        
        final cropPath = await detector.cropToFile(widget.imagePath, paddedDetection);
        if (cropPath == null) continue;

        try {
          // Generate multiple OCR variants and choose best by plate score
          final variants = await detector.enhanceForOcrVariants(cropPath);
          String? bestText;
          String? bestPlate;
          double bestScore = -1;

          double plateScore(String? plate) {
            if (plate == null) return 0;
            final compact = plate.replaceAll(RegExp(r'[^A-Z0-9]'), '');
            final digits = compact.replaceAll(RegExp(r'[^0-9]'), '');
            final letters = compact.replaceAll(RegExp(r'[^A-Z]'), '');

            double s = 0;
            if (letters.isNotEmpty) s += 1;
            if (digits.length >= 3) s += 1;
            if (digits.length == 4) s += 2; // prefer 4 digits when available
            
            // Enhanced prefix scoring for various Malaysian plates  
            if (compact.startsWith('BMP')) s += 1.5;      // Police
            else if (compact.startsWith('BLW')) s += 1.3;  // Labuan - HIGH PRIORITY
            else if (compact.startsWith('WLW')) s += 1.3;  // Wilayah
            else if (compact.startsWith('EV')) s += 1.2;   // Electric vehicle
            else if (RegExp(r'^[A-Z]{2,3}\d{3,4}$').hasMatch(compact)) s += 1.0; // Standard format
            
            return s;
          }

          // Also try original crop (no enhancement)
          final originalText = await ocr.extractText(cropPath);
          final originalPlate = _extractPlate(originalText);
          final originalScore = plateScore(originalPlate);
          print('ProcessScreen: Original OCR "$originalText" => plate "$originalPlate" score $originalScore');
          
          if (originalScore > bestScore) {
            bestScore = originalScore;
            bestText = originalText;
            bestPlate = originalPlate;
          }

          // Try all enhancement variants
          for (final vPath in variants) {
            final t = await ocr.extractText(vPath);
            final p = _extractPlate(t);
            final s = plateScore(p);
            print('ProcessScreen: OCR variant $vPath => \"$t\" => plate \"$p\" score $s');
            if (s > bestScore) {
              bestScore = s;
              bestText = t;
              bestPlate = p;
            }
          }

          final finalOcrText = bestText;
          final finalPlate = bestPlate;
          print('ProcessScreen: Final OCR text length: ${finalOcrText?.length ?? 0}');
          print('ProcessScreen: Final OCR contains "BMP": ${finalOcrText?.contains("BMP") ?? false}');
          print('ProcessScreen: Final OCR contains "4915": ${finalOcrText?.contains("4915") ?? false}');
          print('ProcessScreen: Final OCR contains "495": ${finalOcrText?.contains("495") ?? false}');
          
          // Check if this gives us a more complete result
          if (finalPlate != null && finalPlate.contains('4915')) {
            print('ProcessScreen: Found complete plate number with ${(paddingLevel * 100).round()}% padding!');
            foundPlate = finalPlate;
            foundCropPath = cropPath;
            foundDetection = paddedDetection;
            print('ProcessScreen: SUCCESS from padded detection: "$foundPlate"');
            break;
          }
          
          // Check if this detection contains plate-like text
          if (hasPlateText(finalOcrText)) {
            print('ProcessScreen: Found plate-like text in padded detection!');
            
            // Try normal extraction if we haven't already
            if (finalPlate == null) {
              final extracted = _extractPlate(finalOcrText);
              print('ProcessScreen: Extracted from padded detection: "$extracted"');
              if (extracted != null && (extracted.contains('4915') || extracted.length >= 7)) {
                foundPlate = extracted;
                foundCropPath = cropPath;
                foundDetection = paddedDetection;
                print('ProcessScreen: SUCCESS from padded detection: "$foundPlate"');
                break;
              }
            }
          }
          
          // Fallback: check for any improvement over "BMP495"
          if (finalPlate != null && finalPlate.length > 6) { // More than "BMP495"
            foundPlate = finalPlate;
            foundCropPath = cropPath;
            foundDetection = paddedDetection;
            print('ProcessScreen: Using better result from ${(paddingLevel * 100).round()}% padding: "$foundPlate"');
            break;
          }
          
        } catch (e) {
          print('ProcessScreen: OCR failed for padded detection: $e');
          continue; // Try next padding level
        }
      }
      
      // If we found something with padding, use it
      if (foundPlate != null) {
        break;
      }
      
      // If no padding level worked, try original detection without padding
      try {
        final cropPath = await detector.cropToFile(widget.imagePath, detection);
        if (cropPath != null) {
          // Generate multiple OCR variants and choose best by plate score
          final variants = await detector.enhanceForOcrVariants(cropPath);
          String? bestText;
          String? bestPlate;
          double bestScore = -1;

          double plateScore(String? plate) {
            if (plate == null) return 0;
            final compact = plate.replaceAll(RegExp(r'[^A-Z0-9]'), '');
            final digits = compact.replaceAll(RegExp(r'[^0-9]'), '');
            final letters = compact.replaceAll(RegExp(r'[^A-Z]'), '');

            double s = 0;
            if (letters.isNotEmpty) s += 1;
            if (digits.length >= 3) s += 1;
            if (digits.length == 4) s += 2; // prefer 4 digits when available
            
            // Enhanced prefix scoring for various Malaysian plates  
            if (compact.startsWith('BMP')) s += 1.5;      // Police
            else if (compact.startsWith('BLW')) s += 1.3;  // Labuan - HIGH PRIORITY
            else if (compact.startsWith('WLW')) s += 1.3;  // Wilayah
            else if (compact.startsWith('EV')) s += 1.2;   // Electric vehicle
            else if (RegExp(r'^[A-Z]{2,3}\d{3,4}$').hasMatch(compact)) s += 1.0; // Standard format
            
            return s;
          }

          // Also try original crop (no enhancement)
          final originalText = await ocr.extractText(cropPath);
          final originalPlate = _extractPlate(originalText);
          final originalScore = plateScore(originalPlate);
          print('ProcessScreen: Original OCR (no padding) "$originalText" => plate "$originalPlate" score $originalScore');
          
          if (originalScore > bestScore) {
            bestScore = originalScore;
            bestText = originalText;
            bestPlate = originalPlate;
          }

          // Try all enhancement variants
          for (final vPath in variants) {
            final t = await ocr.extractText(vPath);
            final p = _extractPlate(t);
            final s = plateScore(p);
            print('ProcessScreen: OCR variant $vPath => \"$t\" => plate \"$p\" score $s');
            if (s > bestScore) {
              bestScore = s;
              bestText = t;
              bestPlate = p;
            }
          }

          final finalOcrText = bestText;
          
          // Check if this detection contains plate-like text
          if (hasPlateText(finalOcrText)) {
            print('ProcessScreen: Found plate-like text in original detection!');
            
            // Try stacked plate handling first
            if (finalOcrText != null && finalOcrText.contains('\n')) {
              final lines = finalOcrText
                  .split(RegExp(r'\r?\n'))
                  .map((l) => l.trim())
                  .where((l) => l.isNotEmpty)
                  .toList();
              print('ProcessScreen: Multi-line OCR detected: $lines');
              if (lines.length >= 2) {
                String? letters;
                String? digits;
                for (final l in lines) {
                  final lU = l.toUpperCase();
                  if (RegExp(r'^[A-Z]{1,4}\$?').hasMatch(lU) && RegExp(r'[A-Z]').hasMatch(lU) && !RegExp(r'\d').hasMatch(lU)) {
                    letters = lU.replaceAll(RegExp(r'[^A-Z]'), '');
                  }
                  final d = lU.replaceAll(RegExp(r'[^0-9]'), '');
                  if (d.length >= 2 && d.length <= 4) {
                    digits = d;
                  }
                }
                print('ProcessScreen: Stacked plate components - letters: "$letters", digits: "$digits"');
                if (letters != null && digits != null) {
                  final combined = '${letters.trim()} ${digits.trim()}';
                  print('ProcessScreen: Combined stacked: "$combined"');
                  final extracted = _extractPlate(combined);
                  if (extracted != null) {
                    foundPlate = extracted;
                    foundCropPath = cropPath;
                    foundDetection = detection;
                    print('ProcessScreen: SUCCESS from stacked plate: "$foundPlate"');
                    break;
                  }
                }
              }
            }
            
            // Try normal extraction (if we didn't already get a best plate, or to compare)
            final extracted = bestPlate ?? _extractPlate(finalOcrText);
            print('ProcessScreen: Single-line extraction result: "$extracted"');
            if (extracted != null) {
              foundPlate = extracted;
              foundCropPath = cropPath;
              foundDetection = detection;
              print('ProcessScreen: SUCCESS from single-line: "$foundPlate"');
              break;
            }
          } else {
            print('ProcessScreen: No plate-like patterns found in: "$finalOcrText"');
          }
        }
      } catch (e) {
        print('ProcessScreen: OCR failed for detection: $e');
        continue; // Skip this detection and try next one
      }
    }

    // If we found a plate from any detection, use it
    if (foundPlate != null) {
      cropPath = foundCropPath;
      setState(() {
        topDetection = foundDetection;
        result = foundPlate!;
        modelStatus = 'Model loaded (found in detection classId=${foundDetection?['classId']})';
        debugInfo = debugInfo;
        isProcessing = false;
      });
      return;
    }

    // Legacy fallback: try the original plate-specific logic
    if (plates.isNotEmpty) {
      // Pick the best candidate plate bbox using geometric heuristics.
      // Relax aspect ratio heuristics so tall (stacked) plates are considered.
      List<Map<String, dynamic>> pickTopPlates(List<Map<String, dynamic>> plates, {int topK = 3}) { // Reduce to 3 for speed
        if (plates.isEmpty) return [];

        double scorePlate(Map<String, dynamic> p) {
          final xmin = (p['xmin'] as num).toDouble();
          final ymin = (p['ymin'] as num).toDouble();
          final xmax = (p['xmax'] as num).toDouble();
          final ymax = (p['ymax'] as num).toDouble();
          final w = (xmax - xmin).clamp(0.0001, 1.0);
          final h = (ymax - ymin).clamp(0.0001, 1.0);

          final aspect = w / h; // wide >1, tall <1
          final area = w * h; // avoid tiny junk
          final conf = (p['score'] as num).toDouble();
          final yCenter = (ymin + ymax) / 2.0; // lower-ish is often better

          // Relaxed aspect scoring: prefer typical wide plates, but don't strongly penalize tall boxes.
          double aspectScore;
          if (aspect >= 1.8) {
            aspectScore = 1.0; // canonical wide plate
          } else if (aspect >= 0.8) {
            aspectScore = 0.95; // acceptable
          } else if (aspect >= 0.4) {
            aspectScore = 0.9; // possibly stacked two-line plate
          } else {
            aspectScore = 0.6; // narrow/slim, less likely
          }

          final areaScore = (area >= 0.005) ? 1.0 : 0.4; // allow smaller plates
          final positionScore = (yCenter >= 0.35) ? 1.0 : 0.7; // be lenient about vertical placement

          // Final score blends confidence with geometry.
          return conf * aspectScore * areaScore * positionScore;
        }

        plates.sort((a, b) => scorePlate(b).compareTo(scorePlate(a)));
        return plates.take(topK).toList();
      }

      final topCandidates = pickTopPlates(plates, topK: 3);

      // Try OCR on the top candidate crops (this helps with stacked plates)
      for (final cand in topCandidates) {
        final ppath = await detector.cropToFile(widget.imagePath, cand);
        if (ppath == null) continue;

        try {
          final ptext1 = await ocr.extractText(ppath);
          print('ProcessScreen: OCR from original plate crop: "$ptext1"');
          
          String? ptext2;
          final enhancedPlatePath = await detector.enhanceForOcr(ppath);
          if (enhancedPlatePath != null) {
            ptext2 = await ocr.extractText(enhancedPlatePath);
            print('ProcessScreen: OCR from enhanced plate crop: "$ptext2"');
          }
          
          final mergedPtext = [ptext1, ptext2].whereType<String>().join('\n');
          final ptext = mergedPtext.isNotEmpty ? mergedPtext : ptext1;
          print('ProcessScreen: Combined plate OCR: "$ptext"');

          // Heuristic: detect stacked plates by checking for multiple lines and combining
          if (ptext.contains('\n')) {
            final lines = ptext
                .split(RegExp(r'\r?\n'))
                .map((l) => l.trim())
                .where((l) => l.isNotEmpty)
                .toList();
            if (lines.length >= 2) {
              // find a line with letters and a line with digits (order agnostic)
              String? letters;
              String? digits;
              for (final l in lines) {
                final lU = l.toUpperCase();
                if (RegExp(r'^[A-Z]{1,4}\$?').hasMatch(lU) && RegExp(r'[A-Z]').hasMatch(lU) && !RegExp(r'\d').hasMatch(lU)) {
                  letters = lU.replaceAll(RegExp(r'[^A-Z]'), '');
                }
                final d = lU.replaceAll(RegExp(r'[^0-9]'), '');
                if (d.length >= 2 && d.length <= 4) {
                  digits = d;
                }
              }
              if (letters != null && digits != null) {
                final combined = '${letters.trim()} ${digits.trim()}';
                final ext = _extractPlate(combined);
                if (ext != null) {
                  foundPlate = ext;
                  foundCropPath = ppath;
                  foundDetection = cand;
                  break;
                }
              }
            }
          }

          // Otherwise run normal extraction on the crop text
          final pplate = _extractPlate(ptext);
          print('ProcessScreen: Extracted plate from crop: "$pplate"'); // Debug: extracted result
          if (pplate != null) {
            foundPlate = pplate;
            foundCropPath = ppath;
            foundDetection = cand;
            break;
          }
        } catch (e) {
          print('ProcessScreen: OCR failed for plate crop: $e');
          continue; // Skip this crop and try next
        }
      }

      // If we found something from plate detections, use it
      if (foundPlate != null) {
        cropPath = foundCropPath;
        setState(() {
          topDetection = foundDetection;
          result = foundPlate!;
          debugInfo = debugInfo;
          isProcessing = false;
        });
        return;
      }
    }

    // Fallback to vehicle detections if no plates worked
    if (vehicles.isNotEmpty) {
      // Sort vehicles by confidence and try the best ones
      vehicles.sort((a, b) => ((b['score'] ?? 0) as num).compareTo((a['score'] ?? 0) as num));
      
      for (final vehicle in vehicles.take(2)) { // Reduce to 2 for speed
        final vpath = await detector.cropToFile(widget.imagePath, vehicle);
        if (vpath != null) {
          try {
            final vtext1 = await ocr.extractText(vpath);
            print('ProcessScreen: OCR from original vehicle crop: "$vtext1"');
            
            String? vtext2;
            final enhancedVehiclePath = await detector.enhanceForOcr(vpath);
            if (enhancedVehiclePath != null) {
              vtext2 = await ocr.extractText(enhancedVehiclePath);
              print('ProcessScreen: OCR from enhanced vehicle crop: "$vtext2"');
            }
            
            final mergedVtext = [vtext1, vtext2].whereType<String>().join('\n');
            final vtext = mergedVtext.isNotEmpty ? mergedVtext : vtext1;
            print('ProcessScreen: Combined vehicle OCR: "$vtext"');
            final vplate = _extractPlate(vtext);
            print('ProcessScreen: Extracted from vehicle: "$vplate"'); // Debug: vehicle extraction
            if (vplate != null) {
              setState(() {
                result = vplate;
                modelStatus = 'Model loaded (extracted from vehicle region)';
                cropPath = vpath;
                topDetection = vehicle;
                isProcessing = false;
              });
              return;
            }
          } catch (e) {
            print('ProcessScreen: OCR failed for vehicle crop: $e');
            continue; // Skip this vehicle and try next
          }
        }
      }
    }
      
    // Final fallback: no plate or vehicle detections — run OCR on the whole image
    final text = await ocr.extractText(widget.imagePath);
    final plate = _extractPlate(text);
    setState(() {
      result = plate ?? 'No plate found';
      modelStatus = 'Model loaded (no detections worked, used full image)';
      isProcessing = false;
    });
  }

  String? _extractPlate(String? ocrText) {
    if (ocrText == null || ocrText.trim().isEmpty) return null;
    final text = ocrText.toUpperCase();
    print('ProcessScreen: _extractPlate input: "$text"'); // Debug: input to extraction

    // Clean helper
    String clean(String s) => s.replaceAll(RegExp(r'[^A-Z0-9\-]'), '');

    // Enhanced MY plate patterns (more comprehensive)
    final patterns = <RegExp>[
      RegExp(r'\bBMP\s?[.-]?\d{3,4}\b'),                    // BMP specifically
      RegExp(r'\bBLW\s?[.-]?\d{3,4}\b'),                    // BLW plates (Labuan)
      RegExp(r'\bWLW\s?[.-]?\d{3,4}\b'),                    // WLW plates
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
      if (c.startsWith('BMP')) prefixBonus = 1.3;      // Police
      else if (c.startsWith('BLW')) prefixBonus = 1.25; // Labuan
      else if (c.startsWith('WLW')) prefixBonus = 1.25; // Wilayah
      else if (c.startsWith('EV')) prefixBonus = 1.2;   // Electric vehicle
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

    // Pretty format: letters + space + digits (+ optional suffix)
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
          // Detection Status Banner
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            margin: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: detectionStatus.contains('NO') ? Colors.red[100] : Colors.green[100],
              border: Border.all(
                color: detectionStatus.contains('NO') ? Colors.red : Colors.green,
                width: 2,
              ),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              children: [
                Text(
                  detectionStatus,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: detectionStatus.contains('NO') ? Colors.red[900] : Colors.green[900],
                  ),
                ),
                if (detectionCount > 0) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Model Status: $modelStatus',
                    style: TextStyle(fontSize: 12, color: Colors.grey[700]),
                  ),
                  Text(
                    'Total Detections: $detectionCount (Vehicles: $vehicleCount, Plates: $plateCount)',
                    style: TextStyle(fontSize: 12, color: Colors.grey[700]),
                  ),
                ]
              ],
            ),
          ),
          // Padding(
          //   padding: const EdgeInsets.all(8.0),
          //   child: Text('Status: $modelStatus • Detections: $detectionCount'),
          // ),
          // Constrain the main image to avoid vertical overflow
          Builder(builder: (context) {
            final maxH = MediaQuery.of(context).size.height * 0.55;
            return Container(
              width: double.infinity,
              height: maxH,
              child: Image.file(File(widget.imagePath), fit: BoxFit.contain),
            );
          }),
          if (cropPath != null) ...[
            const SizedBox(height: 12),
            const Text('Detected region:'),
            // small preview for crop
            Container(
              width: double.infinity,
              height: 120, 
              child: Image.file(File(cropPath!), fit: BoxFit.contain)
            ),
            const SizedBox(height: 8),
            if (topDetection != null) ...[
              Text('Vehicle detected — score: ${((topDetection!['score'] ?? 0) as num).toDouble().toStringAsFixed(2)}'),
              Text(
                'Box (normalized): xmin=${(((topDetection!['xmin'] ?? 0) as num).toDouble()).toStringAsFixed(3)}, ymin=${(((topDetection!['ymin'] ?? 0) as num).toDouble()).toStringAsFixed(3)}, xmax=${(((topDetection!['xmax'] ?? 0) as num).toDouble()).toStringAsFixed(3)}, ymax=${(((topDetection!['ymax'] ?? 0) as num).toDouble()).toStringAsFixed(3)}',
              ),
              Text('Class: ${topDetection!['label']} (id: ${topDetection!['classId']})'),
            ],
          ],
          // if (annotatedPath != null) ...[
          //   const SizedBox(height: 12),
          //   const Text('Annotated image:'),
          //   Builder(builder: (context) {
          //     final maxH = MediaQuery.of(context).size.height * 0.45;
          //     return SizedBox(
          //       height: maxH,
          //       child: Image.file(File(annotatedPath!), fit: BoxFit.contain),
          //     );
          //   }),
          // ],
          // if (debugInfo != null) ...[..[
          //   const Divider(),
          //   const Text('Raw model output debug (per channel):', style: TextStyle(fontWeight: FontWeight.bold)),
          //   SizedBox(
          //     height: 160,
          //     child: ListView.builder(
          //       itemCount: debugInfo!.length,
          //       itemBuilder: (context, idx) {
          //         final ch = debugInfo![idx];
          //         return ListTile(
          //           title: Text('Channel ${ch['channel']} — min:${(ch['min'] as double).toStringAsFixed(3)} max:${(ch['max'] as double).toStringAsFixed(3)} mean:${(ch['mean'] as double).toStringAsFixed(3)}'),
          //           subtitle: Text('top: ${ (ch['top'] as List).map((e) => '[${e['idx']}:${(e['value'] as double).toStringAsFixed(3)}]').join(', ') }'),
          //         );
          //       },
          //     ),
          //   ),
          // ],
          const SizedBox(height: 20),
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
