// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:camera/camera.dart';

import 'package:vehicle_scanner/camera_screen.dart';

void main() {
  testWidgets('Camera screen loads test', (WidgetTester tester) async {
    // Create a mock camera list
    const mockCameras = <CameraDescription>[];
    
    // Build our app and trigger a frame.
    await tester.pumpWidget(MaterialApp(
      home: CameraScreen(cameras: mockCameras),
    ));

    // Verify that the camera screen loads without errors
    expect(find.byType(CameraScreen), findsOneWidget);
  });
}
