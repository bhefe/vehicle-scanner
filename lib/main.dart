import 'package:flutter/material.dart';    
import 'package:camera/camera.dart';

import 'camera_screen.dart';
import 'process_screen.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();

  runApp(MaterialApp(
    title: 'Vehicle Scanner',
    routes: {
      "/": (_) => CameraScreen(cameras: cameras),
      "/process": (context) {
        final path = ModalRoute.of(context)!.settings.arguments as String;
        return ProcessScreen(imagePath: path);
      },
    },
  ));
}
