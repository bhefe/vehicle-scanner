import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

class PlateOCR {
  final textRecognizer = TextRecognizer();

  Future<String> extractText(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    final recognisedText = await textRecognizer.processImage(inputImage);

    return recognisedText.text;
  }
}
