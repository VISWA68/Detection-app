import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_tts/flutter_tts.dart';

void main() => runApp(YOLOApp());

class YOLOApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _imageFile;
  String? _outputText;
  ImageProvider? _outputImage;
  bool _isProcessing = false;

  final ImagePicker _picker = ImagePicker();
  final FlutterTts _flutterTts = FlutterTts(); // TTS instance

  // Function to pick image from gallery
  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _imageFile = File(pickedFile.path);
        _outputText = null;
        _outputImage = null;
      });
    }
  }

  // Function to capture image using camera
  Future<void> _captureImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      setState(() {
        _imageFile = File(pickedFile.path);
        _outputText = null;
        _outputImage = null;
      });
    }
  }

  // Function to send image to the server and process it
  Future<void> _processImage() async {
    if (_imageFile == null) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://192.168.234.231:5000/predict'),
      );
      request.files
          .add(await http.MultipartFile.fromPath('image', _imageFile!.path));

      final response = await request.send();
      if (response.statusCode == 200) {
        final responseData = await response.stream.bytesToString();
        final data = json.decode(responseData);

        setState(() {
          _outputText = _generateSummary(data['predictions']);
          _outputImage = MemoryImage(base64Decode(data['image'].split(',')[1]));
        });

        // Use TTS to speak the result
        _speak(_outputText!);
      } else {
        throw Exception('Failed to process the image');
      }
    } catch (e) {
      setState(() {
        _outputText = 'Error: ${e.toString()}';
        print(_outputText);
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  // Function to generate summary from predictions
  String _generateSummary(List<dynamic> predictions) {
    final counts = <String, int>{};
    for (var pred in predictions) {
      counts[pred['label']] = (counts[pred['label']] ?? 0) + 1;
    }

    return counts.entries
        .map((entry) =>
            '${entry.value} ${entry.key}${entry.value > 1 ? 's' : ''}')
        .join(', ');
  }

  // Function to speak the text using TTS
  Future<void> _speak(String text) async {
    await _flutterTts.speak(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Object Detection'),
        centerTitle: true,
        backgroundColor: Colors.teal,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            _imageFile == null
                ? const Text(
                    'No image selected',
                    style: TextStyle(fontSize: 18),
                  )
                : Image.file(_imageFile!, height: 200),
            if (_isProcessing)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 16.0),
                child: CircularProgressIndicator(),
              ),
            if (_outputImage != null)
              Image(
                image: _outputImage!,
                height: 200,
              ),
            if (_outputText != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 16.0),
                child: Text(
                  _outputText!,
                  style: const TextStyle(
                      fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
            const Spacer(),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _pickImage,
                  icon: const Icon(Icons.photo),
                  label: const Text('Gallery'),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.teal),
                ),
                ElevatedButton.icon(
                  onPressed: _captureImage,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.teal),
                ),
              ],
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isProcessing ? null : _processImage,
              style: ElevatedButton.styleFrom(backgroundColor: Colors.teal),
              child: const Text('Process Image'),
            ),
          ],
        ),
      ),
    );
  }
}
