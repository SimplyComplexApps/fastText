import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:fasttext/fasttext.dart';
import 'package:flutter/services.dart';

Future<void> main() async {
  await RustLib.init();
  runApp(const MyApp());
}

String formatDuration(Duration duration) {
  final minutes = duration.inMinutes;
  final seconds = duration.inSeconds % 60; // Seconds within the current minute
  final milliseconds = duration.inMilliseconds % 1000; // Milliseconds within the current second
  final microseconds = duration.inMicroseconds % 1000; // Microseconds within the current millisecond

  return '${minutes}m ${seconds}s ${milliseconds}ms $microsecondsµs';
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late final TextEditingController kTextEditingController = TextEditingController(text: k.toString());
  late final TextEditingController thresholdTextEditingController = TextEditingController(text: threshold.toString());
  late final TextEditingController inputTextEditingController = TextEditingController();
  FastText? fasttext;

  int k = 2;
  double threshold = 0.0;

  Duration? lastPredictionTime;
  String? predsStr;

  void init() async {
    fasttext = FastText();
    print('fasttext loaded!');
    print('Loading model');
    final modelByteData = await rootBundle.load('assets/models/lid.176.ftz');
    final buffer = modelByteData.buffer.asUint8List().toList(growable: false);
    fasttext?.loadModelFromBuffer(buffer: buffer);
    print('Model loaded!');
    setState(() {});
  }

  void onKChanged(String text) {
    final k = int.tryParse(text);
    if (k != null && k > 0) {
      this.k = k;
    }
    kTextEditingController.text = text;
    setState(() {});

    makePrediction();
  }

  void onThresholdChanged(String text) {
    final threshold = double.tryParse(text);
    if (threshold != null && threshold >= 0 && threshold <= 1) {
      this.threshold = threshold;
    }
    thresholdTextEditingController.text = text;
    setState(() {});

    makePrediction();
  }

  void onTextChanged(String _) {
    makePrediction();
  }

  void makePrediction() async {
    if (fasttext == null) return;

    final sw = Stopwatch()..start();
    final preds = await fasttext?.predict(
      text: inputTextEditingController.text,
      k: k,
      threshold: threshold,
    );
    sw.stop();
    lastPredictionTime = sw.elapsed;
    predsStr = preds
        ?.map((p) => { 'label': p.label, 'probability': p.probability})
        .map((obj) => jsonEncode(obj))
        .join('\n');
    setState(() {});
  }

  @override
  void initState() {
    init();
    super.initState();
  }

  @override
  void dispose() {
    kTextEditingController.dispose();
    thresholdTextEditingController.dispose();
    inputTextEditingController.dispose();
    fasttext?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(title: const Text('fasttext')),
        body: Center(
          child: fasttext == null ? const CircularProgressIndicator() : Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: kTextEditingController,
                        onChanged: onKChanged,
                        keyboardType: TextInputType.number,
                        inputFormatters: <TextInputFormatter>[
                          FilteringTextInputFormatter.digitsOnly,
                        ],
                        decoration: InputDecoration(
                          labelText: 'k',
                          border: OutlineInputBorder(),  // Optional border style
                        ),
                      ),
                    ),
                    const SizedBox(width: 18,),
                    Expanded(
                      child: TextField(
                        controller: thresholdTextEditingController,
                        onChanged: onThresholdChanged,
                        keyboardType: TextInputType.numberWithOptions(decimal: true),
                        inputFormatters: <TextInputFormatter>[
                          FilteringTextInputFormatter.allow(RegExp(r'^[01]$|^0\.\d*$')),
                        ],
                        decoration: InputDecoration(
                          labelText: 'threshold',
                          border: OutlineInputBorder(),  // Optional border style
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: TextField(
                  onChanged: onTextChanged,
                  controller: inputTextEditingController,
                  decoration: InputDecoration(
                    labelText: 'Text to predict',
                    border: OutlineInputBorder(),  // Optional border style
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(left: 8.0, right: 8.0, bottom: 16.0),
                child: Text.rich(
                  TextSpan(
                    children: [
                      TextSpan(text: 'k=$k'),
                      const TextSpan(text: '\t\t\t'),
                      TextSpan(text: 'threshold=$threshold'),
                      const TextSpan(text: '\n'),
                      const TextSpan(text: 'Made prediction in: '),
                      TextSpan(
                        text: formatDuration(lastPredictionTime ?? Duration.zero),
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    children: [
                      Text(predsStr ?? 'Enter some text in "Text to predict"'),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
