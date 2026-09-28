// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'dart:io';
import 'package:puppeteer/puppeteer.dart';

void main() async {
  print('Launching browser...');
  final browser = await puppeteer.launch(
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  );
  final page = await browser.newPage();

  page.onConsole.listen((msg) {
    print('CONSOLE [${msg.type}]: ${msg.text}');
  });

  page.onError.listen((err) {
    print('PAGE ERROR: $err');
  });

  print('Navigating to http://localhost:8080 ...');
  try {
    await page.goto('http://localhost:8080', wait: Until.networkIdle);
  } catch (e) {
    print('Navigation error: $e');
  }

  await Future.delayed(Duration(seconds: 3));

  final screenshot = await page.screenshot();
  File('/tmp/notebook_debug.png').writeAsBytesSync(screenshot);
  print('Saved debug screenshot to /tmp/notebook_debug.png');

  final statusText = await page.evaluate(
    "() => document.getElementById('statusText') ? document.getElementById('statusText').innerText : 'NO STATUS'",
  );
  print('Status Text on Page: $statusText');

  final cellCount = await page.evaluate(
    "() => document.querySelectorAll('.cell').length",
  );
  print('Cell Count on Page: $cellCount');

  await browser.close();
}
