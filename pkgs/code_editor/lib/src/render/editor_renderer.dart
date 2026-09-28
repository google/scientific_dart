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

import 'render_viewport.dart';

/// Callback signature for renderer user interaction events (clicks, keypresses, scroll).
typedef RenderEventListener =
    void Function(String eventType, Map<String, dynamic> data);

/// Abstract contract for editor presentation backends.
abstract class EditorRenderer {
  /// Whether this renderer is attached to a visual output surface.
  bool get isAttached;

  /// The most recent viewport snapshot rendered by this backend.
  RenderViewport? get lastViewport;

  /// Attach the renderer to a target platform node or surface container.
  void attach(Object target);

  /// Detach the renderer from its target surface.
  void detach();

  /// Render the given viewport snapshot onto the target output surface.
  void render(RenderViewport viewport);

  /// Register an event listener for user inputs captured by the renderer backend.
  void addEventListener(RenderEventListener listener);

  /// Remove a registered event listener.
  void removeEventListener(RenderEventListener listener);

  /// Dispose resources used by this renderer.
  void dispose();
}
