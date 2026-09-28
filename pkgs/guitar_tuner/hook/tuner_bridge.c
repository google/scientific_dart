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

#define MINIAUDIO_IMPLEMENTATION
#include "../lib/src/miniaudio.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    ma_device device;
    float* ringBuffer;
    int bufferSize;
    int writeIndex;
} TunerContext;

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    TunerContext* pContext = (TunerContext*)pDevice->pUserData;
    if (pInput == NULL) return;

    float* fInput = (float*)pInput;
    for (ma_uint32 i = 0; i < frameCount; i++) {
        pContext->ringBuffer[pContext->writeIndex] = fInput[i];
        pContext->writeIndex = (pContext->writeIndex + 1) % pContext->bufferSize;
    }

    (void)pOutput;
}

TunerContext* tuner_init(int sampleRate, int ringBufferSize) {
    TunerContext* pContext = (TunerContext*)malloc(sizeof(TunerContext));
    pContext->bufferSize = ringBufferSize;
    pContext->ringBuffer = (float*)calloc(ringBufferSize, sizeof(float));
    pContext->writeIndex = 0;

    ma_device_config deviceConfig;
    deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32;
    deviceConfig.capture.channels = 1;
    deviceConfig.sampleRate       = sampleRate;
    deviceConfig.dataCallback     = data_callback;
    deviceConfig.pUserData         = pContext;

    if (ma_device_init(NULL, &deviceConfig, &pContext->device) != MA_SUCCESS) {
        free(pContext->ringBuffer);
        free(pContext);
        return NULL;
    }

    if (ma_device_start(&pContext->device) != MA_SUCCESS) {
        ma_device_uninit(&pContext->device);
        free(pContext->ringBuffer);
        free(pContext);
        return NULL;
    }

    return pContext;
}

void tuner_get_samples(TunerContext* pContext, float* output, int count) {
    int readIndex = (pContext->writeIndex - count + pContext->bufferSize) % pContext->bufferSize;
    for (int i = 0; i < count; i++) {
        output[i] = pContext->ringBuffer[(readIndex + i) % pContext->bufferSize];
    }
}

void tuner_close(TunerContext* pContext) {
    if (pContext == NULL) return;
    ma_device_stop(&pContext->device);
    ma_device_uninit(&pContext->device);
    free(pContext->ringBuffer);
    free(pContext);
}
