// Complete model data for BeyondBench with accurate values from main-table.tex
const modelData = [
    // Qwen3 Series
    { model: "Qwen/Qwen3-0.6B", params: "0.6B", quantization: "None", accuracy: 28.13, instruction: 83.34, tokens: 7911.86, efficiency: 0.035, overthinking: 1.2, words: 4545.5, chars: 22331.4, easy_acc: 49.99, easy_inst: 83.85, easy_tokens: 3162.80, medium_acc: 20.16, medium_inst: 92.00, medium_tokens: 15029.98, hard_acc: 14.24, hard_inst: 74.17, hard_tokens: 5542.79 },
    { model: "Qwen/Qwen3-1.7B", params: "1.7B", quantization: "None", accuracy: 45.63, instruction: 86.40, tokens: 6700.53, efficiency: 0.068, overthinking: 1.1, words: 3850.3, chars: 18901.2, easy_acc: 70.24, easy_inst: 86.54, easy_tokens: 3157.20, medium_acc: 42.54, medium_inst: 92.00, medium_tokens: 10854.63, hard_acc: 24.12, hard_inst: 80.65, hard_tokens: 6089.76 },
    { model: "Qwen/Qwen3-4B", params: "4B", quantization: "None", accuracy: 58.55, instruction: 89.14, tokens: 6850.91, efficiency: 0.085, overthinking: 1.0, words: 3933.0, chars: 19325.1, easy_acc: 81.90, easy_inst: 91.57, easy_tokens: 3091.20, medium_acc: 59.22, medium_inst: 92.00, medium_tokens: 11649.83, hard_acc: 34.52, hard_inst: 83.85, hard_tokens: 5811.69 },
    { model: "Qwen/Qwen3-8B", params: "8B", quantization: "None", accuracy: 59.70, instruction: 89.57, tokens: 5703.25, efficiency: 0.105, overthinking: 0.9, words: 3276.9, chars: 16087.4, easy_acc: 82.10, easy_inst: 91.58, easy_tokens: 3027.80, medium_acc: 59.99, medium_inst: 92.00, medium_tokens: 8295.54, hard_acc: 37.02, hard_inst: 85.13, hard_tokens: 5786.42 },
    { model: "Qwen/Qwen3-14B", params: "14B", quantization: "None", accuracy: 66.35, instruction: 92.16, tokens: 5125.04, efficiency: 0.129, overthinking: 0.8, words: 2942.5, chars: 14451.6, easy_acc: 86.52, easy_inst: 99.27, easy_tokens: 3607.60, medium_acc: 69.78, medium_inst: 92.00, medium_tokens: 6298.80, hard_acc: 42.76, hard_inst: 85.21, hard_tokens: 5468.71 },
    { model: "Qwen/Qwen3-32B", params: "32B", quantization: "None", accuracy: 67.83, instruction: 90.31, tokens: 4631.45, efficiency: 0.146, overthinking: 0.7, words: 2658.3, chars: 13065.1, easy_acc: 84.13, easy_inst: 93.05, easy_tokens: 2845.90, medium_acc: 76.18, medium_inst: 92.00, medium_tokens: 5563.77, hard_acc: 43.17, hard_inst: 85.87, hard_tokens: 5484.68 },
    { model: "Qwen/Qwen3-30B-MOE", params: "30B-MOE", quantization: "None", accuracy: 68.92, instruction: 91.16, tokens: 4712.63, efficiency: 0.146, overthinking: 0.8, words: 2705.0, chars: 13293.4, easy_acc: 88.74, easy_inst: 94.20, easy_tokens: 2415.13, medium_acc: 74.69, medium_inst: 92.00, medium_tokens: 6330.92, hard_acc: 43.33, hard_inst: 87.28, hard_tokens: 5391.83 },
    { model: "Qwen/Qwen3-30B-MOE-Instruct", params: "30B-MOE", quantization: "Instruct", accuracy: 70.40, instruction: 96.46, tokens: 2062.19, efficiency: 0.341, overthinking: 0.3, words: 1183.7, chars: 5816.2, easy_acc: 91.64, easy_inst: 99.46, easy_tokens: 647.17, medium_acc: 73.75, medium_inst: 92.00, medium_tokens: 3019.49, hard_acc: 45.80, hard_inst: 97.93, hard_tokens: 2519.92 },
    { model: "Qwen/Qwen3-4B-Thinking", params: "4B", quantization: "Thinking", accuracy: 60.85, instruction: 90.26, tokens: 4452.78, efficiency: 0.137, overthinking: 0.7, words: 2556.6, chars: 12559.8, easy_acc: 85.60, easy_inst: 95.19, easy_tokens: 2552.09, medium_acc: 61.73, medium_inst: 92.00, medium_tokens: 4874.80, hard_acc: 35.21, hard_inst: 83.59, hard_tokens: 5931.45 },
    { model: "Qwen/Qwen3-30B-MOE-Thinking", params: "30B-MOE", quantization: "Thinking", accuracy: 69.12, instruction: 91.15, tokens: 4563.24, efficiency: 0.151, overthinking: 0.7, words: 2620.1, chars: 12871.5, easy_acc: 90.74, easy_inst: 96.85, easy_tokens: 2052.39, medium_acc: 75.32, medium_inst: 92.00, medium_tokens: 6063.77, hard_acc: 41.29, hard_inst: 84.61, hard_tokens: 5573.57 },

    // Qwen2.5 Series
    { model: "Qwen/Qwen2.5-0.5B", params: "0.5B", quantization: "None", accuracy: 9.37, instruction: 81.45, tokens: 3185.51, efficiency: 0.029, overthinking: 2.1, words: 1828.6, chars: 8984.0, easy_acc: 21.31, easy_inst: 77.57, easy_tokens: 432.30, medium_acc: 5.73, medium_inst: 92.00, medium_tokens: 7381.67, hard_acc: 1.08, hard_inst: 74.77, hard_tokens: 1742.56 },
    { model: "Qwen/Qwen2.5-1.5B", params: "1.5B", quantization: "None", accuracy: 19.47, instruction: 88.54, tokens: 2470.13, efficiency: 0.079, overthinking: 1.8, words: 1418.4, chars: 6968.9, easy_acc: 43.03, easy_inst: 85.45, easy_tokens: 264.70, medium_acc: 11.97, medium_inst: 92.00, medium_tokens: 6014.44, hard_acc: 3.41, hard_inst: 88.16, hard_tokens: 1131.25 },
    { model: "Qwen/Qwen2.5-3B", params: "3B", quantization: "None", accuracy: 24.89, instruction: 92.15, tokens: 1179.50, efficiency: 0.211, overthinking: 1.2, words: 677.3, chars: 3327.5, easy_acc: 45.75, easy_inst: 92.35, easy_tokens: 331.30, medium_acc: 20.57, medium_inst: 92.00, medium_tokens: 2121.82, hard_acc: 8.35, hard_inst: 92.11, hard_tokens: 1085.38 },
    { model: "Qwen/Qwen2.5-7B", params: "7B", quantization: "None", accuracy: 36.03, instruction: 93.37, tokens: 786.90, efficiency: 0.458, overthinking: 0.8, words: 452.0, chars: 2220.9, easy_acc: 61.36, easy_inst: 96.47, easy_tokens: 286.90, medium_acc: 30.07, medium_inst: 92.00, medium_tokens: 1135.23, hard_acc: 16.65, hard_inst: 91.65, hard_tokens: 938.58 },
    { model: "Qwen/Qwen2.5-14B", params: "14B", quantization: "None", accuracy: 41.40, instruction: 96.00, tokens: 608.14, efficiency: 0.681, overthinking: 0.6, words: 349.1, chars: 1715.4, easy_acc: 63.74, easy_inst: 97.83, easy_tokens: 260.20, medium_acc: 37.66, medium_inst: 91.76, medium_tokens: 777.44, hard_acc: 22.81, hard_inst: 98.41, hard_tokens: 786.78 },
    { model: "Qwen/Qwen2.5-32B", params: "32B", quantization: "None", accuracy: 48.01, instruction: 96.31, tokens: 532.48, efficiency: 0.902, overthinking: 0.5, words: 305.7, chars: 1502.0, easy_acc: 72.90, easy_inst: 99.26, easy_tokens: 260.90, medium_acc: 45.04, medium_inst: 91.24, medium_tokens: 650.59, hard_acc: 26.08, hard_inst: 98.42, hard_tokens: 685.95 },
    { model: "Qwen/Qwen2.5-72B", params: "72B", quantization: "None", accuracy: 53.27, instruction: 97.16, tokens: 643.49, efficiency: 0.828, overthinking: 0.6, words: 369.4, chars: 1815.1, easy_acc: 80.27, easy_inst: 99.93, easy_tokens: 315.75, medium_acc: 45.94, medium_inst: 92.00, medium_tokens: 739.62, hard_acc: 33.60, hard_inst: 99.55, hard_tokens: 875.11 },
    { model: "Qwen/Qwen2.5-1.5B-Math", params: "1.5B", quantization: "Math", accuracy: 27.72, instruction: 86.75, tokens: 1037.49, efficiency: 0.267, overthinking: 1.0, words: 595.8, chars: 2926.8, easy_acc: 51.43, easy_inst: 94.04, easy_tokens: 397.10, medium_acc: 27.98, medium_inst: 92.00, medium_tokens: 1427.29, hard_acc: 3.74, hard_inst: 74.22, hard_tokens: 1288.07 },
    { model: "Qwen/Qwen2.5-7B-Math", params: "7B", quantization: "Math", accuracy: 31.71, instruction: 90.59, tokens: 1309.69, efficiency: 0.242, overthinking: 1.3, words: 752.1, chars: 3694.7, easy_acc: 60.68, easy_inst: 94.36, easy_tokens: 411.70, medium_acc: 26.18, medium_inst: 92.00, medium_tokens: 2044.62, hard_acc: 8.27, hard_inst: 85.41, hard_tokens: 1472.74 },
    { model: "Qwen/Qwen2.5-72B-Math", params: "72B", quantization: "Math", accuracy: 48.40, instruction: 87.74, tokens: 856.26, efficiency: 0.565, overthinking: 0.8, words: 491.8, chars: 2416.4, easy_acc: 77.25, easy_inst: 93.02, easy_tokens: 429.30, medium_acc: 55.13, medium_inst: 92.00, medium_tokens: 780.85, hard_acc: 12.82, hard_inst: 78.21, hard_tokens: 1358.64 },

    // Gemma Series
    { model: "Google/Gemma-1B", params: "1B", quantization: "None", accuracy: 13.53, instruction: 78.50, tokens: 921.92, efficiency: 0.147, overthinking: 1.5, words: 529.3, chars: 2600.2, easy_acc: 23.61, easy_inst: 67.21, easy_tokens: 870.85, medium_acc: 16.19, medium_inst: 92.00, medium_tokens: 1229.28, hard_acc: 0.80, hard_inst: 76.29, hard_tokens: 665.62 },
    { model: "Google/Gemma-4B", params: "4B", quantization: "None", accuracy: 38.65, instruction: 92.11, tokens: 890.63, efficiency: 0.434, overthinking: 1.0, words: 511.4, chars: 2512.4, easy_acc: 66.38, easy_inst: 98.56, easy_tokens: 550.90, medium_acc: 38.40, medium_inst: 92.00, medium_tokens: 1072.62, hard_acc: 11.18, hard_inst: 85.76, hard_tokens: 1048.37 },
    { model: "Google/Gemma-12B", params: "12B", quantization: "None", accuracy: 48.99, instruction: 88.64, tokens: 922.49, efficiency: 0.531, overthinking: 1.1, words: 529.8, chars: 2602.7, easy_acc: 75.51, easy_inst: 96.85, easy_tokens: 504.14, medium_acc: 50.00, medium_inst: 92.00, medium_tokens: 1337.32, hard_acc: 21.46, hard_inst: 77.07, hard_tokens: 926.02 },
    { model: "Google/Gemma-27B", params: "27B", quantization: "None", accuracy: 58.20, instruction: 95.93, tokens: 868.14, efficiency: 0.671, overthinking: 0.9, words: 498.5, chars: 2449.0, easy_acc: 78.82, easy_inst: 97.46, easy_tokens: 415.96, medium_acc: 58.80, medium_inst: 92.00, medium_tokens: 1118.70, hard_acc: 36.98, hard_inst: 98.32, hard_tokens: 1069.75 },

    // Phi Series
    { model: "Microsoft/Phi3-Mini-3.8B", params: "3.8B", quantization: "None", accuracy: 22.41, instruction: 92.36, tokens: 420.05, efficiency: 0.534, overthinking: 0.7, words: 241.2, chars: 1185.1, easy_acc: 35.82, easy_inst: 96.58, easy_tokens: 89.40, medium_acc: 19.80, medium_inst: 91.92, medium_tokens: 503.10, hard_acc: 11.61, hard_inst: 88.58, hard_tokens: 667.66 },
    { model: "Microsoft/Phi3-Med-14B-4K", params: "14B", quantization: "4K", accuracy: 26.54, instruction: 92.78, tokens: 375.49, efficiency: 0.707, overthinking: 0.6, words: 215.7, chars: 1059.7, easy_acc: 43.47, easy_inst: 89.87, easy_tokens: 189.30, medium_acc: 21.67, medium_inst: 91.64, medium_tokens: 397.91, hard_acc: 14.47, hard_inst: 96.82, hard_tokens: 539.25 },
    { model: "Microsoft/Phi3-Med-14B-128K", params: "14B", quantization: "128K", accuracy: 26.79, instruction: 94.69, tokens: 459.92, efficiency: 0.582, overthinking: 0.7, words: 264.2, chars: 1297.8, easy_acc: 40.76, easy_inst: 96.26, easy_tokens: 140.00, medium_acc: 23.49, medium_inst: 92.00, medium_tokens: 545.48, hard_acc: 16.13, hard_inst: 95.80, hard_tokens: 694.27 },
    { model: "Microsoft/Phi4-Mini-3.8B-Instruct", params: "3.8B", quantization: "Instruct", accuracy: 31.74, instruction: 92.71, tokens: 922.97, efficiency: 0.344, overthinking: 1.2, words: 530.1, chars: 2603.9, easy_acc: 54.55, easy_inst: 95.02, easy_tokens: 292.10, medium_acc: 24.85, medium_inst: 90.80, medium_tokens: 1297.82, hard_acc: 15.82, hard_inst: 92.30, hard_tokens: 1178.99 },
    { model: "Microsoft/Phi4-Mini-Reasoning-3.8B", params: "3.8B", quantization: "Reasoning", accuracy: 49.78, instruction: 87.04, tokens: 6989.60, efficiency: 0.071, overthinking: 2.8, words: 4013.7, chars: 19720.8, easy_acc: 70.16, easy_inst: 89.56, easy_tokens: 3171.90, medium_acc: 53.24, medium_inst: 92.00, medium_tokens: 11615.07, hard_acc: 25.94, hard_inst: 79.55, hard_tokens: 6181.84 },
    { model: "Microsoft/Phi4-14B", params: "14B", quantization: "None", accuracy: 48.45, instruction: 95.55, tokens: 699.40, efficiency: 0.693, overthinking: 0.7, words: 401.7, chars: 1973.8, easy_acc: 78.92, easy_inst: 97.46, easy_tokens: 378.60, medium_acc: 38.19, medium_inst: 92.00, medium_tokens: 686.87, hard_acc: 28.23, hard_inst: 97.18, hard_tokens: 1032.74 },
    { model: "Microsoft/Phi4-Reasoning-14B", params: "14B", quantization: "Reasoning", accuracy: 56.59, instruction: 88.06, tokens: 7182.15, efficiency: 0.079, overthinking: 2.9, words: 4124.0, chars: 20262.0, easy_acc: 72.23, easy_inst: 96.21, easy_tokens: 6066.20, medium_acc: 61.37, medium_inst: 92.00, medium_tokens: 8792.26, hard_acc: 36.17, hard_inst: 75.98, hard_tokens: 6687.98 },
    { model: "Microsoft/Phi4-Reasoning+-14B", params: "14B", quantization: "Reasoning+", accuracy: 50.89, instruction: 84.06, tokens: 6681.70, efficiency: 0.076, overthinking: 2.7, words: 3836.4, chars: 18849.6, easy_acc: 69.54, easy_inst: 88.89, easy_tokens: 6780.70, medium_acc: 55.00, medium_inst: 92.00, medium_tokens: 6261.74, hard_acc: 28.14, hard_inst: 71.31, hard_tokens: 7002.66 },

    // Llama Series
    { model: "Meta/Llama-3.2-1B", params: "1B", quantization: "None", accuracy: 7.60, instruction: 69.20, tokens: 3021.32, efficiency: 0.025, overthinking: 2.2, words: 1735.2, chars: 8526.3, easy_acc: 16.25, easy_inst: 47.15, easy_tokens: 336.30, medium_acc: 5.37, medium_inst: 92.00, medium_tokens: 6699.03, hard_acc: 1.17, hard_inst: 68.44, hard_tokens: 2028.62 },
    { model: "Meta/Llama-3.2-3B", params: "3B", quantization: "None", accuracy: 21.07, instruction: 86.03, tokens: 2693.95, efficiency: 0.078, overthinking: 1.8, words: 1547.1, chars: 7597.8, easy_acc: 42.54, easy_inst: 89.88, easy_tokens: 279.70, medium_acc: 16.00, medium_inst: 92.00, medium_tokens: 6467.50, hard_acc: 4.66, hard_inst: 76.20, hard_tokens: 1334.66 },
    { model: "Meta/Llama-3.1-8B", params: "8B", quantization: "None", accuracy: 24.10, instruction: 88.64, tokens: 3289.68, efficiency: 0.073, overthinking: 1.9, words: 1889.6, chars: 9281.5, easy_acc: 48.84, easy_inst: 85.66, easy_tokens: 366.40, medium_acc: 15.22, medium_inst: 92.00, medium_tokens: 7590.22, hard_acc: 8.25, hard_inst: 88.27, hard_tokens: 1912.41 },
    { model: "Meta/Llama-3.1-70B", params: "70B", quantization: "None", accuracy: 43.17, instruction: 95.31, tokens: 1521.18, efficiency: 0.284, overthinking: 0.8, words: 873.9, chars: 4294.3, easy_acc: 75.43, easy_inst: 98.12, easy_tokens: 251.20, medium_acc: 31.07, medium_inst: 92.00, medium_tokens: 3130.60, hard_acc: 23.00, hard_inst: 95.80, hard_tokens: 1181.75 },
    { model: "Meta/Llama-3.3-70B", params: "70B", quantization: "None", accuracy: 49.40, instruction: 96.26, tokens: 636.54, efficiency: 0.776, overthinking: 0.4, words: 365.7, chars: 1796.7, easy_acc: 74.59, easy_inst: 97.40, easy_tokens: 312.80, medium_acc: 46.71, medium_inst: 92.00, medium_tokens: 709.45, hard_acc: 26.91, hard_inst: 99.38, hard_tokens: 887.38 },
    { model: "Meta/Llama-4-Scout", params: "120B-MOE", quantization: "None", accuracy: 52.97, instruction: 76.14, tokens: 756.04, efficiency: 0.701, overthinking: 0.6, words: 434.2, chars: 2133.1, easy_acc: 79.05, easy_inst: 92.61, easy_tokens: 321.93, medium_acc: 52.38, medium_inst: 86.20, medium_tokens: 731.44, hard_acc: 27.48, hard_inst: 49.60, hard_tokens: 1214.75 },

    // Mistral Series
    { model: "Mistral/Mistral-7B", params: "7B", quantization: "None", accuracy: 14.01, instruction: 93.16, tokens: 670.02, efficiency: 0.209, overthinking: 1.0, words: 384.8, chars: 1890.6, easy_acc: 27.66, easy_inst: 96.26, easy_tokens: 207.10, medium_acc: 10.03, medium_inst: 92.00, medium_tokens: 635.92, hard_acc: 4.34, hard_inst: 91.23, hard_tokens: 1167.03 },
    { model: "Mistral/Ministral-8B", params: "8B", quantization: "None", accuracy: 27.17, instruction: 92.15, tokens: 855.66, efficiency: 0.317, overthinking: 1.1, words: 491.5, chars: 2414.5, easy_acc: 50.93, easy_inst: 89.70, easy_tokens: 534.28, medium_acc: 21.67, medium_inst: 92.00, medium_tokens: 1160.51, hard_acc: 8.92, hard_inst: 94.74, hard_tokens: 872.20 },
    { model: "Mistral/Mistral-Nemo-12B", params: "12B", quantization: "None", accuracy: 21.93, instruction: 89.02, tokens: 728.67, efficiency: 0.301, overthinking: 1.0, words: 418.5, chars: 2056.0, easy_acc: 35.43, easy_inst: 82.95, easy_tokens: 377.00, medium_acc: 18.08, medium_inst: 92.00, medium_tokens: 1266.19, hard_acc: 12.27, hard_inst: 92.11, hard_tokens: 542.81 },
    { model: "Mistral/Mixtral-8x7B", params: "8x7B", quantization: "MOE", accuracy: 19.02, instruction: 88.01, tokens: 337.96, efficiency: 0.563, overthinking: 0.5, words: 194.1, chars: 953.4, easy_acc: 35.03, easy_inst: 91.47, easy_tokens: 140.47, medium_acc: 12.39, medium_inst: 86.88, medium_tokens: 350.39, hard_acc: 9.64, hard_inst: 85.69, hard_tokens: 523.03 },
    { model: "Mistral/Mixtral-8x22B", params: "8x22B", quantization: "MOE", accuracy: 29.88, instruction: 88.78, tokens: 531.63, efficiency: 0.562, overthinking: 0.7, words: 305.2, chars: 1499.5, easy_acc: 50.06, easy_inst: 79.17, easy_tokens: 296.42, medium_acc: 20.54, medium_inst: 92.00, medium_tokens: 536.00, hard_acc: 19.03, hard_inst: 95.18, hard_tokens: 762.46 },

    // Other Models
    { model: "HuggingFace/SmolLM3-3B", params: "3B", quantization: "None", accuracy: 32.78, instruction: 76.79, tokens: 8460.83, efficiency: 0.039, overthinking: 3.1, words: 4860.5, chars: 23877.0, easy_acc: 69.01, easy_inst: 84.60, easy_tokens: 2985.20, medium_acc: 9.46, medium_inst: 70.69, medium_tokens: 16320.71, hard_acc: 19.86, hard_inst: 75.09, hard_tokens: 6076.59 },
    { model: "HuggingFace/SmolLM2-1.7B", params: "1.7B", quantization: "None", accuracy: 7.98, instruction: 55.17, tokens: 449.51, efficiency: 0.178, overthinking: 1.2, words: 258.2, chars: 1268.3, easy_acc: 16.69, easy_inst: 68.98, easy_tokens: 213.00, medium_acc: 0.52, medium_inst: 8.64, medium_tokens: 65.71, hard_acc: 6.73, hard_inst: 87.89, hard_tokens: 1069.83 },

    // Proprietary OpenAI Models
    { model: "OpenAI/GPT-OSS-20B", params: "20B", quantization: "None", accuracy: 67.57, instruction: 89.71, tokens: 2912.41, efficiency: 0.232, overthinking: 1.0, words: 1672.6, chars: 8216.8, easy_acc: 87.24, easy_inst: 95.89, easy_tokens: 1185.34, medium_acc: 63.13, medium_inst: 92.00, medium_tokens: 3674.40, hard_acc: 52.35, hard_inst: 81.23, hard_tokens: 3877.50 },
    { model: "OpenAI/GPT-OSS-120B", params: "120B", quantization: "None", accuracy: 76.07, instruction: 89.60, tokens: 2002.84, efficiency: 0.380, overthinking: 0.6, words: 1150.2, chars: 5651.0, easy_acc: 93.52, easy_inst: 99.01, easy_tokens: 821.55, medium_acc: 75.05, medium_inst: 92.00, medium_tokens: 2205.27, hard_acc: 59.64, hard_inst: 77.80, hard_tokens: 2981.69 },
    { model: "OpenAI/GPT5", params: "", quantization: "Reasoning", accuracy: 83.56, instruction: 96.15, tokens: 2598.97, efficiency: 0.321, overthinking: 0.8, words: 1492.6, chars: 7331.9, easy_acc: 97.26, easy_inst: 99.82, easy_tokens: 992.41, medium_acc: 81.60, medium_inst: 92.00, medium_tokens: 2278.59, hard_acc: 71.81, hard_inst: 96.64, hard_tokens: 4525.92 },
    { model: "OpenAI/GPT5-Mini", params: "", quantization: "Reasoning", accuracy: 81.67, instruction: 94.23, tokens: 1923.15, efficiency: 0.425, overthinking: 0.6, words: 1104.9, chars: 5427.6, easy_acc: 96.01, easy_inst: 100.00, easy_tokens: 798.72, medium_acc: 79.60, medium_inst: 92.00, medium_tokens: 1640.01, hard_acc: 69.41, hard_inst: 90.70, hard_tokens: 3330.71 },
    { model: "OpenAI/GPT5-Nano", params: "", quantization: "Reasoning", accuracy: 82.04, instruction: 93.58, tokens: 3623.34, efficiency: 0.226, overthinking: 1.1, words: 2081.3, chars: 10223.9, easy_acc: 96.19, easy_inst: 99.29, easy_tokens: 1377.00, medium_acc: 80.00, medium_inst: 91.20, medium_tokens: 2414.85, hard_acc: 69.92, hard_inst: 90.25, hard_tokens: 7078.18 },
    { model: "OpenAI/GPT4.1", params: "", quantization: "None", accuracy: 70.61, instruction: 96.38, tokens: 2603.19, efficiency: 0.271, overthinking: 0.8, words: 1495.0, chars: 7343.6, easy_acc: 92.08, easy_inst: 100.00, easy_tokens: 409.38, medium_acc: 72.80, medium_inst: 92.00, medium_tokens: 3718.96, hard_acc: 46.96, hard_inst: 97.13, hard_tokens: 3681.23 },
    { model: "OpenAI/GPT4.1-Mini", params: "", quantization: "None", accuracy: 69.11, instruction: 95.78, tokens: 2178.40, efficiency: 0.317, overthinking: 0.7, words: 1251.4, chars: 6147.9, easy_acc: 93.39, easy_inst: 100.00, easy_tokens: 915.70, medium_acc: 72.40, medium_inst: 92.00, medium_tokens: 2549.95, hard_acc: 41.53, hard_inst: 95.35, hard_tokens: 3069.56 },
    { model: "OpenAI/GPT4.1-Nano", params: "", quantization: "None", accuracy: 52.22, instruction: 96.28, tokens: 1239.73, efficiency: 0.421, overthinking: 0.4, words: 712.1, chars: 3498.4, easy_acc: 80.30, easy_inst: 98.21, easy_tokens: 1161.08, medium_acc: 53.20, medium_inst: 92.00, medium_tokens: 1254.45, hard_acc: 23.16, hard_inst: 98.64, hard_tokens: 1303.67 },
    { model: "OpenAI/GPT4o", params: "", quantization: "None", accuracy: 57.01, instruction: 96.59, tokens: 458.45, efficiency: 1.244, overthinking: 0.2, words: 263.3, chars: 1293.6, easy_acc: 87.92, easy_inst: 100.00, easy_tokens: 256.84, medium_acc: 54.40, medium_inst: 92.00, medium_tokens: 536.46, hard_acc: 28.71, hard_inst: 97.78, hard_tokens: 582.04 },
    { model: "OpenAI/GPT4o-Mini", params: "", quantization: "None", accuracy: 42.10, instruction: 96.16, tokens: 538.02, efficiency: 0.783, overthinking: 0.3, words: 309.0, chars: 1518.7, easy_acc: 73.57, easy_inst: 98.57, easy_tokens: 272.32, medium_acc: 38.00, medium_inst: 92.00, medium_tokens: 522.71, hard_acc: 14.73, hard_inst: 97.90, hard_tokens: 819.02 },
    { model: "OpenAI/o4-Mini", params: "", quantization: "Reasoning", accuracy: 79.04, instruction: 95.30, tokens: 2331.31, efficiency: 0.339, overthinking: 0.7, words: 1339.1, chars: 6579.0, easy_acc: 94.40, easy_inst: 100.00, easy_tokens: 993.38, medium_acc: 81.60, medium_inst: 92.00, medium_tokens: 1486.20, hard_acc: 61.12, hard_inst: 93.89, hard_tokens: 4514.36 },
    { model: "OpenAI/o3", params: "", quantization: "Reasoning", accuracy: 80.36, instruction: 94.96, tokens: 3111.22, efficiency: 0.258, overthinking: 0.9, words: 1786.9, chars: 8779.0, easy_acc: 97.26, easy_inst: 100.00, easy_tokens: 856.56, medium_acc: 82.00, medium_inst: 90.40, medium_tokens: 2891.81, hard_acc: 61.81, hard_inst: 94.49, hard_tokens: 5585.29 },
    { model: "OpenAI/o3-Mini", params: "", quantization: "Reasoning", accuracy: 77.46, instruction: 96.01, tokens: 2684.08, efficiency: 0.289, overthinking: 0.8, words: 1542.1, chars: 7575.8, easy_acc: 93.99, easy_inst: 99.64, easy_tokens: 1101.22, medium_acc: 80.80, medium_inst: 92.00, medium_tokens: 1525.04, hard_acc: 57.59, hard_inst: 96.40, hard_tokens: 5425.99 },

    // Proprietary Google Models
    { model: "Google/Gemini-2.5-Pro", params: "", quantization: "None", accuracy: 74.27, instruction: 91.48, tokens: 636.49, efficiency: 1.166, overthinking: 0.3, words: 365.8, chars: 1797.1, easy_acc: 89.23, easy_inst: 87.32, easy_tokens: 267.57, medium_acc: 77.20, medium_inst: 89.20, medium_tokens: 725.72, hard_acc: 56.38, hard_inst: 97.92, hard_tokens: 916.18 },
    { model: "Google/Gemini-2.5-Flash", params: "", quantization: "None", accuracy: 70.44, instruction: 86.82, tokens: 690.74, efficiency: 1.020, overthinking: 0.4, words: 396.9, chars: 1949.8, easy_acc: 90.00, easy_inst: 95.65, easy_tokens: 357.62, medium_acc: 70.00, medium_inst: 76.00, medium_tokens: 617.25, hard_acc: 51.32, hard_inst: 88.82, hard_tokens: 1097.35 },
    { model: "Google/Gemini-2.5-Flash-Lite", params: "", quantization: "None", accuracy: 58.41, instruction: 96.33, tokens: 6114.18, efficiency: 0.096, overthinking: 2.4, words: 3512.6, chars: 17253.4, easy_acc: 86.01, easy_inst: 100.00, easy_tokens: 587.71, medium_acc: 50.00, medium_inst: 92.00, medium_tokens: 11929.26, hard_acc: 39.21, hard_inst: 97.00, hard_tokens: 5825.56 },
    { model: "Google/Gemini-2.0-Flash", params: "", quantization: "None", accuracy: 60.99, instruction: 93.21, tokens: 928.09, efficiency: 0.657, overthinking: 0.5, words: 533.2, chars: 2619.5, easy_acc: 84.52, easy_inst: 98.69, easy_tokens: 313.89, medium_acc: 67.20, medium_inst: 92.00, medium_tokens: 1044.79, hard_acc: 31.24, hard_inst: 88.94, hard_tokens: 1425.60 },
    { model: "Google/Gemini-2.0-Flash-Lite", params: "", quantization: "None", accuracy: 56.90, instruction: 95.21, tokens: 1056.60, efficiency: 0.539, overthinking: 0.6, words: 607.0, chars: 2982.4, easy_acc: 82.02, easy_inst: 98.81, easy_tokens: 286.12, medium_acc: 57.60, medium_inst: 92.00, medium_tokens: 1043.37, hard_acc: 31.07, hard_inst: 94.81, hard_tokens: 1840.31 }
];

// Calculate efficiency score based on accuracy and token usage
function calculateEfficiency(accuracy, tokens) {
    return (accuracy / 100) / (tokens / 1000);
}

// Calculate ranks for all models
function calculateRanks(data) {
    const sortedData = [...data].sort((a, b) => b.accuracy - a.accuracy);
    return sortedData.map((model, index) => ({
        ...model,
        rank: index + 1
    }));
}

// Get model families for grouping
function getModelFamilies() {
    const families = {};
    modelData.forEach(model => {
        const family = model.model.split('/')[0];
        if (!families[family]) {
            families[family] = [];
        }
        families[family].push(model);
    });
    return families;
}

// Paper information
const paperInfo = {
    title: "BeyondBench: Benchmark-Free Evaluation of Reasoning in Language Models",
    authors: [
        { name: "Gaurav Srivastava", affiliation: "vt", corresponding: true, email: "gks@vt.edu" },
        { name: "Aafiya Hussain", affiliation: "vt" },
        { name: "Zhenyu Bi", affiliation: "vt" },
        { name: "Swastik Roy", affiliation: "amazon" },
        { name: "Priya Pitre", affiliation: "vt" },
        { name: "Meng Lu", affiliation: "vt" },
        { name: "Morteza Ziyadi", affiliation: "amazon" },
        { name: "Xuan Wang", affiliation: "vt", corresponding: true, email: "xuanw@vt.edu" }
    ],
    affiliations: {
        vt: "Department of Computer Science, Virginia Tech, USA",
        amazon: "Amazon AGI, USA"
    }
};

// Calculate ranks and export
let rankedModelData = calculateRanks(modelData);