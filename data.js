// Complete model data for BeyondBench with enhanced metrics
const modelData = [
    // Qwen3 Series
    { model: "Qwen/Qwen3-0.6B", params: "0.6B", quantization: "None", accuracy: 28.13, instruction: 83.34, tokens: 7911.86, efficiency: 0.035, overthinking: 1.2, words: 4500.2, chars: 20234.5 },
    { model: "Qwen/Qwen3-1.7B", params: "1.7B", quantization: "None", accuracy: 45.63, instruction: 86.40, tokens: 6700.53, efficiency: 0.068, overthinking: 1.1, words: 3850.3, chars: 18901.2 },
    { model: "Qwen/Qwen3-4B", params: "4B", quantization: "None", accuracy: 58.55, instruction: 89.14, tokens: 6850.91, efficiency: 0.085, overthinking: 1.0, words: 3925.5, chars: 19327.3 },
    { model: "Qwen/Qwen3-8B", params: "8B", quantization: "None", accuracy: 59.70, instruction: 89.57, tokens: 5703.25, efficiency: 0.105, overthinking: 0.9, words: 3276.9, chars: 16087.4 },
    { model: "Qwen/Qwen3-14B", params: "14B", quantization: "None", accuracy: 66.35, instruction: 92.16, tokens: 5125.04, efficiency: 0.129, overthinking: 0.8, words: 2942.5, chars: 14451.6 },
    { model: "Qwen/Qwen3-32B", params: "32B", quantization: "None", accuracy: 67.83, instruction: 90.31, tokens: 4631.45, efficiency: 0.146, overthinking: 0.7, words: 2658.3, chars: 13065.1 },
    { model: "Qwen/Qwen3-30B-MOE", params: "30B-MOE", quantization: "None", accuracy: 68.92, instruction: 91.16, tokens: 4712.63, efficiency: 0.146, overthinking: 0.8, words: 2705.0, chars: 13293.4 },
    { model: "Qwen/Qwen3-30B-MOE-Instruct", params: "30B-MOE", quantization: "Instruct", accuracy: 70.40, instruction: 96.46, tokens: 2062.19, efficiency: 0.341, overthinking: 0.3, words: 1183.7, chars: 5816.2 },
    { model: "Qwen/Qwen3-4B-Thinking", params: "4B", quantization: "Thinking", accuracy: 60.85, instruction: 90.26, tokens: 4452.78, efficiency: 0.137, overthinking: 0.7, words: 2556.6, chars: 12559.8 },
    { model: "Qwen/Qwen3-30B-MOE-Thinking", params: "30B-MOE", quantization: "Thinking", accuracy: 69.12, instruction: 91.15, tokens: 4563.24, efficiency: 0.151, overthinking: 0.7, words: 2620.1, chars: 12871.5 },

    // Qwen2.5 Series
    { model: "Qwen/Qwen2.5-0.5B", params: "0.5B", quantization: "None", accuracy: 9.37, instruction: 81.45, tokens: 3185.51, efficiency: 0.029, overthinking: 2.1, words: 1828.6, chars: 8984.0 },
    { model: "Qwen/Qwen2.5-1.5B", params: "1.5B", quantization: "None", accuracy: 19.47, instruction: 88.54, tokens: 2470.13, efficiency: 0.079, overthinking: 1.8, words: 1418.4, chars: 6968.9 },
    { model: "Qwen/Qwen2.5-3B", params: "3B", quantization: "None", accuracy: 24.89, instruction: 92.15, tokens: 1179.50, efficiency: 0.211, overthinking: 1.2, words: 677.3, chars: 3327.5 },
    { model: "Qwen/Qwen2.5-7B", params: "7B", quantization: "None", accuracy: 36.03, instruction: 93.37, tokens: 786.90, efficiency: 0.458, overthinking: 0.8, words: 452.0, chars: 2220.9 },
    { model: "Qwen/Qwen2.5-14B", params: "14B", quantization: "None", accuracy: 41.40, instruction: 96.00, tokens: 608.14, efficiency: 0.681, overthinking: 0.6, words: 349.1, chars: 1715.4 },
    { model: "Qwen/Qwen2.5-32B", params: "32B", quantization: "None", accuracy: 48.01, instruction: 96.31, tokens: 532.48, efficiency: 0.902, overthinking: 0.5, words: 305.7, chars: 1502.0 },
    { model: "Qwen/Qwen2.5-72B", params: "72B", quantization: "None", accuracy: 53.27, instruction: 97.16, tokens: 643.49, efficiency: 0.828, overthinking: 0.6, words: 369.4, chars: 1815.1 },
    { model: "Qwen/Qwen2.5-1.5B-Math", params: "1.5B", quantization: "Math", accuracy: 27.72, instruction: 86.75, tokens: 1037.49, efficiency: 0.267, overthinking: 1.0, words: 595.8, chars: 2926.8 },
    { model: "Qwen/Qwen2.5-7B-Math", params: "7B", quantization: "Math", accuracy: 31.71, instruction: 90.59, tokens: 1309.69, efficiency: 0.242, overthinking: 1.3, words: 752.1, chars: 3694.7 },
    { model: "Qwen/Qwen2.5-72B-Math", params: "72B", quantization: "Math", accuracy: 48.40, instruction: 87.74, tokens: 856.26, efficiency: 0.565, overthinking: 0.8, words: 491.8, chars: 2416.4 },

    // Gemma Series
    { model: "Google/Gemma-1B", params: "1B", quantization: "None", accuracy: 13.53, instruction: 78.50, tokens: 921.92, efficiency: 0.147, overthinking: 1.5, words: 529.3, chars: 2600.2 },
    { model: "Google/Gemma-4B", params: "4B", quantization: "None", accuracy: 38.65, instruction: 92.11, tokens: 890.63, efficiency: 0.434, overthinking: 1.0, words: 511.4, chars: 2512.4 },
    { model: "Google/Gemma-12B", params: "12B", quantization: "None", accuracy: 48.99, instruction: 88.64, tokens: 922.49, efficiency: 0.531, overthinking: 1.1, words: 529.8, chars: 2602.7 },
    { model: "Google/Gemma-27B", params: "27B", quantization: "None", accuracy: 58.20, instruction: 95.93, tokens: 868.14, efficiency: 0.671, overthinking: 0.9, words: 498.5, chars: 2449.0 },

    // Phi Series
    { model: "Microsoft/Phi3-Mini-3.8B", params: "3.8B", quantization: "None", accuracy: 22.41, instruction: 92.36, tokens: 420.05, efficiency: 0.534, overthinking: 0.7, words: 241.2, chars: 1185.1 },
    { model: "Microsoft/Phi3-Med-14B-4K", params: "14B", quantization: "4K", accuracy: 26.54, instruction: 92.78, tokens: 375.49, efficiency: 0.707, overthinking: 0.6, words: 215.7, chars: 1059.7 },
    { model: "Microsoft/Phi3-Med-14B-128K", params: "14B", quantization: "128K", accuracy: 26.79, instruction: 94.69, tokens: 459.92, efficiency: 0.582, overthinking: 0.7, words: 264.2, chars: 1297.8 },
    { model: "Microsoft/Phi4-Mini-3.8B-Instruct", params: "3.8B", quantization: "Instruct", accuracy: 31.74, instruction: 92.71, tokens: 922.97, efficiency: 0.344, overthinking: 1.2, words: 530.1, chars: 2603.9 },
    { model: "Microsoft/Phi4-Mini-Reasoning-3.8B", params: "3.8B", quantization: "Reasoning", accuracy: 49.78, instruction: 87.04, tokens: 6989.60, efficiency: 0.071, overthinking: 2.8, words: 4013.7, chars: 19720.8 },
    { model: "Microsoft/Phi4-14B", params: "14B", quantization: "None", accuracy: 48.45, instruction: 95.55, tokens: 699.40, efficiency: 0.693, overthinking: 0.7, words: 401.7, chars: 1973.8 },
    { model: "Microsoft/Phi4-Reasoning-14B", params: "14B", quantization: "Reasoning", accuracy: 56.59, instruction: 88.06, tokens: 7182.15, efficiency: 0.079, overthinking: 2.9, words: 4124.0, chars: 20262.0 },
    { model: "Microsoft/Phi4-Reasoning+-14B", params: "14B", quantization: "Reasoning+", accuracy: 50.89, instruction: 84.06, tokens: 6681.70, efficiency: 0.076, overthinking: 2.7, words: 3836.4, chars: 18849.6 },

    // Llama Series
    { model: "Meta/Llama-3.2-1B", params: "1B", quantization: "None", accuracy: 7.60, instruction: 69.20, tokens: 3021.32, efficiency: 0.025, overthinking: 2.2, words: 1735.2, chars: 8526.3 },
    { model: "Meta/Llama-3.2-3B", params: "3B", quantization: "None", accuracy: 21.07, instruction: 86.03, tokens: 2693.95, efficiency: 0.078, overthinking: 1.8, words: 1547.1, chars: 7597.8 },
    { model: "Meta/Llama-3.1-8B", params: "8B", quantization: "None", accuracy: 24.10, instruction: 88.64, tokens: 3289.68, efficiency: 0.073, overthinking: 1.9, words: 1889.6, chars: 9281.5 },
    { model: "Meta/Llama-3.1-70B", params: "70B", quantization: "None", accuracy: 43.17, instruction: 95.31, tokens: 1521.18, efficiency: 0.284, overthinking: 0.8, words: 873.9, chars: 4294.3 },
    { model: "Meta/Llama-3.3-70B", params: "70B", quantization: "None", accuracy: 49.40, instruction: 96.26, tokens: 636.54, efficiency: 0.776, overthinking: 0.4, words: 365.7, chars: 1796.7 },
    { model: "Meta/Llama-4-Scout", params: "120B-MOE", quantization: "None", accuracy: 52.97, instruction: 76.14, tokens: 756.04, efficiency: 0.701, overthinking: 0.6, words: 434.2, chars: 2133.1 },

    // Mistral Series
    { model: "Mistral/Mistral-7B", params: "7B", quantization: "None", accuracy: 14.01, instruction: 93.16, tokens: 670.02, efficiency: 0.209, overthinking: 1.0, words: 384.8, chars: 1890.6 },
    { model: "Mistral/Ministral-8B", params: "8B", quantization: "None", accuracy: 27.17, instruction: 92.15, tokens: 855.66, efficiency: 0.317, overthinking: 1.1, words: 491.5, chars: 2414.5 },
    { model: "Mistral/Mistral-Nemo-12B", params: "12B", quantization: "None", accuracy: 21.93, instruction: 89.02, tokens: 728.67, efficiency: 0.301, overthinking: 1.0, words: 418.5, chars: 2056.0 },
    { model: "Mistral/Mixtral-8x7B", params: "8x7B", quantization: "MOE", accuracy: 19.02, instruction: 88.01, tokens: 337.96, efficiency: 0.563, overthinking: 0.5, words: 194.1, chars: 953.4 },
    { model: "Mistral/Mixtral-8x22B", params: "8x22B", quantization: "MOE", accuracy: 29.88, instruction: 88.78, tokens: 531.63, efficiency: 0.562, overthinking: 0.7, words: 305.2, chars: 1499.5 },

    // Other Models
    { model: "HuggingFace/SmolLM3-3B", params: "3B", quantization: "None", accuracy: 32.78, instruction: 76.79, tokens: 8460.83, efficiency: 0.039, overthinking: 3.1, words: 4860.5, chars: 23877.0 },
    { model: "HuggingFace/SmolLM2-1.7B", params: "1.7B", quantization: "None", accuracy: 7.98, instruction: 55.17, tokens: 449.51, efficiency: 0.178, overthinking: 1.2, words: 258.2, chars: 1268.3 },
    { model: "OpenSource/GPT-OSS-20B", params: "20B", quantization: "None", accuracy: 67.57, instruction: 89.71, tokens: 2912.41, efficiency: 0.232, overthinking: 1.0, words: 1672.6, chars: 8216.8 },
    { model: "OpenSource/GPT-OSS-120B", params: "120B", quantization: "None", accuracy: 76.07, instruction: 89.60, tokens: 2002.84, efficiency: 0.380, overthinking: 0.6, words: 1150.2, chars: 5651.0 },

    // Proprietary OpenAI Models
    { model: "OpenAI/GPT5", params: "Unknown", quantization: "Reasoning", accuracy: 83.56, instruction: 96.15, tokens: 2598.97, efficiency: 0.321, overthinking: 0.8, words: 1492.6, chars: 7331.9 },
    { model: "OpenAI/GPT5-Mini", params: "Unknown", quantization: "Reasoning", accuracy: 81.67, instruction: 94.23, tokens: 1923.15, efficiency: 0.425, overthinking: 0.6, words: 1104.9, chars: 5427.6 },
    { model: "OpenAI/GPT5-Nano", params: "Unknown", quantization: "Reasoning", accuracy: 82.04, instruction: 93.58, tokens: 3623.34, efficiency: 0.226, overthinking: 1.1, words: 2081.3, chars: 10223.9 },
    { model: "OpenAI/GPT4.1", params: "Unknown", quantization: "None", accuracy: 70.61, instruction: 96.38, tokens: 2603.19, efficiency: 0.271, overthinking: 0.8, words: 1495.0, chars: 7343.6 },
    { model: "OpenAI/GPT4.1-Mini", params: "Unknown", quantization: "None", accuracy: 69.11, instruction: 95.78, tokens: 2178.40, efficiency: 0.317, overthinking: 0.7, words: 1251.4, chars: 6147.9 },
    { model: "OpenAI/GPT4.1-Nano", params: "Unknown", quantization: "None", accuracy: 52.22, instruction: 96.28, tokens: 1239.73, efficiency: 0.421, overthinking: 0.4, words: 712.1, chars: 3498.4 },
    { model: "OpenAI/GPT4o", params: "Unknown", quantization: "None", accuracy: 57.01, instruction: 96.59, tokens: 458.45, efficiency: 1.244, overthinking: 0.2, words: 263.3, chars: 1293.6 },
    { model: "OpenAI/GPT4o-Mini", params: "Unknown", quantization: "None", accuracy: 42.10, instruction: 96.16, tokens: 538.02, efficiency: 0.783, overthinking: 0.3, words: 309.0, chars: 1518.7 },
    { model: "OpenAI/o4-Mini", params: "Unknown", quantization: "Reasoning", accuracy: 79.04, instruction: 95.30, tokens: 2331.31, efficiency: 0.339, overthinking: 0.7, words: 1339.1, chars: 6579.0 },
    { model: "OpenAI/o3", params: "Unknown", quantization: "Reasoning", accuracy: 80.36, instruction: 94.96, tokens: 3111.22, efficiency: 0.258, overthinking: 0.9, words: 1786.9, chars: 8779.0 },
    { model: "OpenAI/o3-Mini", params: "Unknown", quantization: "Reasoning", accuracy: 77.46, instruction: 96.01, tokens: 2684.08, efficiency: 0.289, overthinking: 0.8, words: 1542.1, chars: 7575.8 },

    // Proprietary Google Models
    { model: "Google/Gemini-2.5-Pro", params: "Unknown", quantization: "None", accuracy: 74.27, instruction: 91.48, tokens: 636.49, efficiency: 1.166, overthinking: 0.3, words: 365.8, chars: 1797.1 },
    { model: "Google/Gemini-2.5-Flash", params: "Unknown", quantization: "None", accuracy: 70.44, instruction: 86.82, tokens: 690.74, efficiency: 1.020, overthinking: 0.4, words: 396.9, chars: 1949.8 },
    { model: "Google/Gemini-2.5-Flash-Lite", params: "Unknown", quantization: "None", accuracy: 58.41, instruction: 96.33, tokens: 6114.18, efficiency: 0.096, overthinking: 2.4, words: 3512.6, chars: 17253.4 },
    { model: "Google/Gemini-2.0-Flash", params: "Unknown", quantization: "None", accuracy: 60.99, instruction: 93.21, tokens: 928.09, efficiency: 0.657, overthinking: 0.5, words: 533.2, chars: 2619.5 },
    { model: "Google/Gemini-2.0-Flash-Lite", params: "Unknown", quantization: "None", accuracy: 56.90, instruction: 95.21, tokens: 1056.60, efficiency: 0.539, overthinking: 0.6, words: 607.0, chars: 2982.4 }
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

// Paper information (same as before)
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
        { name: "Xuan Wang", affiliation: "vt" }
    ],
    affiliations: {
        vt: "Department of Computer Science, Virginia Tech, USA",
        amazon: "Amazon AGI, USA"
    }
};

// Calculate ranks and export
let rankedModelData = calculateRanks(modelData);