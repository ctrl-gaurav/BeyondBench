# Parser Corpus

Real model response samples for parser testing and validation.

## Structure

```
parser_corpus/
  samples/
    {task_name}/
      {model_family}/
        sample_{NNN}.json
  collect_corpus.py     — collection script
  corpus_stats.json     — summary statistics
```

## Sample Format

```json
{
  "task_name": "sum",
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "model_family": "qwen",
  "response": "The sum of the numbers is \\boxed{42}",
  "expected_answer": 42,
  "expected_type": "int",
  "unified_parse": 42,
  "legacy_parse": 42,
  "unified_correct": true,
  "legacy_correct": true
}
```

## Collecting Samples

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_parser_corpus.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --tasks sum,sorting,comparison,fibonacci_sequence \
  --datapoints 20 \
  --output tests/parser_corpus/
```
