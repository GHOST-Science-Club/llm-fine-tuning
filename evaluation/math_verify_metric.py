from math_verify import parse, verify
from lm_eval.api.registry import register_aggregation, register_metric
from lm_eval.api.registry import register_metric


@register_aggregation("math_verify_metric")
def evaluate(items):
    results = []
    for gold_str, pred_str in items:
        gold = parse(gold_str)
        answer = parse(pred_str)
        results.append(1 if verify(gold, answer) else 0)

    return sum(results) / len(results) if results else 0



@register_metric(
    metric="math_verify_metric",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="math_verify_metric",
)
def math_verify_fn(items):
    return items