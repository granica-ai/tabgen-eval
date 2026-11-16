"""Stable report comparisons independent of live resource snapshots."""


def without_memory_plans(value):
    """Retain scores and procedure metadata while omitting live memory observations."""
    if isinstance(value, dict):
        return {
            key: without_memory_plans(item)
            for key, item in value.items()
            if key not in {"memory_plan", "preparation_memory_plan", "scoring_memory_plan"}
        }
    if isinstance(value, list):
        return [without_memory_plans(item) for item in value]
    return value
