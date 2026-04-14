from typing import Dict, Type
from evaluators.base import Evaluator

_registry: Dict[str, Type[Evaluator]] = {}


def register(evaluator_cls: Type[Evaluator]):
    _registry[evaluator_cls.name] = evaluator_cls


def get_evaluators():
    return [cls() for cls in _registry.values()]