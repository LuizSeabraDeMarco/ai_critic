"""
Audit Pipeline
==============
Orchestrates all evaluators in dependency order and assembles the
final AuditReport.
"""
from __future__ import annotations

import concurrent.futures
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from ai_critic.core.base import BaseEvaluator
from ai_critic.core.types import AuditReport, DimensionResult, ProblemType, Verdict
from ai_critic.utils.inference import infer_problem_type


class AuditPipeline:
    """
    Parameters
    ----------
    evaluators    : list[BaseEvaluator]
        Ordered or unordered list; the pipeline resolves dependency order.
    weights       : dict[str, float] | None
        Override per-evaluator weight.
    feature_names : list[str] | None
        Column names from a DataFrame, forwarded to evaluators via context.
    """

    def __init__(
        self,
        evaluators: List[BaseEvaluator],
        weights: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.evaluators: Dict[str, BaseEvaluator] = {e.name: e for e in evaluators}
        self.feature_names = feature_names

        if weights:
            for name, w in weights.items():
                if name in self.evaluators:
                    self.evaluators[name].weight = w

        self._order = self._resolve_order()

    # ------------------------------------------------------------------
    def run(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        parallel: bool = False,
    ) -> AuditReport:

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        problem_type = infer_problem_type(y)

        if parallel:
            results = self._run_parallel(model, X, y, problem_type)
        else:
            results = self._run_sequential(model, X, y, problem_type)

        return self._build_report(results, problem_type)

    # ------------------------------------------------------------------
    def _base_context(self) -> Dict[str, Any]:
        """Shared context injected into every evaluator call."""
        return {"feature_names": self.feature_names}

    def _run_sequential(self, model, X, y, problem_type) -> Dict[str, DimensionResult]:
        results: Dict[str, DimensionResult] = {}
        for name in self._order:
            ev = self.evaluators[name]
            ctx = {dep: results[dep] for dep in ev.depends_on if dep in results}
            ctx.update(self._base_context())
            try:
                results[name] = ev.evaluate(model, X, y, problem_type, context=ctx)
            except Exception as exc:
                results[name] = DimensionResult(
                    name=name,
                    score=0.0,
                    verdict=Verdict.FAIL,
                    summary=f"Evaluator error: {exc}",
                    details={"error": str(exc)},
                )
        return results

    def _run_parallel(self, model, X, y, problem_type) -> Dict[str, DimensionResult]:
        """Parallel execution respecting dependency order via level-by-level BFS."""
        results: Dict[str, DimensionResult] = {}
        in_degree = {
            n: len([d for d in ev.depends_on if d in self.evaluators])
            for n, ev in self.evaluators.items()
        }
        levels: List[List[str]] = []
        remaining = set(self.evaluators)
        while remaining:
            level = [n for n in remaining if in_degree[n] == 0]
            if not level:
                break
            levels.append(level)
            remaining -= set(level)
            for n in level:
                for other, ev in self.evaluators.items():
                    if n in ev.depends_on:
                        in_degree[other] -= 1

        for level in levels:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = {}
                for name in level:
                    ev = self.evaluators[name]
                    ctx = {dep: results[dep] for dep in ev.depends_on if dep in results}
                    ctx.update(self._base_context())
                    futures[pool.submit(ev.evaluate, model, X, y, problem_type, ctx)] = name
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as exc:
                        results[name] = DimensionResult(
                            name=name, score=0.0, verdict=Verdict.FAIL,
                            summary=f"Evaluator error: {exc}",
                            details={"error": str(exc)},
                        )
        return results

    # ------------------------------------------------------------------
    def _build_report(
        self, results: Dict[str, DimensionResult], problem_type: ProblemType
    ) -> AuditReport:
        total_weight = sum(self.evaluators[n].weight for n in results)
        overall = (
            sum(results[n].score * self.evaluators[n].weight for n in results) / total_weight
            if total_weight > 0 else 0.0
        )

        n_fail = sum(1 for r in results.values() if r.verdict == Verdict.FAIL)
        n_warn = sum(1 for r in results.values() if r.verdict == Verdict.WARNING)
        if n_fail > 0:
            verdict = Verdict.FAIL
        elif n_warn > 0:
            verdict = Verdict.WARNING
        else:
            verdict = Verdict.PASS

        seen: set = set()
        top_suggestions: List[str] = []
        for dim in sorted(results.values(), key=lambda r: r.score):
            for s in dim.suggestions:
                if s not in seen:
                    seen.add(s)
                    top_suggestions.append(s)

        return AuditReport(
            problem_type=problem_type,
            dimensions=results,
            overall_score=round(overall, 4),
            overall_verdict=verdict,
            top_suggestions=top_suggestions[:10],
        )

    # ------------------------------------------------------------------
    def _resolve_order(self) -> List[str]:
        in_degree = {n: 0 for n in self.evaluators}
        adj: Dict[str, List[str]] = {n: [] for n in self.evaluators}

        for name, ev in self.evaluators.items():
            for dep in ev.depends_on:
                if dep in self.evaluators:
                    adj[dep].append(name)
                    in_degree[name] += 1

        queue = deque(n for n in self.evaluators if in_degree[n] == 0)
        order: List[str] = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(order) != len(self.evaluators):
            raise ValueError("Circular dependency detected among evaluators.")
        return order