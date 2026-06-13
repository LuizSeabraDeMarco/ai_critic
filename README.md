# model-audit

**Avaliação completa de modelos ML — muito além da acurácia.**

```bash
pip install model-audit
```

---

## Por que usar?

Acurácia isolada mente. Um modelo pode ter 99 % de acurácia e ainda assim:

- estar vazando o alvo (*data leakage*),
- colapsar sob ruído mínimo nos dados de produção,
- ser injusto com algum grupo demográfico,
- ter probabilidades completamente mal calibradas,
- depender de uma única feature que é um proxy do target.

`model-audit` quantifica **7 dimensões independentes** e entrega um
relatório unificado, acionável e JSON-pronto.

---

## Início rápido

```python
import model_audit
from model_audit.reporters import print_report

# modelo já treinado (sklearn-compatible)
report = model_audit.audit(model, X, y)
print_report(report)
```

Saída:

```
══════════════════════════════════════════════════════════════
  MODEL AUDIT REPORT
  Problem type : multiclass_classification
  Overall score: 0.944  ✅ PASS
══════════════════════════════════════════════════════════════
  Dimension               Score  Verdict     Summary
──────────────────────────────────────────────────────────────
  explainability          0.713  ✅ pass      Gini concentration 0.53 | ...
  calibration             0.935  ✅ pass      Mean ECE: 0.022
  performance             0.954  ✅ pass      Accuracy 0.960 | F1-macro 0.960 | MCC 0.940
  robustness              0.980  ✅ pass      Noise drop 0.027 | Dropout drop 0.031
  data_quality            1.000  ✅ pass      Missing 0.0% | Duplicates 0.7%
  complexity              1.000  ✅ pass      Feature/sample ratio: 0.03
  fairness                1.000  ✅ pass      Max gap: 0.000
══════════════════════════════════════════════════════════════
```

---

## Gate de CI/CD

Bloqueia deploy de modelos abaixo do limiar:

```python
report = model_audit.audit(model, X, y)
model_audit.gate(report, min_score=0.75)  # levanta RuntimeError se falhar
```

---

## Relatório como dict / JSON

```python
import json
d = report.to_dict()
print(json.dumps(d, indent=2))
```

---

## Pesos customizados

```python
report = model_audit.audit(
    model, X, y,
    weights={"robustness": 2.0, "fairness": 1.5}
)
```

---

## Execução paralela

```python
report = model_audit.audit(model, X, y, parallel=True)
```

---

## Evaluadores customizados

```python
from model_audit.core.base import BaseEvaluator
from model_audit.core.types import DimensionResult, ProblemType, Verdict

class MyEvaluator(BaseEvaluator):
    name = "my_check"
    weight = 1.0
    depends_on = []  # ou ["performance"] se precisar do resultado anterior

    def evaluate(self, model, X, y, problem_type, context=None):
        score = 0.95  # sua lógica aqui
        return DimensionResult(
            name=self.name,
            score=score,
            verdict=self._score_to_verdict(score),
            summary="Tudo certo.",
        )

report = model_audit.audit(model, X, y, evaluators=[..., MyEvaluator()])
```

---

## As 7 dimensões

| Dimensão | O que mede |
|---|---|
| **performance** | Accuracy, F1-macro, MCC, R², RMSE — métricas reais, não apenas acurácia |
| **robustness** | Degradação sob ruído gaussiano (4 intensidades), dropout de features, injeção de outliers |
| **explainability** | Importância por permutação, índice de Gini de concentração, detecção de features dominantes |
| **calibration** | ECE, MCE, Brier Score — as probabilidades do modelo são confiáveis? |
| **data_quality** | Missing values, duplicatas, features constantes, outliers, leakage por correlação, imbalance |
| **fairness** | Disparidade de performance entre grupos categóricos, Disparate Impact Ratio |
| **complexity** | Profundidade de árvores, ratio features/amostras, latência de inferência |

---

## Score e veredito

Cada dimensão retorna um score de **0.0 a 1.0** e um veredito:

| Score | Veredito |
|---|---|
| ≥ 0.75 | ✅ `pass` |
| 0.50 – 0.75 | ⚠️ `warning` |
| < 0.50 | ❌ `fail` |

O **overall score** é a média ponderada pelos `weight` de cada evaluador.

---

## Diferenças em relação ao `ai_critic` original

| Aspecto | ai_critic | model-audit |
|---|---|---|
| Métricas de performance | Só accuracy via CV | F1, MCC, ROC-AUC, R², RMSE, MAE, MAPE |
| Robustez | 1 nível de ruído | 4 níveis + dropout + outliers |
| Calibração | ❌ ausente | ECE, MCE, Brier, reliability bins |
| Fairness | ❌ ausente | Disparate Impact, gap de performance por grupo |
| Explainability | Permutação capped em 10 features | Permutação completa + Gini de concentração |
| Data quality | Correlação simples | NaN, duplicatas, constantes, outliers, imbalance |
| Complexidade | Heurística de depth | Depth + ratio + latência real de inferência |
| Tipos | `score` como dict solto | `DimensionResult` + `AuditReport` tipados |
| Extensibilidade | Plugin via registry global | Subclasse `BaseEvaluator`, sem estado global |

---

## Licença

MIT
