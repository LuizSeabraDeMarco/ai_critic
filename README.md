# ai-critic: Automated Risk Auditor for Machine Learning Models**

---

## 🚀 What is ai-critic?

`ai-critic` é um **auditor de risco automatizado baseado em heurísticas** para modelos de *machine learning*. Ele avalia modelos treinados antes da implantação e traduz riscos técnicos de ML em decisões claras e centradas no ser humano.

Em vez de apenas relatar métricas, o `ai-critic` responde à pergunta crítica:

> “Este modelo pode ser implantado com segurança?”

Ele faz isso analisando as principais áreas de risco:

*   **Integridade dos Dados:** (*data leakage*, desequilíbrio, NaNs)
*   **Estrutura do Modelo:** (risco de *overfitting*, complexidade)
*   **Comportamento de Validação:** (pontuações suspeitamente perfeitas)
*   **Robustez:** (sensibilidade a ruído)

Os resultados são organizados em três camadas semânticas para diferentes *stakeholders*:

*   **Executiva:** (tomadores de decisão)
*   **Técnica:** (engenheiros de ML)
*   **Detalhada:** (auditores e depuração)

## 🎯 Por que o ai-critic Existe: Filosofia Central

A maioria das ferramentas de ML:

*   assume que métricas = verdade
*   confia cegamente na validação cruzada
*   despeja números brutos sem interpretação

O `ai-critic` é cético por design.

Ele trata:

*   pontuações perfeitas como **sinais**, não sucesso
*   métricas de robustez como **dependentes do contexto**
*   implantação como uma **decisão de risco**, não um limite de métrica

A filosofia central é: **Métricas não falham modelos — o contexto falha.**

O `ai-critic` aplica heurísticas de raciocínio humano à avaliação de ML:

*   “Isso é bom demais para ser verdade?”
*   “Isso pode estar vazando o alvo (*target*)?”
*   “A robustez importa se a linha de base estiver errada?”

## 🛠️ Instalação

Instale o `ai-critic` via pip:

```bash
pip install ai-critic
```

**Requisitos:**

*   Python ≥ 3.8
*   `scikit-learn`

## 💡 Início Rápido

Audite seu modelo treinado em apenas algumas linhas:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from ai_critic import AICritic

# 1. Carregar dados e treinar um modelo (exemplo)
X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(max_depth=20, random_state=42)
model.fit(X, y)  # O modelo deve estar treinado

# 2. Inicializar e avaliar com ai-critic
critic = AICritic(model, X, y)
report = critic.evaluate()

# A visualização padrão é 'all' (todas as camadas)
print(report)
```

## 🧩 Saída Multi-Camadas

O `ai-critic` nunca despeja tudo de uma vez. Ele estrutura os resultados em camadas de decisão claras.

### 🔹 Visualização Executiva (`view="executive"`)

Projetado para CTOs, gerentes e *stakeholders*. Sem jargão de ML.

```python
critic.evaluate(view="executive")
```

**Exemplo de Saída:**

```json
{
  "verdict": "❌ Não Confiável",
  "risk_level": "high",
  "deploy_recommended": false,
  "main_reason": "Forte evidência de vazamento de dados inflando o desempenho do modelo."
}
```

### 🔹 Visualização Técnica (`view="technical"`)

Projetado para engenheiros de ML. Acionável, diagnóstico e focado no que precisa ser corrigido.

```python
critic.evaluate(view="technical")
```

**Exemplo de Saída:**

```json
{
  "key_risks": [
    "Vazamento de dados suspeito devido à correlação quase perfeita entre recurso e alvo.",
    "Pontuação de validação cruzada perfeita detectada (estatisticamente improvável).",
    "A profundidade da árvore pode ser muito alta para o tamanho do conjunto de dados."
  ],
  "model_health": {
    "data_leakage": true,
    "suspicious_cv": true,
    "structural_risk": true,
    "robustness_verdict": "misleading"
  },
  "recommendations": [
    "Auditar e remover recursos com vazamento.",
    "Reduzir a complexidade do modelo.",
    "Executar novamente a validação após a mitigação do vazamento."
  ]
}
```

### 🔹 Visualização Detalhada (`view="details"`)

Projetado para auditoria, depuração e conformidade.

```python
critic.evaluate(view="details")
```

Inclui:

*   Métricas brutas
*   Correlações de recursos
*   Pontuações de robustez
*   Avisos estruturais
*   Rastreabilidade completa

### 🔹 Visualização Combinada (`view="all"`)

Retorna todas as três camadas em um único dicionário.

```python
critic.evaluate(view="all")
```

**Retorna:**

```json
{
  "executive": {...},
  "technical": {...},
  "details": {...}
}
```

## ⚙️ API Principal

### `AICritic`

| Parâmetro | Descrição |
| :--- | :--- |
| `model` | Modelo `scikit-learn` treinado |
| `X` | Matriz de recursos |
| `y` | Vetor alvo |

**Uso:** `AICritic(model, X, y)`

### `evaluate()`

| Parâmetro | Descrição |
| :--- | :--- |
| `view` | Camada de saída desejada: `"executive"`, `"technical"`, `"details"`, ou `"all"` (padrão) |

**Uso:** `evaluate(view="all")`

## 🧠 O que o ai-critic Detecta

| Categoria | Riscos Detectados |
| :--- | :--- |
| **🔍 Riscos de Dados** | Vazamento de alvo via correlação, NaNs, desequilíbrio de classes |
| **🧱 Riscos Estruturais** | Árvores excessivamente complexas, altas taxas de recurso/amostra, *configuration smells* |
| **📈 Riscos de Validação** | Pontuações de CV suspeitosamente perfeitas, variância irreal |
| **🧪 Riscos de Robustez** | Sensibilidade a ruído, robustez enganosa se a linha de base estiver inflada |

## 🧪 Exemplo: Detectando Vazamento de Dados

```python
import numpy as np
# ... (imports e código do modelo)

# Vazamento artificial: adicionando o alvo como um recurso
X_leaky = np.hstack([X, y.reshape(-1, 1)])

critic = AICritic(model, X_leaky, y)
executive_report = critic.evaluate(view="executive")

print(executive_report)
```

**Saída (Visualização Executiva):**

```
❌ Não Confiável
Forte evidência de vazamento de dados inflando o desempenho do modelo.
```

## 🛡️ Melhores Práticas

*   Execute o `ai-critic` antes da implantação.
*   Nunca confie cegamente em pontuações de CV perfeitas.
*   Use a Visualização Executiva em seu *pipeline* de CI/CD como um portão de modelo.
*   Use a Visualização Técnica durante a iteração do modelo.
*   Use a Visualização Detalhada para auditoria e conformidade.

## 🧭 Casos de Uso Típicos

*   Auditorias de modelo pré-implantação
*   Governança e conformidade de ML
*   Portões de modelo CI/CD
*   Ensino de ceticismo em ML
*   Explicação de risco de ML para *stakeholders* não técnicos

## 📄 Licença

Distribuído sob a Licença MIT.

## 🧠 Nota Final

O `ai-critic` não é uma ferramenta de *benchmarking*. É uma **ferramenta de decisão**.

Se um modelo falhar aqui, não significa que seja ruim — significa que **não deve ser confiável ainda**.
