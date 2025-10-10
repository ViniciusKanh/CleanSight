---
title: CleanSight API
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# CleanSight — REST API

API REST em **Flask** para **pré-processamento de dados** (upload, análise, limpeza, codificação, normalização, PCA 2D/3D, outliers e perfilamento rápido).  
Endpoints sob `/api/*`. Verificação de saúde em `/health`.

> **Importante:** o subdomínio final depende do nome do Space e do usuário.  
> Ex.: `https://<usuario>-cleansight-api.hf.space`.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Endpoints](#endpoints)
- [Esquemas de Requisição/Resposta](#esquemas-de-requisiçãoresposta)
- [Variáveis de Ambiente](#variáveis-de-ambiente)
- [Como Executar Localmente](#como-executar-localmente)
- [Implantação no Hugging Face (SDK: Docker)](#implantação-no-hugging-face-sdk-docker)
- [Exemplos (cURL)](#exemplos-curl)
- [Limites e Observações](#limites-e-observações)
- [Resolução de Problemas](#resolução-de-problemas)
- [Licença](#licença)

---

## Visão Geral

O **CleanSight API** transforma arquivos brutos em datasets prontos para análise e modelagem de Machine Learning.  
Principais operações:

- Upload e leitura robusta (detecção de *encoding* e *delimiter*).
- Análise de *target* (distribuição, balanceamento, completude).
- Pipeline de pré-processamento configurável:
  - Remoção de duplicatas
  - Tratamento de ausentes (numérico/categórico)
  - Codificação de categóricas/booleanas
  - Normalização (z-score)
  - Seleção simples de *features*
  - Balanceamento de classes (oversampling simples)
- Estatísticas e gráficos (matriz de correlação, distribuições, boxplots, *heatmap* de ausentes, **PCA 2D/3D** com variância explicada).
- *Download* do dataset processado.

---

## Endpoints

| Método | Rota            | Descrição                                                |
|:------:|-----------------|----------------------------------------------------------|
| GET    | `/health`       | Verifica a saúde da API                                  |
| POST   | `/api/upload`   | Upload do dataset (`.csv`, `.txt`, `.tsv`)               |
| POST   | `/api/analyze`  | Análise inicial com base na coluna **target**            |
| POST   | `/api/process`  | Pré-processamento completo com parâmetros configuráveis  |
| POST   | `/api/statistics` | Estatísticas (describe, gráficos, profiling* minimal) |
| POST   | `/api/pca`      | Gera **PCA 2D** (imagem base64)                          |
| POST   | `/api/pca3d`    | Gera **PCA 3D** (HTML Plotly embutido)                   |
| POST   | `/api/outliers` | Gráfico de outliers (imagem base64)                      |
| GET    | `/api/download` | *Download* do dataset processado (`processed_dataset.csv`) |
| POST   | `/api/clear`    | Limpa arquivos temporários e estado de sessão            |

\* O *profiling* via `ydata-profiling` é gerado no modo **minimal**, quando disponível.

---

## Esquemas de Requisição/Resposta

### 1) `POST /api/upload`
- **Form-Data**: `file` (obrigatório) — tipos suportados: `.csv`, `.txt`, `.tsv` (até 50–100 MB, ver limites).
- **Resposta (sucesso)**:
```json
{
  "success": true,
  "info": {
    "filename": "dataset.csv",
    "shape": [linhas, colunas],
    "columns": ["col1", "col2", "..."],
    "dtypes": {"col1": "float64", "col2": "object"},
    "missing_values": {"col1": 0, "col2": 3},
    "unique_values": {"col1": 100, "col2": 5},
    "sample_values": {"col2": ["A","B","..."]},
    "duplicates": 0,
    "numeric_columns": ["..."],
    "categorical_columns": ["..."],
    "boolean_columns": ["..."],
    "datetime_columns": ["..."],
    "data_quality": {"completeness": 98.2, "uniqueness": 100.0, "consistency": 100},
    "encoding_used": "utf-8",
    "delimiter_used": ","
  }
}
````

### 2) `POST /api/analyze`

* **JSON**:

```json
{ "target_column": "classe" }
```

* **Resposta (sucesso)**:

```json
{
  "success": true,
  "analysis": {
    "target_column": "classe",
    "target_type": "object",
    "target_classes": {"A": 120, "B": 80},
    "target_balance_ratio": 0.67,
    "missing_target_values": 0,
    "total_missing_percentage": 1.5,
    "duplicates_count": 0,
    "numeric_columns_count": 10,
    "categorical_columns_count": 3,
    "boolean_columns_count": 1,
    "datetime_columns_count": 0,
    "dataset_shape": [200, 14],
    "data_quality": {"completeness": 98.5, "uniqueness": 100, "consistency": 100},
    "ml_readiness": {
      "target_quality": "good",
      "class_balance": "imbalanced",
      "data_completeness": 98.5,
      "recommendation": ["Considerar balanceamento de classes"]
    },
    "needs_balancing": true
  }
}
```

### 3) `POST /api/process`

* **JSON**:

```json
{
  "target_column": "classe",
  "config": {
    "remove_duplicates": true,
    "treat_missing": true,
    "encode_categories": true,
    "normalize_data": false,
    "select_features": true,
    "num_features_percent": 50,
    "balance_classes": false
  }
}
```

* **Resposta (sucesso)**:

```json
{
  "success": true,
  "processing_stats": {
    "original_rows": 200,
    "original_columns": 14,
    "missing_values_treated": 12,
    "duplicates_removed": 0,
    "outliers_removed": 0,
    "categorical_encoded": 30,
    "boolean_encoded": 10,
    "normalized_columns": 0,
    "balanced_samples": 0,
    "final_rows": 200,
    "final_columns": 8,
    "features_selected": 7,
    "target_mapping": {"A": "0", "B": "1"},
    "categorical_mappings": {"col_cat": {"x": "0", "y": "1"}},
    "improvement_ratio": 1.0
  },
  "final_shape": [200, 8],
  "final_columns": ["f1","f2","...","classe"],
  "data_quality_improvement": {
    "completeness_before": 98.5,
    "completeness_after": 100,
    "duplicates_removed": 0,
    "features_optimized": 6
  }
}
```

### 4) `POST /api/statistics`

* **JSON**:

```json
{ "target_column": "classe" }
```

* **Resposta (sucesso)**: contém **imagens base64** e, quando possível, `profiling_report` (HTML minimal).

```json
{
  "success": true,
  "plots": {
    "correlation_matrix": "data:image/png;base64,...",
    "distribution_plots": "data:image/png;base64,...",
    "boxplots": "data:image/png;base64,...",
    "target_distribution": "data:image/png;base64,...",
    "missing_values_heatmap": "data:image/png;base64,..."
  },
  "statistics_summary": {
    "total_features": 14,
    "numeric_features": 10,
    "categorical_features": 3,
    "data_quality_score": 98.5
  },
  "describe_table": { "...": "..." },
  "profiling_report": "<html>...</html>"
}
```

### 5) `POST /api/pca`

* **JSON**:

```json
{ "target_column": "classe" }
```

* **Resposta (sucesso)**:

```json
{
  "success": true,
  "pca_plot": "data:image/png;base64,...",
  "explained_variance": [0.523, 0.287],
  "cumulative_variance": 0.81,
  "pca_interpretation": {
    "pc1_description": "PC1 explica 52.3% da variância",
    "pc2_description": "PC2 explica 28.7% da variância",
    "recommendation": "Use as componentes para visualizar separação das classes"
  }
}
```

### 6) `POST /api/pca3d`

* **JSON**:

```json
{ "target_column": "classe" }
```

* **Resposta (sucesso)**:

```json
{
  "success": true,
  "pca_plot": "<div>...</div>",
  "explained_variance": [0.41, 0.23, 0.12],
  "cumulative_variance": 0.76
}
```

### 7) `POST /api/outliers`

* **Resposta (sucesso)**:

```json
{
  "success": true,
  "outliers_plot": "data:image/png;base64,...",
  "outliers_count": 18,
  "outliers_percentage": 2.1,
  "detection_method": "Isolation Forest + Z-Score",
  "recommendation": "Outliers detectados podem ser tratados ou removidos"
}
```

### 8) `GET /api/download`

* Retorna `processed_dataset.csv` (se disponível).

### 9) `POST /api/clear`

* Limpa arquivos e estado (upload/processados).

---

## Variáveis de Ambiente

| Variável          | Padrão                                              | Descrição                                  |
| ----------------- | --------------------------------------------------- | ------------------------------------------ |
| `PORT`            | `7860`                                              | Porta de execução (exigida pelo HF Spaces) |
| `ALLOWED_ORIGINS` | `https://<usuario>.github.io,http://localhost:3000` | Origens autorizadas para CORS              |
| `SECRET_KEY`      | `cleansight-secret`                                 | Chave Flask                                |
| `MPLBACKEND`      | `Agg`                                               | Backend para renderização headless         |

---

## Como Executar Localmente

```bash
# Clonar seu repositório (ex.: branch backend do Space)
git clone https://huggingface.co/spaces/<usuario>/CleanSight-API
cd CleanSight-API

# Python 3.11+
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
export FLASK_ENV=development
export PORT=5000
export ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"

# Executar (dev)
python -c "from src.main import create_app; app=create_app(); app.run(host='0.0.0.0', port=5000, debug=True)"
# ou com gunicorn (prod-like)
gunicorn -w 2 -k gthread -b 0.0.0.0:5000 src.main:app --timeout 120
```

---

## Implantação no Hugging Face (SDK: Docker)

1. Crie um **Space**:

   * **Name**: `CleanSight-API`
   * **SDK**: `Docker`

2. Estrutura mínima:

```
CleanSight-API/
├─ src/
│  ├─ routes/
│  │  └─ preprocessing_enhanced.py
│  └─ main.py
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

3. *Push*:

```bash
git add .
git commit -m "Initial CleanSight API (Flask + Docker + CORS)"
git push
```

4. Após o build, teste:

```bash
curl -s https://<usuario>-cleansight-api.hf.space/health
```

---

## Exemplos (cURL)

> Substitua `<BASE>` por `https://<usuario>-cleansight-api.hf.space`.

```bash
# Health
curl -s <BASE>/health

# Upload
curl -s -X POST -F "file=@./meu_dataset.csv" <BASE>/api/upload

# Analyze (target)
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"target_column":"classe"}' <BASE>/api/analyze

# Process (pipeline completo)
curl -s -X POST -H "Content-Type: application/json" \
  -d '{
        "target_column": "classe",
        "config": {
          "remove_duplicates": true,
          "treat_missing": true,
          "encode_categories": true,
          "normalize_data": false,
          "select_features": true,
          "num_features_percent": 50,
          "balance_classes": false
        }
      }' \
  <BASE>/api/process

# Estatísticas + gráficos
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"target_column":"classe"}' <BASE>/api/statistics

# PCA 2D
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"target_column":"classe"}' <BASE>/api/pca

# PCA 3D
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"target_column":"classe"}' <BASE>/api/pca3d

# Outliers
curl -s -X POST <BASE>/api/outliers

# Download do processado
curl -L -o processed_dataset.csv <BASE>/api/download

# Limpar sessão
curl -s -X POST <BASE>/api/clear
```

---

## Limites e Observações

* **Tamanho de upload**: até **100 MB** (ajustável por `MAX_CONTENT_LENGTH` e limites do Space).
* **Armazenamento**: `/tmp` é **efêmero** (reinícios perdem arquivos).
* **Tempo de execução**: mantenha requisições < 120s (ajustável no `--timeout` do gunicorn).
* **Profiling**: `ydata-profiling` em modo minimal; pode ser desativado removendo a dependência.

---

## Resolução de Problemas

* **CORS bloqueado**: inclua seu domínio GitHub Pages em `ALLOWED_ORIGINS`.
  Ex.: `ALLOWED_ORIGINS="https://<usuario>.github.io,http://localhost:3000"`
* **Build lento/falha por dependências**: fixe versões no `requirements.txt` ou remova `ydata-profiling`.
* **Timeout em processamento pesado**: reduza tamanho do dataset, ative opções do pipeline de forma incremental ou aumente `--timeout`.
* **400/404 em rotas**: confirme o *path* (`/api/*`) e `target_column` existente no dataset.

---

## Licença

Este projeto é licenciado sob **MIT License**. Sinta-se livre para usar, modificar e compartilhar.

