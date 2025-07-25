# TCC - Análise de EEG com Pipelines Fractal, LogPower, CSP e FBCSP

Este projeto realiza uma análise rigorosa de sinais EEG em tarefas de imaginação motora, utilizando múltiplos pipelines de extração de características e classificação, conforme padrões científicos e instruções do orientador.

---

## Objetivos

- Implementar e comparar os seguintes pipelines:
  - Fractal
  - LogPower
  - CSP + Fractal
  - CSP + LogPower
  - FBCSP + Fractal
  - FBCSP + LogPower
- Separar e salvar resultados por sujeito e por método, conforme padrão científico.
- Calcular estatísticas descritivas e realizar teste de Wilcoxon para comparação entre métodos.
- Garantir rigor científico e reprodutibilidade dos resultados.

---

## Estrutura do Projeto

```
EEG-TCC/
├── dataset/                # Dados brutos (.mat) dos sujeitos
│   ├── BCICIV2a/
│   ├── BCICIV2b/
│   └── wcci2020/
├── results/
│   ├── bciciv2a/
│   ├── bciciv2b/
│   ├── wcci2020/
│   │   └── <metodo>/
│   │       ├── evaluate/   # CSVs por sujeito (teste)
│   │       └── training/   # CSVs por sujeito (treino)
│   └── <metodo>_classification_results.csv # CSV geral do método
├── graphics/
│   ├── scripts/
│   └── results/
├── methods/
│   ├── features/
│   └── pipelines/
├── modules/
├── main.py                 # Script principal
└── README.md
```

---

## Datasets Utilizados

- **WCCI2020 (CBCIC):** 9 sujeitos, 80 trials/sujeito, 12 canais, 4096 amostras, 512Hz
- **BCICIV2a:** 9 sujeitos, 288 trials/sujeito, 22 canais, 1000 amostras, 100Hz
- **BCICIV2b:** 9 sujeitos, 288 trials/sujeito, 22 canais, 1000 amostras, 100Hz

---

## Pipelines Implementados

Todos os pipelines seguem rigorosamente as instruções:

- Filtro Chebyshev II 4-40Hz (ou bancos de filtros para FBCSP)
- Extração de características (HFD, LogPower)
- CSP e FBCSP quando especificado
- Seleção de características MIBIF (apenas FBCSP)
- Classificação com LDA
- Validação cruzada StratifiedKFold (5-fold, random_state=42)
- Sem normalização fora dos pipelines permitidos

---

## Resultados

- Resultados separados por dataset, método e sujeito
- 1 CSV por sujeito em evaluate e training, 1 CSV geral por método
- Estrutura: `results/<dataset>/<metodo>/evaluate/Pxx_evaluate.csv`, `results/<dataset>/<metodo>/training/Pxx_training.csv`, `results/<dataset>/<metodo>_classification_results.csv`
- Prints padronizados no terminal conforme instruções
- Teste de Wilcoxon e cálculo de kappa para comparação entre métodos

---

## Autor

Daniel Thielmann
Curso de Engenharia Computacional
Universidade Federal de Juiz de Fora (UFJF)

---

## Orientador

Gabriel Souza
Departamento de Ciência da Computação (UFJF)
Universidade Federal de Juiz de Fora (UFJF)

---

## Resultados Estatísticos

Após a execução dos pipelines, são calculados os valores de kappa e realizado o teste de Wilcoxon para cada dataset e par de métodos, permitindo identificar se as diferenças de desempenho entre os métodos são estatisticamente significativas e cientificamente relevantes.
