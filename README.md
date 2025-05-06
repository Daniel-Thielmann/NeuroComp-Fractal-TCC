# 🧠 TCC - Análise de EEG com Dimensão Fractal de Higuchi e LogPower

Este projeto realiza uma análise computacional detalhada de sinais EEG durante tarefas de imaginação motora, utilizando dois métodos distintos de extração de características: **Dimensão Fractal de Higuchi** e **Potência Logarítmica (LogPower)**. O objetivo é avaliar comparativamente o desempenho desses métodos na discriminação entre atividades mentais de movimento da mão esquerda e da mão direita, aplicando classificadores LDA e validação estatística.

---

## 🎯 Objetivos

- Explorar a **Dimensão Fractal de Higuchi (HFD)** como ferramenta de extração de características para EEG.
- Comparar seu desempenho com o método tradicional de **LogPower**.
- Classificar os sinais utilizando **LDA** com validação cruzada.
- Aplicar o **teste de Wilcoxon** para avaliar diferenças estatísticas significativas entre os métodos.
- Visualizar os resultados com gráficos analíticos.

---

## 📁 Estrutura do Projeto

```
EEG-TCC/
├── dataset/                      # Arquivos .mat dos sujeitos (parsed_P01T.mat etc.)
├── results/                      # Saídas dos classificadores
│   ├── Higuchi/
│   │   ├── Training/
│   │   └── Evaluate/
│   ├── LogPower/
│   │   ├── Training/
│   │   └── Evaluate/
│   └── higuchi_vs_logpower_comparison.csv
├── graphics/
│   ├── scripts/                  # Scripts para geração dos gráficos
│   └── results/                  # Gráficos salvos (.png)
├── methods/
│   ├── higuchi.py                # Implementação do método Higuchi
│   └── logpower.py               # Implementação do método LogPower
├── main.py                       # Script principal que executa o pipeline completo
└── README.md
```

---

## 📦 Dataset: WCCI 2020

Este projeto utiliza os dados do desafio **WCCI 2020**. O conjunto de dados contém sinais EEG registrados durante a realização de tarefas de imaginação motora (mão esquerda ou direita).

- **Formato dos arquivos**: `.mat`
- **Tarefas**: Imaginação de movimento da mão esquerda (classe 1) e direita (classe 2)
- **Canais**: 22 canais EEG
- **Duração das trials**: ~4 segundos

---

## 🧪 Pipeline de Execução

O pipeline executado pela `main.py` realiza as seguintes etapas:

1. **Leitura dos arquivos `.mat` para cada sujeito**
2. **Extração de características** com:
   - `HiguchiFractalEvolution` (DF)
   - `LogPowerEnhanced` (Potência Log)
3. **Treinamento e avaliação via validação cruzada 5-fold**
4. **Classificação com LDA (Linear Discriminant Analysis)**
5. **Geração de 40 arquivos CSV (Training + Evaluate)** com probabilidades
6. **Construção do CSV final comparando Higuchi vs LogPower**
7. **Cálculo das estatísticas descritivas**
8. **Teste de Wilcoxon para comparação estatística**
9. **Geração de 8 gráficos explicativos**

---

## 🖼 Gráficos Gerados

Local: `graphics/results/`

- `boxplot_higuchi_vs_logpower.png`
- `histogram_higuchi_vs_logpower.png`
- `accuracy_per_subject.png`
- `scatter_higuchi_vs_logpower.png`
- `wilcoxon_pvalue_plot.png`
- `heatmap_subject_vs_method.png`
- `roc_curve_comparison.png`
- `violinplot_higuchi_vs_logpower.png`
- `confusion_matrix_comparison.png`

> Todos os gráficos são gerados automaticamente pelo script `graphics/scripts/generate_all_graphs.py`.

---

## ▶️ Como Executar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Execute o pipeline completo

```bash
python main.py
```

### 3. Gere todos os gráficos

```bash
python graphics/scripts/generate_all_graphs.py
```

---

## 📊 Resultados Estatísticos

Após a execução, o teste de Wilcoxon aponta se a diferença entre os métodos é estatisticamente significativa.

**Exemplo de saída**:

```
=== Wilcoxon Test (40 CSVs combinados) ===
Statistic: 588424.0000
P-value : 0.0063
Conclusão: Diferença significativa entre os métodos
```

Isso confirma que **Higuchi superou o LogPower** na análise de imaginação motora com EEG.

---

## Autor

**Daniel Thielmann**
Curso de Engenharia Computacional  
Universidade Federal de Juiz de Fora (UFJF)

---

## Orientador

**Gabriel Souza**
Departamento de Ciência da Computação (UFJF)  
Universidade Federal de Juiz de Fora (UFJF)

---
