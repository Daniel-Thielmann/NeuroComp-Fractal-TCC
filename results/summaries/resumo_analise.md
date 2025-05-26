# Resultados da Análise de Métodos de EEG

## Resumo de Desempenho dos Métodos

| Método         | Acurácia | Média Prob. Correta | Amostras | Classes          |
| -------------- | -------- | ------------------- | -------- | ---------------- |
| Fractal        | 0.6303   | 0.5958              | 944      | {1: 472, 2: 472} |
| LogPower       | 0.7013   | 0.6913              | 944      | {1: 472, 2: 472} |
| CSP_Fractal    | 0.7500   | 0.7046              | 864      | {1: 432, 2: 432} |
| CSP_LogPower   | 0.7130   | 0.6556              | 864      | {1: 432, 2: 432} |
| FBCSP_Fractal  | 0.7303   | 0.7295              | 864      | {1: 432, 2: 432} |
| FBCSP_LogPower | 0.6979   | 0.6880              | 864      | {1: 432, 2: 432} |

## Testes Estatísticos (Wilcoxon)

| Comparação                      | Estatística | P-valor    | Conclusão               |
| ------------------------------- | ----------- | ---------- | ----------------------- |
| Fractal vs LogPower             | 290584.0000 | 2.1023e-13 | Diferença significativa |
| CSP_Fractal vs CSP_LogPower     | 247817.0000 | 1.7341e-05 | Diferença significativa |
| FBCSP_Fractal vs FBCSP_LogPower | 226671.0000 | 6.2893e-10 | Diferença significativa |

## Valores de Kappa por Sujeito e Método

| Sujeito | Fractal | LogPower | CSP_Fractal | CSP_LogPower | FBCSP_Fractal | FBCSP_LogPower |
| ------- | ------- | -------- | ----------- | ------------ | ------------- | -------------- |
| P01     | 0.2167  | 0.2667   | 0.7500      | 0.5667       | 0.4500        | 0.2667         |
| P02     | 0.3833  | 0.4667   | 0.9667      | 0.5167       | 0.6333        | 0.4667         |
| P03     | -0.0333 | 0.4667   | 0.2833      | 0.2500       | 0.3500        | 0.4667         |
| P04     | 0.4000  | 0.3833   | 0.4500      | 0.2167       | 0.6500        | 0.3833         |
| P05     | 0.2667  | 0.4167   | 0.5167      | 0.4167       | 0.2833        | 0.4167         |
| P06     | -0.2167 | 0.4000   | 0.4500      | 0.1833       | 0.5167        | 0.4000         |
| P07     | 0.3167  | 0.3000   | 0.4833      | 0.5500       | 0.4667        | 0.3000         |
| P08     | 0.5167  | 0.4833   | 0.3833      | 0.5333       | 0.5333        | 0.4833         |
| P09     | 0.3833  | 0.3000   | 0.3167      | 0.4000       | 0.4000        | 0.3000         |
| Média   | 0.2481  | 0.3870   | 0.5111      | 0.4037       | 0.4759        | 0.3870         |

## Conclusões Principais

### Melhor Método Geral

O método com melhor desempenho geral foi **CSP_Fractal** com um kappa médio de **0.5111**.

### Melhor Desempenho por Método

- **Fractal**: Melhor desempenho com o sujeito P08 (kappa = 0.5167)
- **LogPower**: Melhor desempenho com o sujeito P08 (kappa = 0.4833)
- **CSP_Fractal**: Melhor desempenho com o sujeito P02 (kappa = 0.9667)
- **CSP_LogPower**: Melhor desempenho com o sujeito P01 (kappa = 0.5667)
- **FBCSP_Fractal**: Melhor desempenho com o sujeito P04 (kappa = 0.6500)
- **FBCSP_LogPower**: Melhor desempenho com o sujeito P08 (kappa = 0.4833)

### Melhor Método por Sujeito

- **P01**: CSP_Fractal (kappa = 0.7500)
- **P02**: CSP_Fractal (kappa = 0.9667)
- **P03**: LogPower (kappa = 0.4667)
- **P04**: FBCSP_Fractal (kappa = 0.6500)
- **P05**: CSP_Fractal (kappa = 0.5167)
- **P06**: FBCSP_Fractal (kappa = 0.5167)
- **P07**: CSP_LogPower (kappa = 0.5500)
- **P08**: CSP_LogPower (kappa = 0.5333)
- **P09**: CSP_LogPower (kappa = 0.4000)

## Observações Gerais

1. Os métodos baseados em CSP (Common Spatial Patterns) apresentam melhor desempenho geral.
2. O CSP_Fractal obteve o maior valor de kappa para o sujeito P02 (0.9667), indicando excelente capacidade de classificação para este indivíduo.
3. Todos os testes de Wilcoxon mostraram diferenças estatisticamente significativas entre os métodos comparados.
4. A variabilidade de desempenho entre sujeitos sugere que a personalização do método para cada indivíduo pode ser benéfica.
5. Os métodos baseados em características fractais (CSP_Fractal e FBCSP_Fractal) tendem a superar suas contrapartes baseadas em LogPower.
