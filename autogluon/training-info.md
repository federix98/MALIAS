# AutoGluon Report

## High Quality Setting

### Models Trained
SeasonalNaive, RecursiveTabular, DirectTabular, NPTS,  
DynamicOptimizedTheta, AutoETS, ChronosZeroShot [bolt_base],  
ChronosFineTuned [bolt_small], TemporalFusionTransformer,  
DeepAR, PatchTST, TiDE

### Training Time (5 time series)

**Runtimes per run:**  
1993.92 s, 1864.06 s, 2142.89 s, 1972.59 s, 2072.05 s  

**Total (5 runs):** 10045.5 s (~3h)  

### Estimation of the total runtime for the entire dataset
- For 288 time series (AZ dataset): **~6.7 days**

---


## Medium Quality Setting

### Models Trained
Naive, SeasonalNaive, RecursiveTabular, DirectTabular,  
ETS, Theta, Chronos[bolt_small],
TemporalFusionTransformer

### Training Time (5 time series)

**Runtimes per run:**  
205.39 s, 306.38 s, 255.73 s, 177.18 s, 431.34 s  

**Total (5 runs):** 1376.02 s (~23m)  

### Estimation of the total runtime for the entire dataset
- For 288 time series (AZ dataset): **~1 day (22 h)**

---

# FLAML Report

Local setting --> Train a model for each time series

### Forecasting (Test on Three Samples)

- **Series 0:** Approximately **2 hours** — from **15:06** to **17:12**
- **Series 1:** Approximately **1 hour** — from **17:12** to **18:21**
- **Series 2:** Approximately **2 hours** — from **18:21** to **20:26**

### Estimation of the total runtime for the entire dataset
- For 288 time series (AZ dataset): **~20 days**