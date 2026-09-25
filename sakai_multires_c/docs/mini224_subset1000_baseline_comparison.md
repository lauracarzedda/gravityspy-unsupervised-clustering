# Confronto finale della baseline mini224 subset1000

## 1. Scopo e checkpoint

Questo documento confronta le otto configurazioni ufficiali della baseline
centrale mini224 subset1000 pool14: quattro profili di rappresentazione,
ciascuno con `k=22` e `k=36`. I valori provengono esclusivamente dagli
artefatti ufficiali rivalidati al checkpoint Git
`2a1a2ca3c8d62978cf581d14827a843b1f5039e7` sul branch
`baseline-mini224-subset1000-pool14`.

Il confronto non introduce nuovi fit, trasformazioni, assegnazioni o
metriche. Le label Gravity Spy entrano soltanto nella valutazione a
posteriori delle assegnazioni congelate.

## 2. Subset congelato

Il subset mini224 contiene 1000 eventi: il prefisso di 800 righe dello split
train ufficiale e il prefisso di 200 righe dello split validation ufficiale,
entrambi derivati dagli split immutabili con seed 123. La selezione del
subset è label-blind: le label non sono state usate per scegliere eventi,
estrarre feature, standardizzare, eseguire PCA o fare clustering.

Il subset, il manifest, gli split, la row map e i tensori sono congelati e
comuni a tutte le configurazioni qui confrontate.

## 3. Protocollo comune

- 800 eventi train e 200 eventi validation, per un totale di 1000;
- prefissi degli split ufficiali seed 123, senza nuovo campionamento;
- k-means++ deterministico con seed 123 e massimo 300 iterazioni;
- `k=22`, controllo coerente con le 22 classi Gravity Spy, e `k=36`;
- fit k-means soltanto sul train;
- validation assegnata ai centroidi appresi sul train;
- label escluse da rappresentazione, standardizzazione, PCA e clustering;
- purity pesata, NMI con normalizzazione aritmetica, ARI, homogeneity e
  completeness calcolate soltanto a posteriori;
- nessuna augmentation e nessun whitening.

La validation comprende soltanto 200 eventi. I gap train-validation e le
variazioni tra configurazioni possono quindi essere sensibili alla ridotta
numerosità e non vanno interpretati come stime precise della generalizzazione.

## 4. Rappresentazione pool14

Ogni evento centrale è un tensore `uint8` channel-major `4×224×224`.
L'average pooling non sovrapposto `14×14`, con stride 14 e senza padding,
produce una griglia `4×16×16`, cioè 1024 feature per evento. La
standardizzazione è appresa soltanto sul train e applicata alla validation
senza refit.

## 5. Profili confrontati

| Profilo | PCA | Dimensione | Frazione mantenuta | Compressione | Soglia | Varianza cumulativa |
|---|---|---:|---:|---:|---:|---:|
| no_pca | no | 1024 | 1.000000 | 1.000000x | n/a | n/a |
| pca90 | sì | 15 | 0.014648 | 68.266667x | 0.90 | 0.901284 |
| pca95 | sì | 38 | 0.037109 | 26.947368x | 0.95 | 0.950058 |
| pca99 | sì | 168 | 0.164062 | 6.095238x | 0.99 | 0.990021 |

Gli score PCA sono usati direttamente, senza whitening o standardizzazione
post-PCA. Il fit PCA è train-only e la validation è trasformata senza refit.

## 6. Regime PCA rank-deficient

Con 800 righe train centrate e 1024 feature, il rango teorico massimo è 799.
I tre profili PCA hanno rango numerico 799, nullspace di dimensione 225 e
tolleranza di rango `1.2022240890916001e-07`.

Si applica la Policy C: modello completo `1024×1024`; parte informativa
0–167 bitwise invariata; score conservati bitwise; nullspace validato tramite
il proiettore; nessuna modifica al core PCA. I profili PCA-90 e PCA-95 sono
prefissi per riga degli score PCA-99.

## 7. Risultati train

| Profilo | k | Dim. | Purity | NMI | ARI | Homogeneity | Completeness | Iter. | Inerzia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_pca | 22 | 1024 | 0.513750 | 0.491261 | 0.132644 | 0.405356 | 0.623368 | 10 | 133876.143676 |
| no_pca | 36 | 1024 | 0.588750 | 0.566722 | 0.236241 | 0.527033 | 0.612876 | 27 | 110085.953168 |
| pca90 | 22 | 15 | 0.482500 | 0.452285 | 0.115735 | 0.365904 | 0.592056 | 15 | 60030.318758 |
| pca90 | 36 | 15 | 0.566250 | 0.541449 | 0.210338 | 0.492215 | 0.601627 | 15 | 39851.512228 |
| pca95 | 22 | 38 | 0.465000 | 0.436790 | 0.099384 | 0.347491 | 0.587862 | 16 | 90459.981609 |
| pca95 | 36 | 38 | 0.553750 | 0.550311 | 0.283839 | 0.476257 | 0.651634 | 12 | 67555.009171 |
| pca99 | 22 | 168 | 0.528750 | 0.481110 | 0.156145 | 0.366789 | 0.698963 | 9 | 124681.262037 |
| pca99 | 36 | 168 | 0.600000 | 0.580491 | 0.333053 | 0.532467 | 0.638037 | 11 | 98892.755274 |

## 8. Risultati validation

| Profilo | k | Dim. | Purity | NMI | ARI | Homogeneity | Completeness | Iter. | Inerzia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_pca | 22 | 1024 | 0.515000 | 0.572115 | 0.160154 | 0.465855 | 0.741174 | 10 | 65767.464589 |
| no_pca | 36 | 1024 | 0.610000 | 0.652694 | 0.287555 | 0.593516 | 0.724981 | 27 | 62147.969992 |
| pca90 | 22 | 15 | 0.445000 | 0.503946 | 0.121012 | 0.396658 | 0.690789 | 15 | 33558.406699 |
| pca90 | 36 | 15 | 0.590000 | 0.631946 | 0.259294 | 0.559530 | 0.725893 | 15 | 31331.688612 |
| pca95 | 22 | 38 | 0.430000 | 0.493383 | 0.107479 | 0.380704 | 0.700803 | 16 | 43482.887271 |
| pca95 | 36 | 38 | 0.540000 | 0.609303 | 0.253794 | 0.505650 | 0.766412 | 12 | 41652.165634 |
| pca99 | 22 | 168 | 0.520000 | 0.544503 | 0.148290 | 0.407172 | 0.821619 | 9 | 58238.965395 |
| pca99 | 36 | 168 | 0.580000 | 0.622844 | 0.275920 | 0.551190 | 0.715911 | 11 | 55154.485181 |

## 9. Gap train-validation

Il gap è definito come `validation - train`; non è un punteggio aggregato.

| Profilo | k | Δ purity | Δ NMI | Δ ARI | Δ homogeneity | Δ completeness |
|---|---:|---:|---:|---:|---:|---:|
| no_pca | 22 | 0.001250 | 0.080855 | 0.027510 | 0.060500 | 0.117806 |
| no_pca | 36 | 0.021250 | 0.085972 | 0.051314 | 0.066483 | 0.112105 |
| pca90 | 22 | -0.037500 | 0.051660 | 0.005277 | 0.030754 | 0.098733 |
| pca90 | 36 | 0.023750 | 0.090497 | 0.048956 | 0.067315 | 0.124266 |
| pca95 | 22 | -0.035000 | 0.056592 | 0.008095 | 0.033213 | 0.112941 |
| pca95 | 36 | -0.013750 | 0.058993 | -0.030045 | 0.029392 | 0.114778 |
| pca99 | 22 | -0.008750 | 0.063393 | -0.007855 | 0.040383 | 0.122656 |
| pca99 | 36 | -0.020000 | 0.042353 | -0.057133 | 0.018724 | 0.077873 |

La prevalenza di gap positivi per NMI, homogeneity e completeness non implica
un miglioramento causale sulla validation: con 200 eventi, composizione e
sbilanciamento possono produrre variazioni visibili. In particolare, i gap
negativi di purity e ARI in alcuni profili non vanno sovrainterpretati.

## 10. Passaggio da k=22 a k=36

Le differenze sono `k36 - k22` nello stesso profilo e split.

| Profilo | Split | Δ purity | Δ NMI | Δ ARI | Δ homogeneity | Δ completeness |
|---|---|---:|---:|---:|---:|---:|
| no_pca | train | 0.075000 | 0.075462 | 0.103597 | 0.121678 | -0.010492 |
| no_pca | validation | 0.095000 | 0.080579 | 0.127401 | 0.127661 | -0.016194 |
| pca90 | train | 0.083750 | 0.089164 | 0.094604 | 0.126311 | 0.009571 |
| pca90 | validation | 0.145000 | 0.128001 | 0.138282 | 0.162872 | 0.035104 |
| pca95 | train | 0.088750 | 0.113520 | 0.184455 | 0.128766 | 0.063772 |
| pca95 | validation | 0.110000 | 0.115921 | 0.146315 | 0.124946 | 0.065608 |
| pca99 | train | 0.071250 | 0.099382 | 0.176908 | 0.165678 | -0.060926 |
| pca99 | validation | 0.060000 | 0.078341 | 0.127630 | 0.144019 | -0.105708 |

Purity, NMI, ARI e homogeneity aumentano passando a `k=36` per tutti i
profili e per entrambi gli split. La completeness diminuisce per no-PCA e
PCA-99, ma aumenta per PCA-90 e PCA-95. Questo comportamento, da solo, non
seleziona un modello.

## 11. Delta PCA rispetto a no-PCA

Ogni differenza è `PCA - no-PCA` per lo stesso `k` e split.

| Profilo | k | Split | Δ purity | Δ NMI | Δ ARI | Δ homogeneity | Δ completeness |
|---|---:|---|---:|---:|---:|---:|---:|
| pca90 | 22 | train | -0.031250 | -0.038976 | -0.016909 | -0.039452 | -0.031312 |
| pca90 | 22 | validation | -0.070000 | -0.068170 | -0.039142 | -0.069197 | -0.050385 |
| pca90 | 36 | train | -0.022500 | -0.025273 | -0.025902 | -0.034818 | -0.011249 |
| pca90 | 36 | validation | -0.020000 | -0.020748 | -0.028261 | -0.033986 | 0.000913 |
| pca95 | 22 | train | -0.048750 | -0.054470 | -0.033259 | -0.057865 | -0.035506 |
| pca95 | 22 | validation | -0.085000 | -0.078732 | -0.052675 | -0.085151 | -0.040371 |
| pca95 | 36 | train | -0.035000 | -0.016412 | 0.047599 | -0.050776 | 0.038757 |
| pca95 | 36 | validation | -0.070000 | -0.043391 | -0.033761 | -0.087867 | 0.041431 |
| pca99 | 22 | train | 0.015000 | -0.010151 | 0.023501 | -0.038567 | 0.075595 |
| pca99 | 22 | validation | 0.005000 | -0.027613 | -0.011864 | -0.058684 | 0.080445 |
| pca99 | 36 | train | 0.011250 | 0.013769 | 0.096812 | 0.005433 | 0.025161 |
| pca99 | 36 | validation | -0.030000 | -0.029850 | -0.011635 | -0.042326 | -0.009070 |

Il CSV dei delta conserva anche inerzia e statistiche del cluster maggiore.
Le inerzie appartengono però a spazi di dimensionalità diversa: sono
descrittive e non vengono usate per confrontare la qualità cross-profile o
per selezionare una configurazione.

## 12. Dimensione dei cluster train

| Profilo | k | Iter. | Min train | Max train | Max/train |
|---|---:|---:|---:|---:|---:|
| no_pca | 22 | 10 | 1 | 413 | 0.516250 |
| no_pca | 36 | 27 | 1 | 291 | 0.363750 |
| pca90 | 22 | 15 | 1 | 426 | 0.532500 |
| pca90 | 36 | 15 | 1 | 318 | 0.397500 |
| pca95 | 22 | 16 | 1 | 440 | 0.550000 |
| pca95 | 36 | 12 | 1 | 327 | 0.408750 |
| pca99 | 22 | 9 | 1 | 527 | 0.658750 |
| pca99 | 36 | 11 | 1 | 271 | 0.338750 |

Tutte le configurazioni hanno almeno un cluster train di una sola
osservazione. PCA-99/k22 presenta il cluster train massimo più grande, pari
al 65.8750% del train; PCA-99/k36 presenta invece la frazione massima più
bassa, 33.8750%. Queste statistiche descrivono il bilanciamento e non
sostituiscono le metriche esterne.

## 13. Fronte di Pareto sulle metriche validation

Una configurazione A domina B se A è almeno pari a B su tutte le cinque
metriche validation ed è strettamente migliore su almeno una. Non sono usati
media delle metriche, somma dei rank, pesi o inerzia.

Configurazioni non dominate:

- no-PCA/k36;
- PCA-90/k36;
- PCA-95/k36;
- PCA-99/k22.

Configurazioni dominate e un dominatore verificato:

| Configurazione dominata | Dominatore |
|---|---|
| no-PCA/k22 | PCA-95/k36 |
| PCA-90/k22 | PCA-90/k36 |
| PCA-95/k22 | PCA-95/k36 |
| PCA-99/k36 | no-PCA/k36 |

## 14. Leader validation

| Metrica | Configurazione | Valore |
|---|---|---:|
| Purity | no-PCA/k36 | 0.610000 |
| NMI aritmetica | no-PCA/k36 | 0.652694 |
| ARI | no-PCA/k36 | 0.287555 |
| Homogeneity | no-PCA/k36 | 0.593516 |
| Completeness | PCA-99/k22 | 0.821619 |

No-PCA/k36 guida dunque purity, NMI, ARI e homogeneity validation.
PCA-99/k22 guida la completeness, ma usa `k=22`, ha il cluster train massimo
più grande e non è la configurazione primaria.

## 15. Interpretazione scientifica

No-PCA/k36 conserva tutte le 1024 feature e ottiene il migliore equilibrio
complessivo delle metriche validation: guida quattro metriche su cinque e
non introduce la riduzione PCA. È pertanto la baseline primaria del subset
mini224.

PCA-95/k36 usa 38 componenti, una compressione di circa 26.95x, e raggiunge
la maggiore completeness tra tutte le configurazioni `k=36` (0.766412).
Costituisce il compromesso PCA compatto, pur senza dominare no-PCA/k36.

PCA-90/k36 usa soltanto 15 componenti, con compressione di circa 68.27x, e
mantiene metriche sorprendentemente vicine alla baseline: rispetto a
no-PCA/k36 perde 0.020000 in purity, 0.020748 in NMI, 0.028261 in ARI e
0.033986 in homogeneity, mentre la completeness differisce di +0.000913.
È l'ablation di compressione aggressiva.

PCA-99/k36 usa 168 componenti ed è l'ablation PCA ad alta conservazione.
Sui dati validation è dominata metricamente da no-PCA/k36, ma resta utile
per coerenza metodologica con le baseline mini75 e mini112.

PCA-99/k22 mostra la migliore completeness validation, ma la forte
concentrazione del cluster maggiore e l'uso di `k=22` impediscono di
considerarla una vincitrice universale.

## 16. Ruoli metodologici

Non emerge un vincitore universale. La configurazione no-PCA con `k36` è la
baseline primaria subset224; `pca99/k36` rappresenta l'ablation ad alta
conservazione della varianza, `pca95/k36` il compromesso compatto e `pca90`
la compressione aggressiva. `k22` resta il controllo legato al numero di
classi.

Questi ruoli derivano congiuntamente da protocollo, metriche validation,
dimensionalità, bilanciamento e confronto matched disponibile; non dalla
massimizzazione isolata di una metrica, da medie arbitrarie o da somme di
rank. `k=22` resta il controllo coerente con il numero di classi Gravity Spy.

## 17. Limiti

- La validation contiene 200 eventi: gap e differenze possono essere
  sensibili alla composizione del subset.
- Le metriche esterne dipendono dalle label e sono usate soltanto a
  posteriori; non misurano da sole la struttura fisica dei glitch.
- Le classi sono sbilanciate e ogni configurazione contiene almeno un
  cluster train unitario.
- Un singolo seed congelato garantisce confrontabilità, ma non quantifica la
  variabilità fra inizializzazioni.
- Le inerzie non sono confrontabili come qualità fra spazi di dimensionalità
  diversa.
- Mini224 usa 1000 eventi; le baseline congelate mini75 e mini112 ne usano
  8583. Le differenze osservate non possono essere attribuite causalmente
  alla risoluzione.
- Il confronto osservazionale fra risoluzioni è documentato separatamente
  mediante il matched-subset sugli stessi 1000 eventi per mini75, mini112 e
  mini224.

## 18. File sorgente e SHA-256

Tutti i percorsi sono relativi alla radice del repository.

| Profilo | Tipo | File | SHA-256 |
|---|---|---|---|
| no_pca/k22 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/label_evaluation_seed123/k22/metrics.csv` | `416e04802967c4a133d2776660d8d9a4d7c73e2a7774a7b364db17b39a876bf3` |
| no_pca/k22 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/kmeans_seed123/k22/summary.txt` | `74acfa6349b431cfa0be23444354a0b9260acf5f0c4008e74b07bd4a15055c05` |
| no_pca/k36 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/label_evaluation_seed123/k36/metrics.csv` | `2c349c6d250ae54ef671749a90c75aa43d73979c3e3b88de04f01457a130526c` |
| no_pca/k36 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/kmeans_seed123/k36/summary.txt` | `6e2bf38b9250cdf98c4452cc9a5d88c2499e44a5e31d417adef1b568cebe7101` |
| pca90 | specifica PCA | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca90/pca_spec.txt` | `389d9527769191d9313cce58ce02b6b61249ead552f457000f37be1405eb78cd` |
| pca90/k22 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca90/label_evaluation_seed123/k22/metrics.csv` | `74a74d2ce0b10ba47e0db8ec385aa6989bdbbd936b55acb8f38d6e46b531c2fb` |
| pca90/k22 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca90/kmeans_seed123/k22/summary.txt` | `7260244c1266221e3c6a09fd62f8bb6388cf6164dd789fd8bf006ead18c625cd` |
| pca90/k36 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca90/label_evaluation_seed123/k36/metrics.csv` | `e050a099cdef0ffbb1bbb990ea7d0f5bd4ed214ecef2386044a31f2a3020e504` |
| pca90/k36 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca90/kmeans_seed123/k36/summary.txt` | `e68c1a2783ef4cc19148a16c4cbd05485841975d9a30510029782e199d591a99` |
| pca95 | specifica PCA | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca95/pca_spec.txt` | `77e3b79cd499605d52b8faa781ee5339aea9c022c1c795f957c08414c1c0a52b` |
| pca95/k22 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca95/label_evaluation_seed123/k22/metrics.csv` | `4b98ec6790e2e718a852f8d4399a8be4f0e11e92b162900a675f8a75368e2a64` |
| pca95/k22 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca95/kmeans_seed123/k22/summary.txt` | `33a1ba072e1fe680ecca7bdd7b14023146256d92b11639a5921fc123969e97ab` |
| pca95/k36 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca95/label_evaluation_seed123/k36/metrics.csv` | `f95b08bc588c525a71356dd5db23182aa1441d44d2c84c732b37285610538db9` |
| pca95/k36 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca95/kmeans_seed123/k36/summary.txt` | `9027a59b1782d71b1e1842fb637fceff503272c1ae964ae9d8a5852d5ad7abfe` |
| pca99 | specifica PCA | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca99/pca_spec.txt` | `858d8b878e145e175f42616cf8ea550dd2b682979ce3d806c0a46d2af236fc4b` |
| pca99/k22 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca99/label_evaluation_seed123/k22/metrics.csv` | `7a3cd81efb3bccfcbfd04f4bba491d05481ccaa41dd4de1482fe4ceca76d498b` |
| pca99/k22 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca99/kmeans_seed123/k22/summary.txt` | `465fa36cdd2d5a6dd2da35d71af8a2ebbf755b26a0c90b1f2f78fd47f204d01a` |
| pca99/k36 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca99/label_evaluation_seed123/k36/metrics.csv` | `0d191e96d9f799c408cf0905af0c3b10623023b441c936f525b58521f890f35a` |
| pca99/k36 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_pca99/kmeans_seed123/k36/summary.txt` | `aaebc690fe2d84e703f9f66f1a989f382c7283b8edc31fb43a38c93d93629299` |

## 19. Decisione finale

Non esiste un vincitore universale. No-PCA/k36 è la baseline primaria del
subset224 per il suo equilibrio validation e per l'assenza di riduzione.
PCA-99/k36, PCA-95/k36 e PCA-90 mantengono ruoli di ablation rispettivamente
ad alta conservazione, compatta e di compressione aggressiva. `k=22` è il
controllo legato al numero di classi. Questa decisione non usa inerzia
cross-profile né un ranking aggregato arbitrario.

## 20. Sintesi matched-subset 75/112/224

Il confronto matched fra risoluzioni riutilizza esattamente gli
stessi 1000 eventi, gli stessi prefissi train/validation e lo stesso
protocollo per mini75, mini112 e mini224. Questo disegno matched-subset
consente il confronto osservazionale fra risoluzioni senza
confonderlo con la diversa numerosità delle baseline complete mini75 e
mini112. I relativi artefatti sono documentati nel report matched-subset
dedicato.
