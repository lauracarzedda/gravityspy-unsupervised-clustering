# Confronto scientifico matched-subset1000 no-PCA

## 1. Scopo

Questo documento confronta le baseline no-PCA `mini75_pool5`,
`mini112_pool7` e `mini224_pool14` sullo stesso subset congelato di 1000
eventi. Per ciascun extractor sono confrontati `k=22` e `k=36`.

Il confronto usa soltanto artefatti ufficiali già esistenti e rivalidati. Non
introduce nuovi fit, trasformazioni, assegnazioni o metriche. Le label Gravity
Spy sono usate esclusivamente a posteriori.

## 2. Branch e checkpoint

- branch: `baseline-matched-subset1000-no-pca`;
- checkpoint: `60fcdeeab3cb416033c4901999b1cf1888435675`;
- commit: `Make matched validators downstream-aware`;
- stato iniziale: working tree pulita e staging vuoto.

## 3. Subset congelato e identità degli eventi

Tutte le configurazioni usano gli stessi 800 eventi train e 200 eventi
validation. I tre `train_rows.csv` sono byte-identici, così come i tre
`val_rows.csv`.

| Mappa | Righe | SHA-256 |
|---|---:|---|
| train | 800 | `fd5bb3c48f87e98d5ec8ed0e561d5fe077ab96a7effa0c7425a41a222cd4375c` |
| validation | 200 | `87a3c7fc715fe0f4a6158474e46f6903d5267a45741d338836e8fecd61df7ed8` |

L'identità è stata verificata per `row_index`, `split_pos`,
`manifest_index`, `event_index`, `event_id` e `split`. Gli insiemi train e
validation non hanno sovrapposizioni.

```text
ALL_THREE_MATCHED_ROW_MAPS_IDENTICAL=true
MATCHED_SUBSET_EVENT_SET_IDENTICAL=true
TRAIN_ROWS=800
VALIDATION_ROWS=200
OVERLAP=0
```

## 4. Protocollo comune

- subset `subset224_seed123_prefix_800_200`;
- standardizzazione con statistiche calcolate soltanto sul train, `ddof=0`;
- k-means++ deterministico con seed 123 e massimo 300 iterazioni;
- fit k-means soltanto sul train;
- validation assegnata ai centroidi train senza refit;
- `k=22`, controllo legato alle 22 classi note, e `k=36`;
- nessuna PCA, augmentation o whitening;
- label escluse da feature, standardizzazione e clustering;
- purity pesata, NMI aritmetica, ARI, homogeneity e completeness calcolate
  esclusivamente a posteriori.

Le sei configurazioni sono state rivalidate direttamente in situ. Per tutte
sono stati verificati checksum, refit train-only bitwise, assegnazioni,
centroidi, contingenze e metriche ricostruite.

## 5. Extractor

| Profilo | Tensore | Pooling | Griglia pooled | Feature |
|---|---|---|---|---:|
| mini75_pool5 | 4x75x75 | 5x5, stride 5 | 4x15x15 | 900 |
| mini112_pool7 | 4x112x112 | 7x7, stride 7 | 4x16x16 | 1024 |
| mini224_pool14 | 4x224x224 | 14x14, stride 14 | 4x16x16 | 1024 |

Risoluzione, kernel di pooling e griglia pooled sono accoppiati. Il confronto
è quindi fra configurazioni complete dell'extractor, non un esperimento
causale sulla sola risoluzione.

## 6. Risultati train

| Profilo | k | Feature | Purity | NMI | ARI | Homogeneity | Completeness | Iter. | Inerzia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mini75_pool5 | 22 | 900 | 0.498750 | 0.457405 | 0.152075 | 0.361809 | 0.621659 | 9 | 122864.886797 |
| mini75_pool5 | 36 | 900 | 0.593750 | 0.569771 | 0.291554 | 0.515789 | 0.636373 | 18 | 96579.171814 |
| mini112_pool7 | 22 | 1024 | 0.543750 | 0.528649 | 0.184627 | 0.449316 | 0.642002 | 12 | 142034.646948 |
| mini112_pool7 | 36 | 1024 | 0.706250 | 0.661359 | 0.386267 | 0.660254 | 0.662467 | 27 | 109683.408172 |
| mini224_pool14 | 22 | 1024 | 0.513750 | 0.491261 | 0.132644 | 0.405356 | 0.623368 | 10 | 133876.143676 |
| mini224_pool14 | 36 | 1024 | 0.588750 | 0.566722 | 0.236241 | 0.527033 | 0.612876 | 27 | 110085.953168 |

L'inerzia è riportata come proprietà del singolo fit. Non viene confrontata
scientificamente fra profili e non è usata per scegliere una configurazione.

## 7. Risultati validation

| Profilo | k | Feature | Purity | NMI | ARI | Homogeneity | Completeness | Iter. | Inerzia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mini75_pool5 | 22 | 900 | 0.490000 | 0.498456 | 0.139845 | 0.381439 | 0.719044 | 9 | 60527.315939 |
| mini75_pool5 | 36 | 900 | 0.630000 | 0.664632 | 0.371290 | 0.586393 | 0.766965 | 18 | 56309.946089 |
| mini112_pool7 | 22 | 1024 | 0.540000 | 0.597823 | 0.207170 | 0.500602 | 0.741906 | 12 | 66551.172916 |
| mini112_pool7 | 36 | 1024 | 0.705000 | 0.733478 | 0.424177 | 0.708425 | 0.760368 | 27 | 63208.567281 |
| mini224_pool14 | 22 | 1024 | 0.515000 | 0.572115 | 0.160154 | 0.465855 | 0.741174 | 10 | 65767.464589 |
| mini224_pool14 | 36 | 1024 | 0.610000 | 0.652694 | 0.287555 | 0.593516 | 0.724981 | 27 | 62147.969992 |

## 8. Gap train-validation

Il gap è `validation - train`. La validation contiene soltanto 200 eventi:
un aumento non dimostra una migliore generalizzazione causale e non è un
punteggio aggregato.

| Profilo | k | Delta purity | Delta NMI | Delta ARI | Delta homogeneity | Delta completeness |
|---|---:|---:|---:|---:|---:|---:|
| mini75_pool5 | 22 | -0.008750 | 0.041051 | -0.012230 | 0.019630 | 0.097385 |
| mini75_pool5 | 36 | 0.036250 | 0.094862 | 0.079736 | 0.070604 | 0.130593 |
| mini112_pool7 | 22 | -0.003750 | 0.069174 | 0.022544 | 0.051286 | 0.099903 |
| mini112_pool7 | 36 | -0.001250 | 0.072119 | 0.037910 | 0.048171 | 0.097901 |
| mini224_pool14 | 22 | 0.001250 | 0.080855 | 0.027510 | 0.060500 | 0.117806 |
| mini224_pool14 | 36 | 0.021250 | 0.085972 | 0.051314 | 0.066483 | 0.112105 |

## 9. Passaggio da k=22 a k=36

Le differenze metriche sono `k36 - k22`. La riduzione del cluster massimo è
`max(k22) - max(k36)`; la variazione della frazione è
`fraction(k36) - fraction(k22)`. Le ultime due colonne descrivono i cluster
train e sono quindi uguali nelle righe train e validation dello stesso
profilo.

| Profilo | Split | Delta purity | Delta NMI | Delta ARI | Delta homogeneity | Delta completeness | Riduzione max train | Delta max/train |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mini75_pool5 | train | 0.095000 | 0.112365 | 0.139479 | 0.153980 | 0.014714 | 194 | -0.242500 |
| mini75_pool5 | validation | 0.140000 | 0.166176 | 0.231445 | 0.204954 | 0.047921 | 194 | -0.242500 |
| mini112_pool7 | train | 0.162500 | 0.132710 | 0.201641 | 0.210938 | 0.020465 | 191 | -0.238750 |
| mini112_pool7 | validation | 0.165000 | 0.135656 | 0.217007 | 0.207823 | 0.018462 | 191 | -0.238750 |
| mini224_pool14 | train | 0.075000 | 0.075462 | 0.103597 | 0.121678 | -0.010492 | 122 | -0.152500 |
| mini224_pool14 | validation | 0.095000 | 0.080579 | 0.127401 | 0.127661 | -0.016194 | 122 | -0.152500 |

`k=36` migliora purity, NMI, ARI e homogeneity e riduce il cluster massimo
in tutti e tre gli extractor. La completeness aumenta per mini75 e mini112,
ma diminuisce per mini224. La diminuzione dell'inerzia non viene usata per
scegliere `k`.

## 10. Delta pairwise

Ogni delta è `left - right`. I valori a precisione completa sono in
`matched_subset1000_no_pca_pairwise_deltas.csv`.

| Confronto | k | Split | Delta purity | Delta NMI | Delta ARI | Delta homogeneity | Delta completeness | Delta max train | Delta max/train |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| mini112-minus-mini75 | 22 | train | 0.045000 | 0.071243 | 0.032552 | 0.087507 | 0.020343 | -110 | -0.137500 |
| mini112-minus-mini75 | 22 | validation | 0.050000 | 0.099366 | 0.067325 | 0.119163 | 0.022862 | -110 | -0.137500 |
| mini112-minus-mini75 | 36 | train | 0.112500 | 0.091588 | 0.094714 | 0.144465 | 0.026095 | -107 | -0.133750 |
| mini112-minus-mini75 | 36 | validation | 0.075000 | 0.068846 | 0.052887 | 0.122033 | -0.006597 | -107 | -0.133750 |
| mini224-minus-mini75 | 22 | train | 0.015000 | 0.033855 | -0.019431 | 0.043547 | 0.001709 | -78 | -0.097500 |
| mini224-minus-mini75 | 22 | validation | 0.025000 | 0.073659 | 0.020309 | 0.084417 | 0.022130 | -78 | -0.097500 |
| mini224-minus-mini75 | 36 | train | -0.005000 | -0.003048 | -0.055313 | 0.011244 | -0.023496 | -6 | -0.007500 |
| mini224-minus-mini75 | 36 | validation | -0.020000 | -0.011938 | -0.083735 | 0.007124 | -0.041985 | -6 | -0.007500 |
| mini224-minus-mini112 | 22 | train | -0.030000 | -0.037388 | -0.051983 | -0.043960 | -0.018634 | 32 | 0.040000 |
| mini224-minus-mini112 | 22 | validation | -0.025000 | -0.025707 | -0.047016 | -0.034747 | -0.000731 | 32 | 0.040000 |
| mini224-minus-mini112 | 36 | train | -0.117500 | -0.094636 | -0.150027 | -0.133221 | -0.049591 | 101 | 0.126250 |
| mini224-minus-mini112 | 36 | validation | -0.095000 | -0.080784 | -0.136622 | -0.114909 | -0.035387 | 101 | 0.126250 |

## 11. Bilanciamento dei cluster

| Profilo | k | Min train | Max train | Max/800 |
|---|---:|---:|---:|---:|
| mini75_pool5 | 22 | 1 | 491 | 0.613750 |
| mini75_pool5 | 36 | 1 | 297 | 0.371250 |
| mini112_pool7 | 22 | 1 | 381 | 0.476250 |
| mini112_pool7 | 36 | 1 | 190 | 0.237500 |
| mini224_pool14 | 22 | 1 | 413 | 0.516250 |
| mini224_pool14 | 36 | 1 | 291 | 0.363750 |

Fra le configurazioni `k=36`, mini112 ha il cluster massimo train più
piccolo: 190/800, contro 297/800 per mini75 e 291/800 per mini224. Tutte le
configurazioni hanno almeno un cluster train unitario.

## 12. Fronte di Pareto validation

La dominanza usa esclusivamente le cinque metriche validation. A domina B se
è almeno pari su tutte e cinque ed è strettamente migliore su almeno una.
Non sono usati medie, pesi, somme di rank o inerzia.

Configurazioni non dominate:

- `mini75_pool5/k36`;
- `mini112_pool7/k36`.

Configurazioni dominate e un dominatore verificato:

| Configurazione dominata | Dominatore |
|---|---|
| mini75_pool5/k22 | mini112_pool7/k22 |
| mini112_pool7/k22 | mini112_pool7/k36 |
| mini224_pool14/k22 | mini112_pool7/k22 |
| mini224_pool14/k36 | mini112_pool7/k36 |

## 13. Leader validation

| Metrica | Configurazione | Valore |
|---|---|---:|
| Purity | mini112_pool7/k36 | 0.705000 |
| NMI aritmetica | mini112_pool7/k36 | 0.733478 |
| ARI | mini112_pool7/k36 | 0.424177 |
| Homogeneity | mini112_pool7/k36 | 0.708425 |
| Completeness | mini75_pool5/k36 | 0.766965 |

Mini112/k36 guida quattro metriche validation su cinque. Mini75/k36 guida
la completeness.

## 14. Interpretazione scientifica

Nel protocollo matched congelato, la combinazione `mini112_pool7/k36`
mostra il miglior equilibrio osservato. Guida purity, NMI, ARI e
homogeneity validation, ha il cluster massimo train più piccolo e mantiene
la pipeline no-PCA direttamente confrontabile con la futura pipeline VAE.

`mini75_pool5/k36` non è dominata perché raggiunge la completeness più alta.
Resta il controllo a risoluzione ridotta. `mini224_pool14/k36` è dominata
metricamente da `mini112_pool7/k36`, ma resta il controllo ad alta
risoluzione sul subset.

Per `k=22`, mini112 domina sia mini75 sia mini224 sulle cinque metriche
validation. `k=36` migliora generalmente separazione e bilanciamento, mentre
`k=22` conserva il ruolo di controllo legato al numero delle classi note.

Non esiste un vincitore universale: il fronte contiene sia mini112/k36 sia
mini75/k36. La scelta primaria non deriva da una media arbitraria delle
metriche né dall'inerzia.

## 15. Limiti

- La validation contiene soltanto 200 eventi; i gap positivi non implicano
  una migliore generalizzazione causale.
- Le metriche esterne dipendono dalle label e descrivono soltanto la
  valutazione a posteriori delle assegnazioni congelate.
- Le classi e i cluster sono sbilanciati; tutte le configurazioni hanno
  almeno un cluster train unitario.
- Il singolo seed congelato garantisce confrontabilità, ma non quantifica la
  variabilità fra inizializzazioni.
- Le inerzie appartengono a spazi di feature diversi e non sono confrontate
  scientificamente fra profili.
- Risoluzione, kernel di pooling e griglia pooled cambiano insieme e non sono
  completamente separabili.
- Il confronto matched è osservazionale: non consente di attribuire
  causalmente i risultati alla sola risoluzione.

## 16. Decisione metodologica

La configurazione primaria matched è `mini112_pool7/k36`, giustificata
congiuntamente da quattro metriche validation leader, cluster massimo più
piccolo, identità del subset, assenza di PCA e confrontabilità con il futuro
mini VAE in C.

`mini75_pool5/k36` e `mini224_pool14/k36` restano rispettivamente i controlli
a risoluzione ridotta e alta. `k=22` resta il controllo legato al numero
delle classi note.

## 17. Ruoli finali

```text
UNIVERSAL_WINNER=false
PRIMARY_MATCHED_BASELINE=mini112_pool7_k36
COMPLETENESS_LEADER=mini75_pool5_k36
LOW_RESOLUTION_CONTROL=mini75_pool5_k36
HIGH_RESOLUTION_CONTROL=mini224_pool14_k36
CLASS_COUNT_CONTROL=k22
RESOLUTION_ONLY_CAUSAL_CLAIM=false
EXTRACTOR_CONFIGURATION_COUPLED=true
MATCHED_BASELINE_PHASE_COMPLETE=true
SAFE_TO_START_MINI_VAE=true
```

## 18. Fonti e checksum

Tutti i percorsi sono relativi alla radice del repository. Per ogni base,
`base_source` è il catalogo checksum ufficiale `sha256.txt`; il valore
riportato è lo SHA-256 del catalogo stesso.

| Profilo | Tipo | File | SHA-256 |
|---|---|---|---|
| mini75_pool5 | base | `sakai_multires_c/results/baseline/matched_subset1000/mini75_pool5_no_pca/sha256.txt` | `081703f4b88f221a58bc7285100552a1fe8afc1dae74993e4a6aa5de94f8ee55` |
| mini75_pool5/k22 | k-means | `sakai_multires_c/results/baseline/matched_subset1000/mini75_pool5_no_pca/kmeans_seed123/k22/summary.txt` | `2e9ae1068a3ec01936429b53ac96a8a3b0b80048179afcb7ba311007abab830d` |
| mini75_pool5/k22 | metriche | `sakai_multires_c/results/baseline/matched_subset1000/mini75_pool5_no_pca/label_evaluation_seed123/k22/metrics.csv` | `358a135134dd96ea493ff9e65d23bb4fe08578734648398300724e3dbdc551c7` |
| mini75_pool5/k36 | k-means | `sakai_multires_c/results/baseline/matched_subset1000/mini75_pool5_no_pca/kmeans_seed123/k36/summary.txt` | `83697e46e5dd89b167092b8a37c6a259a537a16920667b244e15aa832ba13988` |
| mini75_pool5/k36 | metriche | `sakai_multires_c/results/baseline/matched_subset1000/mini75_pool5_no_pca/label_evaluation_seed123/k36/metrics.csv` | `2bb9634797fbcbd035f7ae38e24693beffcbeee034f0aa3088153ce0ad1c33a6` |
| mini112_pool7 | base | `sakai_multires_c/results/baseline/matched_subset1000/mini112_pool7_no_pca/sha256.txt` | `99f51cf2f265f38708fef80c2180dec6bb13cbf01d7bd81b5af27d23a4a30261` |
| mini112_pool7/k22 | k-means | `sakai_multires_c/results/baseline/matched_subset1000/mini112_pool7_no_pca/kmeans_seed123/k22/summary.txt` | `bfce016b62a20d33a54603b4dd8be000ef454a8c4a06e933a49c42aab25c4e28` |
| mini112_pool7/k22 | metriche | `sakai_multires_c/results/baseline/matched_subset1000/mini112_pool7_no_pca/label_evaluation_seed123/k22/metrics.csv` | `ccb759edc9951811dbf0f2a76b2438b0eb0ec2e3cc7f903b89ea30194ba2927d` |
| mini112_pool7/k36 | k-means | `sakai_multires_c/results/baseline/matched_subset1000/mini112_pool7_no_pca/kmeans_seed123/k36/summary.txt` | `7466e322968f9141ffcfd0f48e5a461f0d4c70615f28da21762b9f3585c04f5e` |
| mini112_pool7/k36 | metriche | `sakai_multires_c/results/baseline/matched_subset1000/mini112_pool7_no_pca/label_evaluation_seed123/k36/metrics.csv` | `6c59a2869a54c50f7aa1f6fae6b70d70ca7c292ffd5e51f5a85080be920f433f` |
| mini224_pool14 | base | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/sha256.txt` | `02f44783baf84905e328bdc48a0709fecd095b9db14a5680ec3e46298a9ffffa` |
| mini224_pool14/k22 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/kmeans_seed123/k22/summary.txt` | `74acfa6349b431cfa0be23444354a0b9260acf5f0c4008e74b07bd4a15055c05` |
| mini224_pool14/k22 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/label_evaluation_seed123/k22/metrics.csv` | `416e04802967c4a133d2776660d8d9a4d7c73e2a7774a7b364db17b39a876bf3` |
| mini224_pool14/k36 | k-means | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/kmeans_seed123/k36/summary.txt` | `6e2bf38b9250cdf98c4452cc9a5d88c2499e44a5e31d417adef1b568cebe7101` |
| mini224_pool14/k36 | metriche | `sakai_multires_c/results/baseline/mini224_subset1000_pool14_no_pca/label_evaluation_seed123/k36/metrics.csv` | `2c349c6d250ae54ef671749a90c75aa43d73979c3e3b88de04f01457a130526c` |

I valori completi e le fonti riga per riga sono in
`matched_subset1000_no_pca_metrics.csv`; i delta completi sono in
`matched_subset1000_no_pca_pairwise_deltas.csv`.

## 19. Handoff al mini VAE in C

Il confronto matched no-PCA è formalmente completo. Le tre configurazioni
primarie e di controllo sono congelate su eventi, split e protocollo
identici. È quindi possibile iniziare il mini VAE in C usando
`mini112_pool7/k36` come baseline matched primaria, senza alterare le
baseline e gli artefatti qui confrontati.
