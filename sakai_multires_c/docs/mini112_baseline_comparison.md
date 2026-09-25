# Confronto finale della baseline mini112

## Scopo e checkpoint

Questo documento confronta le otto configurazioni ufficiali della baseline
centrale mini112 pool7: quattro profili di rappresentazione, ciascuno con
`k=22` e `k=36`. I valori provengono esclusivamente dagli artefatti ufficiali
rivalidati al checkpoint Git
`97b5d45b20bf8db57dc42076be0de248da531142` sul branch
`baseline-mini112-pool7`.

Il confronto non introduce nuovi fit, trasformazioni o metriche. Le label
Gravity Spy entrano soltanto nella valutazione a posteriori delle assegnazioni
congelate.

## Dataset e protocollo comune

- 8583 eventi completi, identici per tutte le configurazioni;
- split ufficiale immutabile con seed 123: 6866 train e 1717 validation;
- k-means++ deterministico con seed 123 e massimo 300 iterazioni;
- `k=22`, controllo coerente con le 22 classi Gravity Spy, e `k=36`;
- fit k-means soltanto sul train;
- validation assegnata ai centroidi appresi sul train;
- label escluse da feature, standardizzazione, PCA e clustering;
- purity pesata, NMI con normalizzazione aritmetica, ARI, homogeneity e
  completeness calcolate soltanto a posteriori;
- nessuna augmentation e nessun whitening.

I quattro profili sono:

| Profilo | PCA | Dimensione | Frazione mantenuta | Compressione | Varianza cumulativa |
|---|---|---:|---:|---:|---:|
| no_pca | no | 1024 | 1.000000 | 1.000000x | n/a |
| pca90 | sì | 20 | 0.019531 | 51.200000x | 0.901126 |
| pca95 | sì | 61 | 0.059570 | 16.786885x | 0.950321 |
| pca99 | sì | 263 | 0.256836 | 3.893536x | 0.990012 |

## Risultati train

| Profilo | k | Dim. | Purity | NMI | ARI | Homogeneity | Completeness | Iter. | Inerzia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_pca | 22 | 1024 | 0.523595 | 0.554837 | 0.293032 | 0.471235 | 0.674499 | 23 | 1441584.450660 |
| no_pca | 36 | 1024 | 0.602826 | 0.598595 | 0.337980 | 0.546542 | 0.661608 | 26 | 1218828.542160 |
| pca90 | 22 | 20 | 0.478008 | 0.452322 | 0.135738 | 0.340634 | 0.672983 | 21 | 771499.875711 |
| pca90 | 36 | 20 | 0.575881 | 0.578928 | 0.267181 | 0.526858 | 0.642420 | 40 | 583636.604249 |
| pca95 | 22 | 61 | 0.496650 | 0.445331 | 0.124322 | 0.343730 | 0.632202 | 45 | 1124672.660367 |
| pca95 | 36 | 61 | 0.585931 | 0.587136 | 0.305230 | 0.524536 | 0.666702 | 32 | 882920.732866 |
| pca99 | 22 | 263 | 0.538742 | 0.528794 | 0.182307 | 0.461880 | 0.618381 | 45 | 1414339.530967 |
| pca99 | 36 | 263 | 0.629624 | 0.604756 | 0.293907 | 0.584148 | 0.626870 | 37 | 1191256.909985 |

## Risultati validation

| Profilo | k | Dim. | Purity | NMI | ARI | Homogeneity | Completeness | Iter. | Inerzia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_pca | 22 | 1024 | 0.538148 | 0.580299 | 0.313844 | 0.495079 | 0.700957 | 23 | 419859.327780 |
| no_pca | 36 | 1024 | 0.621433 | 0.626992 | 0.359326 | 0.572530 | 0.692903 | 26 | 362820.963980 |
| pca90 | 22 | 20 | 0.486896 | 0.472885 | 0.145055 | 0.357927 | 0.696626 | 21 | 209385.052527 |
| pca90 | 36 | 20 | 0.588235 | 0.596756 | 0.266608 | 0.544077 | 0.660728 | 40 | 171706.597716 |
| pca95 | 22 | 61 | 0.507280 | 0.467490 | 0.129722 | 0.362362 | 0.658546 | 45 | 318000.101645 |
| pca95 | 36 | 61 | 0.602213 | 0.611650 | 0.319553 | 0.546172 | 0.694964 | 32 | 262996.524960 |
| pca99 | 22 | 263 | 0.552708 | 0.551169 | 0.196543 | 0.481952 | 0.643603 | 45 | 370408.750468 |
| pca99 | 36 | 263 | 0.641817 | 0.628009 | 0.309456 | 0.606590 | 0.650997 | 37 | 336057.067297 |

I leader validation, calcolati metrica per metrica senza aggregazione, sono:
PCA-99/k36 per purity, NMI e homogeneity; no-PCA/k36 per ARI; no-PCA/k22
per completeness.

## Gap train-validation

Il gap è definito qui come `validation - train`. Un valore positivo indica
una metrica validation più alta; non è un punteggio di selezione aggregato.

| Profilo | k | Δ purity | Δ NMI | Δ ARI | Δ homogeneity | Δ completeness |
|---|---:|---:|---:|---:|---:|---:|
| no_pca | 22 | 0.014553 | 0.025462 | 0.020812 | 0.023844 | 0.026458 |
| no_pca | 36 | 0.018607 | 0.028396 | 0.021346 | 0.025989 | 0.031295 |
| pca90 | 22 | 0.008888 | 0.020563 | 0.009317 | 0.017293 | 0.023643 |
| pca90 | 36 | 0.012354 | 0.017827 | -0.000573 | 0.017219 | 0.018308 |
| pca95 | 22 | 0.010630 | 0.022158 | 0.005400 | 0.018632 | 0.026344 |
| pca95 | 36 | 0.016282 | 0.024514 | 0.014323 | 0.021636 | 0.028262 |
| pca99 | 22 | 0.013967 | 0.022375 | 0.014235 | 0.020072 | 0.025222 |
| pca99 | 36 | 0.012193 | 0.023254 | 0.015548 | 0.022442 | 0.024126 |

## Passaggio da k=22 a k=36

Le differenze seguenti sono `k36 - k22` nello stesso profilo e split.

| Profilo | Split | Δ purity | Δ NMI | Δ ARI | Δ homogeneity | Δ completeness |
|---|---|---:|---:|---:|---:|---:|
| no_pca | train | 0.079231 | 0.043759 | 0.044948 | 0.075306 | -0.012891 |
| no_pca | validation | 0.083285 | 0.046693 | 0.045482 | 0.077451 | -0.008054 |
| pca90 | train | 0.097874 | 0.126606 | 0.131443 | 0.186224 | -0.030563 |
| pca90 | validation | 0.101340 | 0.123871 | 0.121553 | 0.186150 | -0.035898 |
| pca95 | train | 0.089281 | 0.141805 | 0.180908 | 0.180807 | 0.034501 |
| pca95 | validation | 0.094933 | 0.144160 | 0.189831 | 0.183811 | 0.036419 |
| pca99 | train | 0.090883 | 0.075962 | 0.111600 | 0.122268 | 0.008490 |
| pca99 | validation | 0.089109 | 0.076840 | 0.112913 | 0.124638 | 0.007394 |

`k=36` migliora purity, NMI, ARI e homogeneity rispetto a `k=22` in
tutti e quattro i profili, sia sul train sia sulla validation. La completeness
diminuisce per no-PCA e PCA-90, mentre aumenta per PCA-95 e PCA-99.

## Delta PCA rispetto a no-PCA

Ogni differenza è `PCA - no-PCA` per lo stesso `k` e split. I valori completi,
incluse inerzia e statistiche dei cluster, sono in
`mini112_baseline_deltas_vs_no_pca.csv`.

| Profilo | k | Split | Δ purity | Δ NMI | Δ ARI | Δ homogeneity | Δ completeness |
|---|---:|---|---:|---:|---:|---:|---:|
| pca90 | 22 | train | -0.045587 | -0.102514 | -0.157294 | -0.130601 | -0.001516 |
| pca90 | 22 | validation | -0.051252 | -0.107414 | -0.168789 | -0.137152 | -0.004331 |
| pca90 | 36 | train | -0.026944 | -0.019667 | -0.070799 | -0.019684 | -0.019188 |
| pca90 | 36 | validation | -0.033197 | -0.030236 | -0.092718 | -0.028453 | -0.032175 |
| pca95 | 22 | train | -0.026944 | -0.109505 | -0.168710 | -0.127506 | -0.042298 |
| pca95 | 22 | validation | -0.030868 | -0.112809 | -0.184122 | -0.132718 | -0.042411 |
| pca95 | 36 | train | -0.016895 | -0.011459 | -0.032750 | -0.022005 | 0.005094 |
| pca95 | 36 | validation | -0.019220 | -0.015342 | -0.039773 | -0.026358 | 0.002061 |
| pca99 | 22 | train | 0.015147 | -0.026043 | -0.110725 | -0.009355 | -0.056119 |
| pca99 | 22 | validation | 0.014560 | -0.029130 | -0.117301 | -0.013128 | -0.057354 |
| pca99 | 36 | train | 0.026799 | 0.006160 | -0.044073 | 0.037606 | -0.034738 |
| pca99 | 36 | validation | 0.020384 | 0.001018 | -0.049870 | 0.034060 | -0.041906 |

Le inerzie appartengono a spazi di dimensionalità diversa: i relativi delta
sono mantenuti nel CSV per completezza descrittiva, ma non sono confronti di
qualità cross-profile e non sono usati per selezionare una configurazione.

## Dimensione dei cluster train

| Profilo | k | Iter. | Min train | Max train | Max/train |
|---|---:|---:|---:|---:|---:|
| no_pca | 22 | 23 | 1 | 2730 | 0.397611 |
| no_pca | 36 | 26 | 1 | 2360 | 0.343723 |
| pca90 | 22 | 21 | 2 | 4631 | 0.674483 |
| pca90 | 36 | 40 | 1 | 2544 | 0.370521 |
| pca95 | 22 | 45 | 2 | 4525 | 0.659045 |
| pca95 | 36 | 32 | 1 | 2551 | 0.371541 |
| pca99 | 22 | 45 | 2 | 2959 | 0.430964 |
| pca99 | 36 | 37 | 1 | 2179 | 0.317361 |

Tutte le configurazioni presentano almeno un cluster train molto piccolo
(una o due osservazioni). PCA-90/k22 e PCA-95/k22 sono anche fortemente
concentrate nel cluster maggiore, rispettivamente 67.4483% e 65.9045% del
train. Il massimo relativo più basso è PCA-99/k36 (31.7361%), seguito da
no-PCA/k36 (34.3723%). Questi dati descrivono il bilanciamento, senza
sostituire le metriche esterne.

## Fronte di Pareto sulle metriche validation

Una configurazione A domina B se A è almeno pari a B su tutte e cinque le
metriche validation ed è strettamente migliore su almeno una. Non sono usati
media delle metriche, somma dei rank, pesi o inerzia.

Configurazioni non dominate:

- no-PCA/k22;
- no-PCA/k36;
- PCA-95/k36;
- PCA-99/k36.

Per le configurazioni dominate è riportato almeno un dominatore verificato:

| Configurazione dominata | Un dominatore |
|---|---|
| PCA-90/k22 | no-PCA/k22 |
| PCA-90/k36 | no-PCA/k36 |
| PCA-95/k22 | no-PCA/k22 |
| PCA-99/k22 | no-PCA/k36 |

## Interpretazione scientifica

No-PCA/k36 ha il miglior ARI validation (0.359326), una NMI quasi uguale al
massimo (0.626992 contro 0.628009), buona completeness (0.692903) e conserva
la pipeline primaria priva di riduzione PCA. È quindi adatta al confronto
diretto fra risoluzioni senza introdurre una trasformazione aggiuntiva.

PCA-99/k36 raggiunge la migliore purity (0.641817), NMI (0.628009) e
homogeneity (0.606590) validation. Rispetto a no-PCA/k36, tuttavia, ha ARI
inferiore di 0.049870 e completeness inferiore di 0.041906. Rappresenta
l'ablation PCA ad alta conservazione, non un vincitore universale.

PCA-95/k36 usa 61 componenti, una compressione di circa 16.79x, e ottiene la
migliore completeness fra le configurazioni PCA con `k=36` (0.694964).
Costituisce il compromesso compatto fra compressione e prestazioni.

PCA-90 usa soltanto 20 componenti, con compressione 51.2x. La perdita netta
delle metriche rispetto a no-PCA, insieme alla forte concentrazione del
cluster maggiore per `k=22`, la rende un'ablation di compressione aggressiva,
non la baseline primaria.

## Ruolo raccomandato delle configurazioni

Non emerge un vincitore universale. La configurazione no-PCA con `k36` è la
baseline primaria per il confronto fra risoluzioni; `pca99/k36` rappresenta
l'ablation ad alta conservazione della varianza, `pca95/k36` il compromesso
compatto e `pca90` la compressione aggressiva. `k22` resta il controllo legato
al numero di classi.

Questi ruoli derivano congiuntamente dal protocollo, dalle cinque metriche
validation, dalla dimensionalità, dal bilanciamento dei cluster e dalla
confrontabilità fra risoluzioni. Non derivano dalla sola massimizzazione di
una metrica né da una media arbitraria. `k=22` resta il controllo coerente con
il numero di classi Gravity Spy, non la configurazione primaria.

## Limiti

- Le metriche esterne dipendono dalle label e sono quindi usate soltanto a
  posteriori; non misurano da sole la struttura fisica dei glitch.
- Le classi sono sbilanciate e le statistiche min/max mostrano cluster molto
  piccoli e, in alcuni profili, un cluster dominante.
- Un singolo seed congelato garantisce confrontabilità, ma non quantifica la
  variabilità fra inizializzazioni.
- Le inerzie non sono confrontabili come misura di qualità tra profili con
  dimensionalità diversa.
- Il fronte di Pareto descrive soltanto le cinque metriche validation
  specificate; dimensionalità e bilanciamento intervengono nell'assegnazione
  metodologica dei ruoli, non nella dominanza.

## File sorgente e SHA-256

Tutti i percorsi sono relativi alla radice del repository.

| Profilo | Tipo | File | SHA-256 |
|---|---|---|---|
| no_pca/k22 | metriche | `sakai_multires_c/results/baseline/mini112_pool7_no_pca/kmeans_seed123/label_evaluation_seed123/k22/metrics.csv` | `ce90c486bdfca317b0583e99791708073f6ce3d4bcf6729bc8ac518cd5afd02b` |
| no_pca/k22 | k-means | `sakai_multires_c/results/baseline/mini112_pool7_no_pca/kmeans_seed123/k22/summary.txt` | `a348b89fd53db3b524481f2286c800d89611be82f5db6a39e4ceea7f61ee6b01` |
| no_pca/k36 | metriche | `sakai_multires_c/results/baseline/mini112_pool7_no_pca/kmeans_seed123/label_evaluation_seed123/k36/metrics.csv` | `a7d1bb0dcf265e0438cfbea07abdccd2d4c6607bc440d2de5c3f1a800bc93e58` |
| no_pca/k36 | k-means | `sakai_multires_c/results/baseline/mini112_pool7_no_pca/kmeans_seed123/k36/summary.txt` | `5770c52d3820995f4e33621ae06692bdc880d10cb9d904d9b9b93ab70681deb9` |
| pca90 | specifica PCA | `sakai_multires_c/results/baseline/mini112_pool7_pca90/pca_spec.txt` | `88ea4a5fed4c4ec44bb6dbd98f0d55465264e99167a63c436bd6b4cdd14fc9eb` |
| pca90/k22 | metriche | `sakai_multires_c/results/baseline/mini112_pool7_pca90/label_evaluation_seed123/k22/metrics.csv` | `15ae8a428b24fe8cc880d43e78ad530ef9c726240414888f2de2c54362fec2e6` |
| pca90/k22 | k-means | `sakai_multires_c/results/baseline/mini112_pool7_pca90/kmeans_seed123/k22/summary.txt` | `f96d1452523ea8151ceda9461c8ee8e5213e063fcb107d0597614f75963f6afa` |
| pca90/k36 | metriche | `sakai_multires_c/results/baseline/mini112_pool7_pca90/label_evaluation_seed123/k36/metrics.csv` | `e20edd0c30936262e8d3d876324fd1271aabf8a6678c1cf7ad26a079fd6f384a` |
| pca90/k36 | k-means | `sakai_multires_c/results/baseline/mini112_pool7_pca90/kmeans_seed123/k36/summary.txt` | `a7be2d9bbccc0da849e89101985f6f983d2e8ddb36a57bedb3c0ab306a472812` |
| pca95 | specifica PCA | `sakai_multires_c/results/baseline/mini112_pool7_pca95/pca_spec.txt` | `d184f1d579814a0b5721d1197ee9f236bbeb0817247c77096af82606fc50ed89` |
| pca95/k22 | metriche | `sakai_multires_c/results/baseline/mini112_pool7_pca95/label_evaluation_seed123/k22/metrics.csv` | `c54c5d124ed94e420108da87879160760908631f261e0e57d8cc2c8bbb8e715a` |
| pca95/k22 | k-means | `sakai_multires_c/results/baseline/mini112_pool7_pca95/kmeans_seed123/k22/summary.txt` | `dedb32c92e438832aa7a04a472d7de26aff7fa3b663ac7be53a26dc72a019b72` |
| pca95/k36 | metriche | `sakai_multires_c/results/baseline/mini112_pool7_pca95/label_evaluation_seed123/k36/metrics.csv` | `cfeaf39cbfa3e07be35f5a15c97fd750bb6faf9d56a3399b3bd34cec3f33e2e4` |
| pca95/k36 | k-means | `sakai_multires_c/results/baseline/mini112_pool7_pca95/kmeans_seed123/k36/summary.txt` | `c20647c3b9a5d5cf84a2f4132b5bb84220282bbdda4119d7e1c0b344ebf82a36` |
| pca99 | specifica PCA | `sakai_multires_c/results/baseline/mini112_pool7_pca99/pca_spec.txt` | `259e794f5c54352626b6b9c95bb883db409fd0844f4f81076f2429ffb127ba92` |
| pca99/k22 | metriche | `sakai_multires_c/results/baseline/mini112_pool7_pca99/label_evaluation_seed123/k22/metrics.csv` | `860e9518fa889fd984886613f735da75dff601f5aded5eb25903499de3a87f82` |
| pca99/k22 | k-means | `sakai_multires_c/results/baseline/mini112_pool7_pca99/kmeans_seed123/k22/summary.txt` | `3a668b1fd23078ef8ca1cca76b3d281d98d6ff4229ad7cdae0b1785abd8c4302` |
| pca99/k36 | metriche | `sakai_multires_c/results/baseline/mini112_pool7_pca99/label_evaluation_seed123/k36/metrics.csv` | `a7d15ba4910eca6cc64f624dd72a4798d4afeecb91047c70a5505b3361338649` |
| pca99/k36 | k-means | `sakai_multires_c/results/baseline/mini112_pool7_pca99/kmeans_seed123/k36/summary.txt` | `92ee6edac8762938050dcb3238ced3b886cf369517bc7ed06aaa4a0933ffaa66` |
