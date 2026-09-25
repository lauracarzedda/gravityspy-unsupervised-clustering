# Decisione di progetto v1 sulla valutazione post-hoc comune

Data: 19 settembre 2026

Versione: 1.0

Autorità della decisione: Laura

Stato: ratificata e documentata separatamente

## 1. Scopo della decisione

Questa decisione autorizza la valutazione post-hoc comune e sistematica delle partizioni spectral già congelate dei rami mini75, mini112, mini224_1000 e sakai224_full.

Non si tratta della prima apertura delle label nel progetto. Sono già esistite valutazioni post-hoc storiche per full224 e mini75. Sul ramo full224 è inoltre in corso una lettura interpretativa post-hoc di tabelle e contact sheet. Tali attività non hanno modificato le partizioni scientifiche e vengono riconosciute come precedenti storici separati dalla nuova valutazione comune.

La presente decisione non esegue evaluator, non calcola metriche e non modifica alcun artefatto scientifico.

## 2. Stato pre-evaluation

Il gate finale label-free ha stabilito:

- branch: `vae-c-sakai-like`;
- HEAD: `7c7b0e4b6d4a93b7e89c20cdf07419a31e088df5`;
- file sotto `sakai_multires_c/results`: 1293;
- fingerprint pre-evaluation: `ea966f1ec1eee9d2adebbd84a3f64d16a291dc2cc3f282d8d567537ab0ec3e86`;
- procedura fingerprint: `RESULTS_FINGERPRINT_V1_POWERSHELL_CASESENSITIVE`;
- blocker label-free residui: 0;
- tutte le partizioni primarie richieste congelate: true;
- full224 PRIMARY e SENSITIVITY disponibili localmente: true.

Il fingerprint identifica lo stato pre-evaluation e deve essere conservato come baseline per distinguere gli artefatti scientifici congelati dagli output della futura valutazione.

## 3. Decisione spectral backend già efficace

La decisione versionata sul backend spectral dei mini è già committata:

- documento: `sakai_multires_c/docs/project_decision_spectral_backend_v1_2026-09-19.md`;
- SHA-256: `598eb9a337719ca3869666361af1990fe9a93384bbf8b2156c508429026cd786`;
- commit: `7c7b0e4b6d4a93b7e89c20cdf07419a31e088df5`.

Per i tre mini la pipeline resta descritta come C11 fino alla matrice di affinità inclusa, seguito dal Python reference spectral bridge. Questa scelta non richiede nuove esecuzioni e lascia canoniche le partizioni già congelate.

## 4. Partizioni autorizzate alla valutazione comune

La valutazione comune deve usare esclusivamente le seguenti partizioni congelate:

| Ruolo | Partizione | Popolazione | Uso nella valutazione |
|---|---|---:|---|
| PRIMARY | mini75 spectral primary | 8583 eventi | Ramo mini primario |
| PRIMARY | mini112 spectral primary | 8583 eventi | Ramo mini primario |
| PRIMARY | mini224_1000 spectral primary | 1000 eventi, 800 train e 200 validation | Ramo mini primario con popolazione eccezionale |
| PRIMARY | full224 `symmetric_v1` | 8583 eventi | Riferimento principale full224 |
| SENSITIVITY | full224 `legacy_text_v1` | 8583 eventi | Analisi secondaria di sensibilità |

Il ramo full224 PRIMARY `symmetric_v1` è il riferimento principale per i confronti con i mini e con Sakai. Il ramo full224 SENSITIVITY `legacy_text_v1` deve essere presentato separatamente come controllo secondario e non deve sostituire il PRIMARY.

mini224_1000 è l'unica eccezione di popolazione. I suoi risultati non devono essere presentati come confronto a parità di numero di eventi con mini75, mini112 o full224.

## 5. Autorizzazione all'uso delle label

Da questa decisione in avanti, le label Gravity Spy possono essere usate esclusivamente per la valutazione e l'interpretazione post-hoc delle partizioni sopra elencate.

Le label non possono influenzare o modificare:

- VAE;
- rappresentazioni latenti μ;
- IIC;
- over-clustering;
- ensemble K = 5;
- H;
- affinity;
- spectral clustering;
- seed;
- popolazioni;
- configurazioni;
- cluster assignment congelati.

Le metriche osservate non autorizzano selezione retrospettiva, retraining, reclustering, variazione dei seed, sostituzione della partizione PRIMARY o modifica di alcun contratto scientifico.

Non devono essere creati nuovi fit IIC o spectral matched da 1000 eventi per mini75, mini112 o full224. Il matched subset1000 resta confinato alle baseline già prodotte.

## 6. Protocollo quantitativo comune

Per ogni partizione valutata devono essere prodotti separatamente:

- matrice di contingenza completa 22 x 36;
- purity;
- normalized mutual information con media aritmetica;
- adjusted Rand index;
- homogeneity;
- completeness.

La join fra partizione e label deve rispettare il contratto di ordine e identità degli eventi congelato per ciascun ramo. La diversa popolazione di mini224_1000 deve essere dichiarata in ogni tabella o confronto che lo includa.

La valutazione di full224 SENSITIVITY deve restare distinta da quella del PRIMARY. Non deve essere effettuata alcuna selezione post-hoc fra PRIMARY e SENSITIVITY sulla base delle metriche.

## 7. Valutazione quantitativa e interpretazione

La valutazione quantitativa comune comprende contingenze e metriche esterne calcolate sulle partizioni immutabili.

L'interpretazione successiva comprende invece:

- analisi morfologica dei cluster;
- contact sheet e immagini rappresentative;
- lettura delle sottoclassi;
- t-SNE e relativa discussione;
- confronto qualitativo con Sakai;
- stesura della tesi.

La revisione visiva full224 già in corso non è un prerequisito per avviare la valutazione quantitativa comune. Non deve guidare modifiche alle partizioni o alla configurazione degli esperimenti.

## 8. Immutabilità e provenance

Gli evaluator futuri devono trattare assignment, row-map, metadata e manifest delle cinque partizioni come input read-only. Gli output della valutazione devono essere pubblicati in una root separata e devono riportare la provenance completa degli input, inclusi path, SHA-256, popolazione e ruolo PRIMARY o SENSITIVITY.

La baseline `results` registrata nella sezione 2 deve essere verificata prima della prima produzione comune. Qualunque nuovo output deve essere distinguibile dai 1293 file preesistenti e non deve sovrascrivere evaluation storiche full224 o mini75.

## 9. Stato della decisione

- `COMMON_POSTHOC_EVALUATION_AUTHORIZED=true`
- `HISTORICAL_POSTHOC_EVALUATIONS_ACKNOWLEDGED=true`
- `ALL_PARTITIONS_REMAIN_FROZEN=true`
- `NEW_MATCHED_IIC_SPECTRAL_RUNS_REQUIRED=false`
- `RESIDUAL_LABEL_FREE_BLOCKERS_COUNT=0`
- `METRICS_NOT_YET_COMPUTED_IN_THIS_DECISION=true`
