# Machine Learning per Clustering ed Analisi dei Glitch dei Rivelatori Interferometrici di Onde Gravitazionali

## Contesto accademico

Tesi di Laurea Triennale in Ingegneria Informatica  
Università Telematica Internazionale UNINETTUNO — Facoltà di Ingegneria  
Corso di Laurea Triennale in Ingegneria Informatica  
Elaborato Finale in Reti Neurali  
Candidata: Laura Carzedda  
Relatore: Prof. Luigi Laura  
Correlatore: Dott. Marco Parrillo  
Anno Accademico 2025/2026

## Panoramica

Questa repository raccoglie il lavoro sperimentale svolto per la tesi sul clustering non supervisionato dei glitch, o transient noise, osservati nei rivelatori interferometrici di onde gravitazionali e rappresentati nel dataset Gravity Spy O1. L'obiettivo è studiare morfologie ricorrenti senza usare le classi note per guidare il training o la costruzione delle partizioni.

Il progetto ricostruisce e adatta il flusso scientifico descritto da Sakai et al. 2022. La pipeline principale `full224`, sviluppata in Python e PyTorch, lavora su viste a risoluzione 224×224. Una componente complementare in C11 documenta contratti, strumenti, test e adattamenti multirisoluzione, con attenzione anche a scenari hardware con risorse limitate. Il lavoro riproduce la sequenza metodologica generale e ne dichiara gli adattamenti, ma non rivendica una replica numerica esatta dei risultati di Sakai.

## Obiettivo scientifico

Il flusso principale apprende una rappresentazione latente dei glitch mediante un variational autoencoder (VAE), utilizza il vettore medio `μ` dell'encoder come embedding e costruisce una partizione attraverso Invariant Information Clustering (IIC), rappresentazione aggregata `H`, matrice di affinity e clustering spectral.

La separazione fra apprendimento non supervisionato e valutazione esterna è un vincolo metodologico centrale. Le label Gravity Spy non guidano il training del VAE, la IIC o il clustering primario. Sono consultate soltanto dopo il congelamento delle partizioni, per misurare e interpretare la corrispondenza fra cluster e classi note.

## Pipeline scientifica

```text
Preprocessing / data contract
→ VAE 224×224
→ μ512
→ IIC
→ H
→ affinity
→ spectral clustering
→ valutazione esterna post-hoc
```

La configurazione IIC primaria usa `C = 36` classi principali, un head di overclustering con `W = 250` e `K = 5` classificatori indipendenti. La rappresentazione risultante ha forma `8583 × 180`, con `180 = K × C`.

La IIC `full224` è transduttiva: opera congiuntamente sull'intera popolazione di 8.583 eventi. Il VAE segue invece uno split distinto di 6.866 eventi di training e 1.717 di validation. Questi due protocolli hanno funzioni diverse e non vanno interpretati come un unico schema train/test.

Il ramo primario è `symmetric_figure`, definito a partire dalla lettura della figura metodologica. Il ramo `legacy_text` è mantenuto separatamente come sensitivity analysis e non sostituisce il risultato primario.

## Risultati principali

Per la configurazione primaria `symmetric_figure`, la valutazione post-hoc sulla popolazione transduttiva congelata riporta:

| Metrica | Valore |
|---|---:|
| Purity pesata | 0,8264 |
| NMI aritmetica | 0,7018 |
| Adjusted Rand Index (ARI) | 0,4510 |
| Sakai-like macro ricostruita | 0,8514 |

La misura “Sakai-like macro” è una ricostruzione specifica del progetto. Non deve essere interpretata come replica diretta del 90,9% riportato da Sakai: definizioni delle metriche, popolazioni e protocolli sperimentali non sono dimostrati equivalenti. Purity pesata, Sakai-like macro e valore pubblicato nel lavoro di riferimento restano quindi quantità distinte.

## Dati

La popolazione censita comprende 8.583 eventi Gravity Spy O1. Lo split usato dal VAE contiene 6.866 eventi di training e 1.717 di validation. Ogni evento è rappresentato da quattro viste temporali, con durate di 0,5 s, 1,0 s, 2,0 s e 4,0 s. Il tensore `full224` ha forma `4 × 224 × 224`, tipo `uint8` e disposizione channel-major.

La repository include manifest, split, subset, configurazioni e altri contratti leggeri necessari a identificare i dati e controllarne la provenienza. Non distribuisce automaticamente immagini raw o preprocessate, tensori, checkpoint, embedding `μ`, matrici `H`, matrici di affinity, artifact spectral o altri payload scientifici pesanti. Questa separazione è intenzionale: il source e i metadata rimangono versionabili, mentre i dati voluminosi sono gestiti come artifact esterni.

## Struttura della repository

- `notebooks/full224/`: workflow scientifico completo, dalla verifica dell'ambiente alla valutazione IIC/spectral.
- `sakai_multires_c/`: implementazione C11, contratti, configurazioni e infrastruttura di test.
- `sakai_multires_c/src/`: implementazioni core registrate nella build supportata.
- `sakai_multires_c/include/`: interfacce e header dei componenti C.
- `sakai_multires_c/tools/`: producer, runner ed evaluator; alcuni richiedono artifact esterni.
- `sakai_multires_c/tests/`: test self-contained, test sui metadata e integration test condizionali.
- `sakai_multires_c/configs/`: profili, subset e configurazioni riproducibili.
- `sakai_multires_c/docs/`: decisioni metodologiche, verifiche e documentazione di provenance.
- `01_*.c`–`06_*.c`: utility manuali per le fasi di preprocessing.
- CSV top-level: manifest e split scientifici leggeri e congelati.

## Notebook full224

I notebook seguono questo ordine scientifico:

1. `00_colab_environment_and_inventory.ipynb` — setup dell'ambiente, inventario e contratti iniziali.
2. `01_sakai224_full_loader_and_data_tests.ipynb` — loader, split, augmentation deterministica e data contract.
3. `02_sakai224_full_model_smoke.ipynb` — smoke test dell'architettura VAE.
4. `03_sakai224_full_benchmark_and_training.ipynb` — benchmark, training VAE, checkpoint e verifica del resume.
5. `04_sakai224_full_posttraining_audit_and_embeddings.ipynb` — verifiche post-training, ricostruzioni, `μ512` e diagnostica t-SNE.
6. `05_sakai224_full_kmeans_and_evaluation.ipynb` — baseline diagnostica k-means label-blind.
7. `06_sakai224_full_iic_and_spectral.ipynb` — pipeline primaria IIC → `H` → affinity → spectral e valutazione esterna.

Il notebook 05 non rappresenta il classificatore finale Sakai-like: k-means costituisce una baseline separata per esaminare la struttura degli embedding prima della pipeline IIC/spectral.

## Build C

La build supportata richiede CMake 3.20 o successivo e usa lo standard C11 con le estensioni del compilatore disabilitate. La policy dei warning imposta `/W4` per MSVC e `-Wall -Wextra -Wpedantic` per GNU/Clang.

La verifica empirica di questa release è stata eseguita con GNU C 16.1.0 su Windows/UCRT, usando Ninja. La configurazione CMake di default e la build hanno entrambe avuto esito positivo: 98 target, 232 azioni di build, nessun warning e nessun errore del compilatore. Il CMake contiene anche rami dedicati a MSVC e GNU/Clang, ma questi dati non costituiscono una validazione empirica su ogni piattaforma.

Esempio minimale della build C verificata:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Test

La configurazione CTest di default comprende 80 test. Prima dell'inizializzazione Git risultano applicabili 43 test: 29 self-contained e 14 dipendenti soltanto dai metadata inclusi nella release. Tutti i 43 test pre-Git applicabili che dipendono soltanto dal source o dai metadata inclusi hanno superato la verifica.

Gli altri 37 test configurati sono intenzionalmente differiti perché richiedono una o più risorse non presenti nel checkout leggero: dataset esterno, checkpoint, embedding `μ`, affinity, artifact spectral, artifact d'integrazione oppure stato VCS. Non sono test falliti. Tre bridge Python spectral aggiuntivi richiedono inoltre lo specifico ambiente scientifico congelato e non sono configurati nel default.

## Baseline e componenti opzionali

`baseline_kmeans` e `clustering_metrics` sono componenti opzionali già inclusi nella build supportata. Le restanti superfici mini75, mini112, mini224 subset1000, matched subset e le varianti PCA/no-PCA sono conservate soprattutto come sorgenti di riferimento per la riproducibilità e non sono tutte parte della build CMake di default. Le varianti PCA richiedono LAPACKE/BLAS. I programmi top-level numerati da 01 a 06 restano utility manuali di preprocessing.

## Riproducibilità

I notebook distinguono due root. `GRAVITYSPY_PROJECT_ROOT` identifica lo spazio persistente per progetto, dataset e artifact; `GRAVITYSPY_LOCAL_RUNTIME_ROOT` identifica invece lo spazio locale per cache e dati runtime rigenerabili. Nel contesto Colab storico i default sono rispettivamente `/content/drive/MyDrive/TESI_SAKAI2022` e `/content`. Fuori da Colab, la project root viene fornita come path assoluto.

Environment software e artifact esterni sono descritti esplicitamente nella documentazione di release associata alle rispettive esecuzioni. Alcuni CSV scientifici sono inoltre preservati byte-for-byte perché i loro SHA-256 fanno parte dei test e della provenance.

### Ambiente Python

[requirements.txt](requirements.txt) raccoglie le dipendenze Python generali della superficie pubblicata dei notebook e dei tool: NumPy, SciPy, scikit-learn 1.6.1, PyTorch, threadpoolctl, Pillow, Matplotlib e psutil. Non è un lock esatto della run full224 storica; il solo pin generale intenzionale è `scikit-learn==1.6.1`.

```bash
python -m pip install -r requirements.txt
```

Il percorso full224 pubblicato è orientato all'esecuzione GPU/CUDA. La run storica documentata nei notebook ha usato Python 3.13.15, PyTorch 2.11.0+cu128 e CUDA 12.8: queste versioni costituiscono provenance della run, non un lock generale della repository. `requirements.txt` non fissa una specifica wheel CUDA di PyTorch e non sostituisce i requisiti hardware, driver e runtime GPU. Il modulo `google.colab` è usato condizionalmente dai notebook quando vengono eseguiti su Colab.

[requirements-spectral.txt](requirements-spectral.txt) è autonomo e rappresenta l'ambiente numerico congelato dei bridge spectral mini75, mini112 e mini224. Richiede esattamente Python 3.12.14 e `numpy==2.5.3`, `scipy==1.16.3`, `scikit-learn==1.6.1`, `joblib==1.6.0`, `threadpoolctl==3.7.0`. Questo ambiente è stato installato e verificato con successo in un virtual environment isolato; i bridge controllano esattamente tali versioni.

Con Python 3.12.14:

```bash
python -m pip install -r requirements-spectral.txt
```

I requirements descrivono le dipendenze software, ma non forniscono dataset, checkpoint, embedding `μ`, rappresentazioni `H`, affinity o artifact spectral necessari alle esecuzioni complete; questi input e artifact scientifici esterni sono descritti separatamente.

## Artifact scientifici esterni

La repository Git non contiene i payload scientifici pesanti: dataset completo, PNG preprocessati, tensori, checkpoint, embedding `μ`, rappresentazioni `H`, affinity, eigensystem, partizioni e bundle di valutazione. Questi artifact sono documentati e identificati esternamente, senza presumerne la disponibilità pubblica.

La pipeline scientifica può essere ricostruita a partire dalla repository, dal dataset sorgente Gravity Spy v1.0.0 ([DOI 10.5281/zenodo.1476156](https://doi.org/10.5281/zenodo.1476156)) e da un ambiente software e hardware adeguato. Questo non implica identità byte-for-byte della specifica run GPU storica su hardware, backend o stack software differenti.

Quando disponibili, gli artifact congelati consentono una riproduzione per stadi o la verifica dei risultati senza rieseguire tutte le fasi: checkpoint → `μ`, `μ` → IIC, `H` → affinity, affinity → spectral, partizione → valutazione post-hoc.

[ARTIFACTS.md](ARTIFACTS.md) documenta dataset sorgente, artifact di preprocessing, checkpoint, embedding `μ`, IIC, `H`, affinity, spectral, valutazione, livelli di ricostruzione, checksum e stato di disponibilità. [artifact_manifest.tsv](artifact_manifest.tsv) è l'indice machine-readable; [artifact_checksums.sha256](artifact_checksums.sha256) raccoglie i digest SHA-256 dei singoli file con identità completa. I percorsi del manifest sono contratti relativi per artifact esterni e non implicano che i file siano inclusi nella repository o ospitati pubblicamente.

[requirements.txt](requirements.txt) e [requirements-spectral.txt](requirements-spectral.txt) descrivono le dipendenze software; `ARTIFACTS.md` e `artifact_manifest.tsv` descrivono invece il dataset e gli artifact scientifici esterni.

## Limiti e scope

- Il progetto ricostruisce e adatta il flusso di Sakai, ma non ne dichiara una replica numerica esatta.
- Dataset e artifact scientifici pesanti non sono inclusi nel checkout leggero.
- Alcuni integration test restano differiti finché le relative dipendenze esterne non sono disponibili.
- La build empiricamente documentata usa GNU C su Windows/UCRT; la presenza di altri rami CMake non equivale a una validazione su tutte le piattaforme.
- Parte delle baseline è mantenuta come source/reference e non appartiene alla superficie di build supportata di default.
- La pipeline IIC primaria è transduttiva sull'intera popolazione di 8.583 eventi.
- Le metriche post-hoc descrivono la corrispondenza con le label note e non dimostrano l'esistenza di nuove sottoclassi fisiche.

## Riferimenti

Il contesto scientifico comprende Sakai et al. 2022, Gravity Spy e i relativi materiali Scientific Reports e Supplemental Material esaminati nel progetto. I riferimenti bibliografici completi e le istruzioni di citazione saranno forniti nei metadata di citazione della release.

## Citazione

I metadata di citazione della repository e della tesi sono disponibili in [CITATION.cff](CITATION.cff).

Se si utilizza il software o il materiale originale di questa repository, si raccomanda di citare la repository e la tesi. Quando pertinenti, devono inoltre essere citati separatamente il dataset Gravity Spy O1 e il riferimento metodologico di Sakai et al. 2022, secondo i metadata riportati nel file di citazione.

## Licenza

Il materiale originale del progetto è distribuito secondo la [MIT License](LICENSE).

I componenti e i contenuti di terze parti mantengono le rispettive licenze e attribuzioni. In particolare, gli header `stb_image.h` e `stb_image_write.h` conservano le licenze incorporate; il dataset Gravity Spy O1 non è rilicenziato dalla licenza del progetto e il relativo record v1.0.0 è distribuito sotto CC BY 4.0. I metadata derivati e gli artifact scientifici esterni hanno uno scope distinto.

Per i dettagli si veda [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
