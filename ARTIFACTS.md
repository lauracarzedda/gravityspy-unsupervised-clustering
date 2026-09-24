# Scientific data and external artifacts

## 1. Ambito della release

La repository Git contiene codice sorgente, notebook, manifest, split e subset, configurazioni, test e documentazione. Non contiene i payload scientifici pesanti descritti in questo documento.

[`artifact_manifest.tsv`](artifact_manifest.tsv) è l'indice machine-readable degli artifact. [`artifact_checksums.sha256`](artifact_checksums.sha256) definisce il contratto SHA-256 per i singoli file esterni che hanno un'identità completa e un percorso pubblico relativo stabile; la presenza di un percorso nel catalogo non implica che il file sia ospitato dalla repository o pubblicamente scaricabile.

La pipeline scientifica può essere ricostruita a partire dalla repository e dal dataset sorgente Gravity Spy v1.0.0; gli artifact intermedi elencati sono prodotti derivati e non sono necessari se tutte le fasi vengono rieseguite. Questo non implica identità byte-for-byte della run GPU storica su hardware o stack software differenti. La provenance software e hardware della run storica è specifica e `requirements.txt` non ne costituisce un lock completo.

## 2. Dataset sorgente

Il dataset sorgente è **Gravity Spy v1.0.0**, *Machine learning for Gravity Spy: Glitch classification and dataset*, DOI [10.5281/zenodo.1476156](https://doi.org/10.5281/zenodo.1476156), licenza CC BY 4.0. Il dataset non è incluso né redistribuito in questa repository.

Il file necessario per una ricostruzione da zero è `trainingsetv1d0.tar.gz`. `trainingset_v1d0_metadata.csv` è il metadata di riferimento. La release contiene già `lista_png.txt`, quindi il metadata CSV non è il producer diretto obbligatorio della pipeline C pubblicata.

## 3. Livelli di ricostruzione

- **LEVEL A — from source dataset.** Input: repository e sorgente raw Gravity Spy. Risultato: ricostruzione scientifica completa.
- **LEVEL B — from preprocessed data.** Input: `out272_gray` e/o tensori centrali. Salta il preprocessing raw.
- **LEVEL C — from checkpoints.** Input: dati preprocessati e checkpoint VAE/IIC selezionati. Salta il training corrispondente.
- **LEVEL D — from derived representations.** Input: `mu`, `H` oppure `affinity`. Consente l'esecuzione downstream dal punto scelto.
- **LEVEL E — result verification.** Input: partizione, payload/checksum di valutazione e metadata inclusi. Consente la verifica di identità, metriche e provenance senza retraining.

## 4. Grafo delle dipendenze

```text
Gravity Spy raw
  -> out272_gray
  -> central tensors
  -> VAE checkpoint
  -> mu
  -> IIC checkpoint
  -> H
  -> affinity
  -> spectral partition
  -> post-hoc evaluation
```

Ogni nodo successivo al dataset sorgente è un `PROJECT_GENERATED_ARTIFACT`. Gli artifact sono rigenerabili dai rispettivi predecessori, con i limiti imposti da hardware, runtime e provenance numerica.

## 5. Artifact di preprocessing

I payload seguenti sono esterni; i manifest associati sono `IN_RELEASE`.

| Artifact | File | Forma e layout | Byte | Manifest in release |
|---|---:|---|---:|---|
| `out272_gray` | 34.332 PNG | 272×224 grayscale | 1.144.867.540 | `manifest_events_272.csv`, `manifest_preprocessing_272_only.csv` |
| `tensors_224_central_uint8` | 8.583 | 4×224×224, uint8, channel-major | 1.722.642.432 | `manifest_tensors_224_central.csv` |
| `tensors_112_central_uint8` | 8.583 | 4×112×112, uint8, channel-major | 430.660.608 | `manifest_tensors_112_central.csv` |
| `tensors_75_central_uint8` | 8.583 | 4×75×75, uint8, channel-major | 193.117.500 | `manifest_tensors_75_central.csv` |

## 6. Archivi di trasferimento

Gli archivi sono rappresentazioni di trasferimento/cache opzionali, non requisiti di LEVEL A.

| File | Byte | SHA-256 |
|---|---:|---|
| `sakai224_full_out272_gray.tar` | 1.171.223.040 | `70c766c3652f586c8da4c8cadf134a8ce83cfd07c8cd10545367f380017ddab2` |
| `sakai224_full_central224.tar` | 1.727.038.464 | `0fad37cdf4ea056e2c3abfa2d1be73b9d9bd55e644d54a58013950697cf4d4b6` |
| `transfer_manifest_sha256.txt` | 711 | `59a04c1098ec4465c1a9864e7bc61181c7981e316c8bc78adbf4c18b78da962f` |

## 7. Pipeline mini

`mini75`, `mini112` e `mini224 subset1000` comprendono, ciascuna, bundle di training VAE, checkpoint VAE `BEST`/`LATEST`/`FINAL`, rappresentazioni `mu`, checkpoint IIC canonico, `H`, affinity, partizione spettrale e valutazione post-hoc comune. Sono artifact esterni generati dal progetto, rigenerabili e utili per una riproduzione per stadi; non sono necessari quando si parte dal dataset sorgente. Non sono dichiarati pubblicamente scaricabili.

I checkpoint VAE `BEST` canonici sono:

- `mini75`: `7926f3a2557d7b02a9b429627c622e5c11559c1d5a273d2e8129a68e5c7ab121`;
- `mini112`: `8857090fb1477fafed6b841bb880a06d61acf38899c7ad2e5a9ef60672989a8b`;
- `mini224 subset1000`: `15a1ecd6dd3c6d11aa0a2ac20ea0b6176d08a18c6f582e7349ed6784ab8528f3`.

`BEST`, `LATEST` e `FINAL` sono identità distinte; il dettaglio è nel TSV.

## 8. Catena FULL224

La catena primaria è:

```text
VAE224 -> mu512 -> IIC C=36 / W=250 / K=5
       -> H [8583,180] -> affinity -> spectral -> post-hoc evaluation
```

Il checkpoint VAE ufficiale `sakai224_full_v1/final.pt` è `REFERENCE_ONLY`, misura 198.545.775 byte e ha SHA-256 `2bb6fd2ce2f73f8f9e479ed6969a52c1cdd41c1ae6e2cfc33da687ebba798f71`. Corrisponde all'epoch 99 zero-based, cioè 100 epoche completate; non è dichiarato localmente disponibile.

## 9. FULL224 mu

L'artifact canonico `sakai224_full_mu_epoch100_v1` contiene:

- train: `[6866,512]`, `float32` little-endian row-major, SHA-256 `b620fd9711719a7bd40293b6536ea6feb235cbadb1d1e37dca0e46651332ad5e`;
- validation: `[1717,512]`, stesso formato, SHA-256 `c1692b3e7a4c86a7f4d6d7172117e11083d2bd2394947838a24d1e2b81f27cf9`;
- manifest `SHA256SUMS.txt`: SHA-256 `2d6205547e6a063b49178595de9a24918bd0add1a4586c800e7519bf842f951d`.

Lo stato della verifica preparatoria è `LOCAL_EXTERNAL_VERIFIED`; l'artifact non è distribuito dalla repository Git.

## 10. FULL224 PRIMARY e SENSITIVITY

Il ramo `symmetric_figure` è `PRIMARY`; il ramo `legacy_text` è `SENSITIVITY`. Entrambi usano `C=36`, `W=250`, `K=5`, 8.583 eventi e IIC transduttivo. `PRIMARY` è il risultato scientifico principale; `SENSITIVITY` è un'analisi di sensibilità separata, senza ranking qualitativo tra i due rami.

## 11. Verifica del risultato primario

I valori congelati per il ramo `PRIMARY` sono:

- purity: `0.8264010252825352`;
- NMI: `0.7018325449722438`;
- ARI: `0.4509773588198464`;
- macro Sakai-like specifica del progetto: `0.8513637314759159`.

Il quarto valore è una metrica project-specific ricostruita: non è il 90,9% di Sakai e non implica parità numerica o metrica con Sakai 2022.

## 12. Baseline K-means

`sakai224_full_kmeans_labelblind_v1` è un `BASELINE_DERIVED_ARTIFACT` derivato da `mu512`; non è la pipeline finale IIC/spectral. Le run C `K=22` e `K=36` sono confronti separati, ciascuno con propri artifact di baseline e valutazione, e non sono fuse con il bundle di Notebook05.

## 13. Stato di disponibilità

Il TSV usa queste categorie esatte:

- `IN_RELEASE`: file presente nella repository;
- `LOCAL_EXTERNAL_VERIFIED`: artifact esterno osservato e verificato durante la preparazione;
- `LOCAL_EXTERNAL_UNVERIFIED`: artifact esterno osservato senza identità completa verificata;
- `REMOTE_KNOWN`: sorgente remota nota e identificata;
- `REFERENCE_ONLY`: identità documentata, disponibilità non attestata;
- `MISSING`: artifact atteso ma non osservato.

`LOCAL_EXTERNAL_*` descrive soltanto lo stato della verifica di preparazione della release e non significa che il file sia pubblicamente scaricabile. Il campo distinto `public_distribution` ammette `IN_REPOSITORY`, `PUBLIC_SOURCE`, `NOT_DISTRIBUTED` e `TO_BE_DETERMINED`. In questa release Gravity Spy è `PUBLIC_SOURCE`, i piccoli file inclusi sono `IN_REPOSITORY`, gli artifact pesanti generati dal progetto sono `NOT_DISTRIBUTED` e gli artifact solo referenziati sono `TO_BE_DETERMINED`.

I valori di `public_path_hint` sono contratti di percorso relativo per un eventuale staging; non sono URL e non attestano hosting pubblico.

## 14. Licenze

Il codice e il materiale originale della repository sono distribuiti secondo [LICENSE](LICENSE) (MIT). Gravity Spy resta soggetto a CC BY 4.0, come documentato in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Gli artifact scientifici esterni generati dal progetto non sono automaticamente rilicenziati dalla licenza MIT della repository; un'eventuale distribuzione e la relativa licenza saranno definite separatamente.

## 15. Politica dei checksum

- Singolo file: SHA-256 dei byte raw.
- Checkpoint: SHA-256 del file e, dove disponibile, hash semantico dello stato.
- File `.npy`: SHA-256 del file e, dove documentato, SHA-256 del contenuto dell'array.
- Bundle: manifest di checksum per-file.
- Directory di grandi dimensioni: conteggio file, conteggio byte e manifest strutturale.
- Archivio di trasferimento: SHA-256 dell'archivio.

Non viene definito né inventato un hash di directory. Il catalogo checksum può riferire file non presenti nella repository: dopo il loro collocamento nel `public_path_hint`, il digest consente di verificarne l'identità.
