# Third-party notices e scope della licenza

La MIT License nella root si applica esclusivamente al materiale originale scritto per questo progetto, salvo ove diversamente indicato. Questo comprende il source, i test, gli strumenti, gli script, le configurazioni e la documentazione originali, oltre al codice sorgente e al Markdown originali dei notebook.

La repository contiene o fa riferimento anche a componenti e contenuti di terze parti, che mantengono i rispettivi termini e le rispettive licenze. Questo documento ne riassume la provenienza e non sostituisce i testi di licenza incorporati nei file upstream.

## stb_image

- File vendorizzato: `stb_image.h`
- Versione: v2.30
- Autore/progetto upstream: Sean Barrett e contributori indicati nel file
- Licenza dichiarata nell'header: MIT License oppure Public Domain
- SHA-256 della copia nella release: `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`

Il testo completo della licenza è incorporato nell'header. Il file conserva la propria licenza upstream e non viene rilicenziato dalla MIT License del progetto.

## stb_image_write

- File vendorizzato: `stb_image_write.h`
- Versione: v1.16
- Autore/progetto upstream: Sean Barrett e contributori indicati nel file
- Attribuzione interna del writer JPEG: Jon Olick, come indicato nell'header
- Licenza dichiarata nell'header: MIT License oppure Public Domain
- SHA-256 della copia nella release: `cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05`

Il testo completo della licenza è incorporato nell'header. Il file conserva la propria licenza upstream e non viene rilicenziato dalla MIT License del progetto.

## Gravity Spy O1 dataset

- Titolo: *Machine learning for Gravity Spy: Glitch classification and dataset*
- Versione: v1.0.0
- DOI: 10.5281/zenodo.1476156
- DOI URL: https://doi.org/10.5281/zenodo.1476156
- Publisher: Zenodo
- Data di pubblicazione: 2018-10-31
- Licenza: Creative Commons Attribution 4.0 International
- Identificatore: CC-BY-4.0
- URL della licenza: https://creativecommons.org/licenses/by/4.0/

Autori, nell'ordine del record:

1. Sara Bahaadini
2. Vahid Noroozi
3. Neda Rohani
4. Scott Coughlin
5. Michael Zevin
6. Joshua Smith
7. Vicky Kalogera
8. Aggelos Katsaggelos

Il dataset Gravity Spy non è redistribuito da questa repository. La MIT License del progetto non rilicenzia il dataset, che resta soggetto alla licenza CC BY 4.0. Restano applicabili i diritti e i requisiti di attribuzione relativi al dataset sorgente.

### Metadata derivati dal dataset

Alcuni file sono generati dal progetto ma derivano dall'identità e dall'organizzazione del dataset Gravity Spy. Rientrano in questa categoria:

- `lista_png.txt`
- `manifest_events_272.csv`
- `manifest_preprocessing_272_only.csv`
- `manifest_tensors_75_central.csv`
- `manifest_tensors_112_central.csv`
- `manifest_tensors_224_central.csv`
- `split_train_indices_seed123.csv`
- `split_val_indices_seed123.csv`
- `sakai_multires_c/configs/subset224_train_800_seed123.csv`
- `sakai_multires_c/configs/subset224_val_200_seed123.csv`

La struttura, gli indici, gli split e le trasformazioni prodotti dal progetto costituiscono contributi del progetto. Identificatori, label e altri elementi provenienti dal dataset Gravity Spy restano tuttavia soggetti ai diritti e ai requisiti di attribuzione applicabili al dataset sorgente CC BY 4.0. Questi file sono quindi metadata misti o derivati: la MIT License del progetto non intende rilicenziare i contenuti di terze parti che vi compaiono.

## Artifact scientifici esterni

La MIT License nella root non si applica automaticamente a dataset o tensori esterni, checkpoint, embedding `μ`, matrici `H`, matrici di affinity, partizioni o artifact spectral e altri risultati scientifici esterni non distribuiti. Il licensing di ciascun artifact dipende dalla sua origine e dalla relativa provenance.

## Riferimenti scientifici

Sakai et al. 2022 è un riferimento scientifico e una fonte metodologica per l'adattamento concettuale della pipeline. Secondo l'audit corrente, la repository non contiene codice Sakai copiato, figure Sakai riprodotte o estratti testuali sostanziali del paper. Il lavoro di Sakai non è una componente software third-party e non viene rilicenziato dalla MIT License del progetto.

## Licenza e citazione

`LICENSE` e questo documento riguardano diritti d'uso, redistribuzione e componenti di terze parti. La citazione accademica del progetto, della tesi, del dataset e dei lavori scientifici è gestita separatamente tramite i metadata di citazione della release.
