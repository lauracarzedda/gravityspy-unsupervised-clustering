# Decisione di progetto v1 sul backend spectral dei mini

Data: 19 settembre 2026

Versione: 1.0

Autorità della decisione: Laura

Stato: ratificata e documentata separatamente

## 1. Scopo e prevalenza

Questa decisione aggiorna esclusivamente il requisito PROJECT relativo al backend del clustering spectral per i profili mini75, mini112 e mini224_1000. Non modifica l'addendum originale, il PAPER Sakai o gli artefatti scientifici già congelati.

Ai sensi della clausola di prevalenza dell'addendum, questa è la revisione esplicita e versionata successiva che sostituisce soltanto il requisito incompatibile indicato nelle sezioni 3 e 5 dell'addendum. Tutte le altre prescrizioni dell'addendum restano in vigore.

Fonte aggiornata:

- `addendum_vincolante_roadmap_sakai_full_e_mini_c11.docx`
- SHA-256: `c39298fd267a5994357038aaa0b4e7c4d1a540c4c981fc6e33277cdddda012ee`

## 2. Requisito precedente

L'addendum stabiliva:

- sezione 3, riga `Spectral clustering sui tre mini`: backend `C11 + BLAS/LAPACK`, obbligatorio come ablation;
- sezione 5, punto 7: combinazione degli output e spectral clustering in C11;
- sezione 11: prevalenza dell'addendum salvo revisione successiva esplicita e versionata.

Il requisito C11 era una scelta PROJECT. Il PAPER Sakai non prescrive il linguaggio C11 per il clustering spectral.

## 3. Decisione ratificata

Il backend spectral definitivo per mini75, mini112 e mini224_1000 è il Python reference spectral bridge già implementato, validato e committato.

La pipeline mini deve essere descritta come:

> C11 fino alla matrice di affinità inclusa, seguito dal Python reference spectral bridge.

Restano nel percorso C11:

- VAE;
- estrazione delle medie latenti mu;
- IIC;
- over-clustering;
- ensemble di K = 5 classificatori;
- costruzione di H;
- costruzione della matrice di affinità.

Solo il passaggio spectral finale usa il bridge Python di riferimento.

## 4. Motivazione metodologica

La decisione è ratificata perché:

1. il PAPER Sakai non prescrive C11;
2. il requisito C11 era una scelta PROJECT;
3. il ramo sakai224_full usa la spectral chain Python del Notebook06;
4. usare la stessa spectral chain per i mini riduce una sorgente di differenza implementativa e migliora la comparabilità mini-versus-full;
5. la matematica spectral resta invariata;
6. i bridge sono stati validati mediante golden test, subset reali, replay in processi freschi e audit numerici;
7. le partizioni già congelate non richiedono rigenerazione.

Questa decisione non dichiara identità numerica con l'ambiente originale di Sakai o con l'ambiente originario del Notebook06 oltre alle verifiche effettivamente documentate nei rispettivi artefatti e audit.

## 5. Ambito sostituito

Sono sostituite esclusivamente le prescrizioni secondo cui il clustering spectral dei tre mini deve essere implementato ed eseguito in C11 con BLAS/LAPACK.

Quando l'addendum descrive l'intera ablation mini come C11, la formulazione aggiornata deve essere interpretata come C11 fino alla costruzione dell'affinity inclusa, con Python reference spectral bridge per il passaggio spectral finale.

Non sono sostituiti i requisiti relativi a VAE, mu, IIC, over-clustering, K = 5, H, affinity, freeze pre-label, identità degli artefatti o valutazione post-hoc.

## 6. Contratto scientifico invariato

Restano invariati:

- sequenza scientifica VAE -> mu -> IIC -> over-clustering -> K = 5 -> H -> affinity -> spectral;
- classi principali C = 36;
- ramo di over-clustering W = 250;
- costruzione della matrice di affinità `W` nel contratto spectral, distinta dal simbolo `W = 250` usato per l'over-clustering;
- degree `d_i = sum_j W_ij`;
- matrice normalizzata `M = D^(-1/2) W D^(-1/2)`;
- eigensolver `eigsh` e relativo contratto versionato;
- ordinamento e sign-fix degli autovettori;
- normalizzazione L2 per riga dell'embedding spectral;
- assegnazione finale con KMeans a 36 cluster;
- ordine delle righe e identità degli eventi;
- policy label-free fino al congelamento delle partizioni.

## 7. Popolazioni primarie

Le popolazioni primarie restano:

- mini75: 8583 eventi;
- mini112: 8583 eventi;
- sakai224_full: 8583 eventi;
- mini224_1000: 1000 eventi, composti da 800 train e 200 validation, unica eccezione.

Non devono essere introdotti nuovi fit IIC o spectral da 1000 eventi per mini75, mini112 o sakai224_full. I confronti matched subset1000 già esistenti restano baseline separate e non modificano le popolazioni primarie IIC/spectral.

I confronti tra i rami primari devono dichiarare esplicitamente che mini224_1000 usa una popolazione diversa e non costituisce un confronto a parità di dataset.

## 8. Bridge e commit di riferimento

| Profilo | Bridge | Commit produttore |
|---|---|---|
| mini75 | `sakai_multires_c/tools/run_vae_c_iic_spectral_mini75.py` | `3513717d5c52f48fc9b0bb95f167ed74d98067a1` |
| mini112 | `sakai_multires_c/tools/run_vae_c_iic_spectral_mini112.py` | `6da682ede0eda10fbec3e0ff87b76897fceb77fc` |
| mini224_1000 | `sakai_multires_c/tools/run_vae_c_iic_spectral_mini224.py` | `9127789928b4827593cbae1dcbee350ae13151a5` |

I metadata dei singoli artefatti restano l'autorità per le versioni degli ambienti Python, gli SHA-256 dei producer e i parametri completi delle rispettive esecuzioni.

## 9. Impatto sugli artefatti

Le partizioni spectral già congelate dei tre mini restano canoniche. Non devono essere rigenerate, rinominate o riscritte in conseguenza di questa decisione.

La decisione non modifica checkpoint, mu, row-map, IIC, H, affinity, eigensystem, embedding spectral, assegnazioni, manifest o backup esistenti. Non autorizza nuovi fit, training, clustering o evaluation.

## 10. Formulazione per la tesi

La formulazione da usare nella tesi è:

> Per i tre profili mini, la pipeline è implementata in C11 fino alla costruzione della matrice di affinità inclusa. Il clustering spectral finale è eseguito tramite un Python reference bridge che riusa la stessa catena spectral di riferimento adottata per il ramo full224. Questa è una scelta PROJECT dichiarata, non una prescrizione del PAPER, e preserva invariati la matematica spectral e gli artefatti congelati.

Nel confronto dei risultati deve inoltre essere dichiarato che mini75, mini112 e sakai224_full usano 8583 eventi, mentre mini224_1000 usa la popolazione primaria di 1000 eventi.

## 11. Stato della decisione

- `SPECTRAL_IMPLEMENTATION_STRATEGY=PYTHON_REFERENCE_BRIDGE`
- `C11_SPECTRAL_REQUIREMENT_SUPERSEDED_IN_SCOPE=true`
- `FROZEN_PARTITIONS_REMAIN_CANONICAL=true`
- `NEW_SPECTRAL_RUNS_REQUIRED=false`
- `LABEL_OPENING_AUTHORIZED=false`
