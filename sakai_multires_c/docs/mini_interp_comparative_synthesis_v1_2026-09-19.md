# Sintesi comparativa morfologica dei tre profili mini e del riferimento full224

## Stato del documento

- Data: 2026-09-19
- Versione: 1.0
- Repository: `gravityspy_preprocessing_official`
- Branch: `vae-c-sakai-like`
- HEAD di redazione: `219c5a0ebede1cf61fc42e445a07cfea78b440c5`
- Stato: sintesi interpretativa post-hoc delle partizioni congelate; nessun nuovo fit, ranking, rendering o calcolo di metriche

Il report di audit High precedente aveva concluso con il verdetto `PASS_WITH_REQUIRED_REWORDING`. Il presente documento incorpora le riformulazioni obbligatorie verificate nell'audit Ultra MINI-INTERP-3U. I report precedenti restano invariati come documentazione storica e non vengono sostituiti o cancellati.

## Fonti effettivamente utilizzate

La sintesi usa esclusivamente risultati, tabelle e revisioni visive già prodotti e congelati:

1. le evaluation comuni ufficiali pubblicate da `evaluate_vae_c_iic_spectral_common`, commit produttore `b4ddd83c46fa9609d72a1908fd860ac4b7c27a2b`, sorgente SHA-256 `83a9b3556f29e5afdeb744999cb648dea6f8b75696c9208133e3d36b91d9bca2` e binario SHA-256 `2ee84331c8d6cb41b4a61007f19afa5bcd543d23672e6dd5455719fe530ffd55`;
2. le letture morfologiche complete MINI-INTERP-2A, MINI-INTERP-2B e MINI-INTERP-2C, basate sulle 36 contact sheet ufficiali di ciascun profilo mini;
3. il report comparativo MINI-INTERP-3 e l'audit scientifico Ultra MINI-INTERP-3U;
4. la revisione post-hoc full224 PRIMARY già completata, insieme al recap locale `recap15_3_notebook06_ultra_dettagliato_full224_finale_2026-09-04(4).pdf`, SHA-256 `af41807d38fcb3733eec09ea93e704d527696a005de4043ad329a552cfd92c6a`;
5. lo snapshot locale del paper Sakai, SHA-256 `ed87489ee24029e0d43177b7475bc0518b03676faef451b100f0d77daac41ec2`;
6. le decisioni PROJECT versionate sul backend spectral e sulla valutazione post-hoc comune.

Le evaluation ufficiali sono sotto `sakai_multires_c/results/vae_c/iic/common_posthoc_evaluation_v1`; le contact sheet mini esaminate sono sotto `sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1`.

Le copie esportate dei tre report morfologici mini esaminate nell'audit avevano rispettivamente SHA-256 `b25ef8f3e2e561d3ba44fa77b84d270ce4cdb5ee88060b13d0bf6a377b825912`, `c660aea20304d7e1beba8e4864076a5340c66b2f1d6d19f8ef750cb9d1d021c7` e `803dd7f2e3f1a1876190c0c627e5de4bd9682ed1b77b0d143cf01df028d15956` per mini75, mini112 e mini224_1000.

### Identità delle evaluation ufficiali

| Profilo | Popolazione | Identità partizione | SHA-256 assignment | SHA-256 row-map | SHA-256 `sha256.txt` evaluation |
|---|---:|---|---|---|---|
| mini75 PRIMARY | 8583 | `mini75_spectral_primary_v1` | `cc9e47a9d1a63c245bb53393c0fb65aa014015a95bbc395f3dc0f41554879ffd` | `6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642` | `b52a4fe8924f8d7e312b7890551cd84cee8a7fcd3f1d4dfbfbd732fc5b7fcc40` |
| mini112 PRIMARY | 8583 | `mini112_spectral_primary_v1` | `17200b8d006962e39a7b49b07030042a0980970f57085b6318165cc30e738d3f` | `6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642` | `0cc4e618a477a70e26d11508b20a4de021e8996685aa652ca0e2f07ddb501ae0` |
| mini224_1000 PRIMARY | 1000 | `mini224_1000_spectral_primary_v1` | `61bcc7a38b49c210626bd5fcf191f7b9171017df85fc7a1373fd674001af75d2` | `d7f5101cfe3f3ae31f24b12833f0125873888ff9ed33c4c33c428b90955db4c2` | `d76a99859ca3f34c942612542a6afa789c69a43421324b4a1b303c0e6d1d2f4e` |
| full224 PRIMARY | 8583 | `sakai224_full_iic_symmetric_v1` | `5a01318f703441bbf291f7b32478043f6d0376dd81265078d6dbeac43e306ac0` | `4aa38fd5c51e0fd1f6b91f299c99dfc16ebe0b8c4d7146484f6ea291ebb6c9e5` | `37c18fde4989f72ee75ce0cd3588d0fbe3bf535dd5ca8f38fdb11007602f2340` |

Tutte le evaluation usano 22 label e 36 cluster canonici non riallineati. Il manifest label canonico ha SHA-256 `620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82`. Le label sono state usate soltanto dopo il congelamento delle partizioni.

## Metriche ufficiali

| Profilo | N | Purity pesata | NMI arithmetic | ARI | Homogeneity | Completeness |
|---|---:|---:|---:|---:|---:|---:|
| mini75 PRIMARY | 8583 | 0.7715251078 | 0.6648044582 | 0.3041814500 | 0.7499852941 | 0.5969992667 |
| mini112 PRIMARY | 8583 | 0.8035651870 | 0.6863822831 | 0.3444911329 | 0.7817697201 | 0.6117408733 |
| full224 PRIMARY | 8583 | 0.8264010253 | 0.7018325450 | 0.4509773588 | 0.7836699717 | 0.6354712758 |
| mini224_1000 PRIMARY | 1000 | 0.7680000000 | 0.6960400259 | 0.4205965163 | 0.7734746167 | 0.6326988679 |

La purity riportata qui è la purity standard pesata per numerosità del cluster, calcolata dall'evaluator comune. Deve restare distinta sia dalla ricostruzione PROJECT della misura Sakai-like macro, sia dall'accuracy del 90,9% riportata nel PAPER. Le tre quantità non condividono necessariamente definizione, popolazione e protocollo; nessuna uguaglianza numerica o equivalenza metodologica con Sakai è dichiarata.

## Confronto osservazionale sulla popolazione comune di 8583 eventi

Mini75, mini112 e full224 PRIMARY usano gli stessi 8583 eventi nello stesso contesto transduttivo. Questo rende descrittivamente confrontabili le metriche comuni e le composizioni di cluster, ma non isola l'effetto della sola risoluzione. Tra i tre rami cambiano anche capacità del modello, dimensionalità latente, rappresentazione appresa e intera traiettoria di ottimizzazione. Nel seguito, pertanto, gli andamenti sono osservazioni tra pipeline complete e non stime causali dell'effetto del numero di pixel.

Sulla popolazione condivisa, i cinque valori ufficiali crescono nell'ordine mini75, mini112, full224. Le differenze full224 meno mini75 sono circa +0,0549 per purity, +0,0370 per NMI, +0,1468 per ARI, +0,0337 per homogeneity e +0,0385 per completeness. L'ARI presenta il maggiore incremento numerico nella propria scala; questo non autorizza a giudicarlo scientificamente più importante degli incrementi delle altre metriche, perché le metriche misurano proprietà diverse e hanno scale e sensibilità differenti.

Le statistiche globali non descrivono da sole la struttura interna. Un'elevata copertura massima di una label indica che una parte consistente di quella label è raccolta in un singolo cluster; il numero di cluster occupati descrive invece la sua dispersione; il numero di cluster dominati conta dove la label è maggioritaria; la purezza è una proprietà del cluster. Questi quattro concetti non sono intercambiabili.

### Quadro aggregato dei cluster

| Profilo | Cluster puri | Cluster con purezza >= 90% | Cluster con dominante < 50% | Cluster con dominante 50–70% | Label dominanti in almeno un cluster |
|---|---:|---:|---:|---:|---:|
| mini75 | 5 | 21 | 4 | 4 | 16 |
| mini112 | 2 | 21 | 2 | 7 | 16 |
| full224 PRIMARY | 9 | 22 | 3 | 6 | 17 |
| mini224_1000 | 7 | 15 | 8 | 8 | 16 |

I conteggi mini224_1000 sono riportati per completezza, ma non appartengono al confronto a N costante.

### Cluster occupati, cluster dominati e copertura massima

Ogni cella riporta `occupati / dominati / copertura massima %`. La tabella conserva separate le tre nozioni e riguarda le principali famiglie discusse nei report morfologici.

| Label | mini75 | mini112 | full224 PRIMARY | mini224_1000 |
|---|---:|---:|---:|---:|
| Blip | 10 / 7 / 20,44 | 11 / 6 / 25,68 | 8 / 6 / 28,30 | 7 / 4 / 59,60 |
| Koi Fish | 7 / 4 / 32,29 | 8 / 4 / 36,14 | 11 / 4 / 38,80 | 5 / 3 / 46,43 |
| Scattered Light | 5 / 2 / 51,20 | 9 / 4 / 27,45 | 10 / 3 / 43,57 | 5 / 4 / 61,40 |
| Low Frequency Burst | 11 / 2 / 59,97 | 9 / 3 / 46,27 | 11 / 3 / 51,45 | 2 / 1 / 93,98 |
| Violin Mode | 10 / 0 / 53,81 | 9 / 1 / 42,58 | 9 / 3 / 42,58 | 7 / 0 / 37,21 |
| Extremely Loud | 8 / 2 / 66,74 | 10 / 2 / 57,27 | 12 / 3 / 33,48 | 6 / 3 / 65,52 |
| Light Modulation | 10 / 4 / 33,68 | 13 / 3 / 35,08 | 10 / 3 / 30,37 | 6 / 4 / 31,58 |
| Scratchy | 7 / 1 / 42,09 | 8 / 2 / 46,33 | 7 / 2 / 71,19 | 5 / 2 / 73,33 |
| Power Line | 6 / 2 / 61,37 | 8 / 2 / 59,38 | 6 / 2 / 58,06 | 4 / 1 / 92,54 |
| Low Frequency Lines | 9 / 4 / 36,20 | 6 / 2 / 59,38 | 7 / 2 / 44,37 | 3 / 1 / 67,39 |
| Whistle | 13 / 0 / 41,31 | 12 / 0 / 53,77 | 11 / 1 / 36,39 | 12 / 1 / 25,64 |
| Tomte | 3 / 1 / 98,28 | 4 / 1 / 96,55 | 3 / 1 / 95,69 | 4 / 1 / 63,64 |

La tabella mostra perché il solo numero di cluster dominati non misura la frammentazione. Per esempio, Blip occupa 10 cluster in mini75 e 11 in mini112 pur dominandone rispettivamente 7 e 6; Light Modulation occupa 10 e 13 cluster pur dominandone 4 e 3. Analogamente, una copertura massima elevata può convivere con presenze minoritarie in altri cluster e non equivale alla purezza del cluster di massima copertura.

## Sintesi morfologica corretta

Le contact sheet mostrano i cinque eventi più vicini alla direzione prototipica del cluster nello spazio delle mu centrali normalizzate. Sono nuclei prototipici selezionati deterministicamente, non un campione casuale dell'intero cluster. Di conseguenza, la coerenza visiva dei top-5 non dimostra l'omogeneità di tutti i membri; composizioni e purezze devono essere lette dalle tabelle complete.

### Blip

La famiglia Blip è distribuita su più cluster in tutti i rami. Nei prototipi full224, i cluster 5, 10 e 26 appaiono prevalentemente stretti e compatti, mentre il cluster 21 mostra eventi relativamente più larghi e intensi. Questa distinzione è una descrizione dei prototipi osservati, non una nuova tassonomia fisica. Nei mini la famiglia conserva più nuclei visivi, ma occupazione, dominanza e copertura non devono essere fuse in un unico concetto di frammentazione.

### Koi Fish e Scattered Light

Koi Fish presenta più cluster prototipicamente coerenti; nel full224 i cluster 9, 28, 33 e 34 sono esempi particolarmente netti, con variazioni interne che restano descrittive. Scattered Light mostra strutture arcuate o diffuse coerenti in diversi nuclei; nel full224 i cluster 16 e 24 risultano molto puliti, mentre il cluster 32 è composizionalmente misto e i suoi top-5 evidenziano soprattutto Air Compressor. Questo caso illustra direttamente il limite dei prototipi: essi possono mettere in risalto una componente visiva senza rappresentare proporzionalmente l'intero cluster.

### Extremely Loud, Low Frequency Burst e famiglie a bassa frequenza

Extremely Loud comprende prototipi con morfologie più nette o di bordo e altri più diffusi o saturi; nel full224 il cluster 29 è visivamente più netto, mentre 14 e 22 offrono esempi più diffusi. Anche qui si tratta di candidati descrittivi, non di sottoclassi fisiche. Low Frequency Burst, Low Frequency Lines e Power Line alternano nuclei compatti e presenze distribuite: le elevate coperture massime osservate per alcune configurazioni non cancellano l'occupazione di altri cluster né implicano automaticamente maggiore purezza.

### Violin Mode, Whistle e Light Modulation

Violin Mode e Whistle restano spesso distribuite tra cluster dominati da altre label o cluster misti. La loro presenza in numerosi cluster, talvolta senza dominarne alcuno, segnala una separazione incompleta rispetto alle altre morfologie nella rappresentazione appresa; non consente però di inferire una causa fisica. Light Modulation mostra più nuclei e ampia occupazione in tutti i profili, con un rapporto tra dispersione e dominanza che varia tra le pipeline.

### Scratchy e Tomte

Scratchy raggiunge coperture massime elevate in full224 e mini224_1000, mantenendo comunque presenze in più cluster. Tomte è fortemente concentrata nel cluster di massima copertura in mini75, mini112 e full224. Il dato è riportato come proprietà osservata di queste partizioni; non viene formulato alcun superlativo di stabilità, perché la stabilità tra fit o campioni richiederebbe un disegno sperimentale diverso.

### Coerenza visiva globale delle contact sheet mini

La classificazione descrittiva delle 36 schede ha contato 29 `VISUALLY_COHERENT`, 7 `MODERATELY_COHERENT` e 0 `VISUALLY_HETEROGENEOUS` per mini75; 30, 5 e 1 per mini112; 21, 8 e 7 per mini224_1000. Questi conteggi riguardano i soli top-5 prototipici e non sono misure sostitutive della composizione completa. Il diverso quadro mini224_1000 non può essere attribuito alla sola risoluzione, perché cambia anche la popolazione.

## Mini224_1000: eccezione di popolazione

Mini224_1000 comprende 1000 eventi, ordinati come 800 train seguiti da 200 validation, e usa l'intera popolazione del subset in modo transduttivo per IIC e spectral. Non è un test indipendente sullo split validation. I suoi valori ufficiali e le sue 36 contact sheet sono scientificamente validi per quella popolazione, ma non sono un quarto punto del confronto a N=8583.

Nel subset, i cluster contengono da 6 a 98 eventi; 7 cluster sono puri, 15 raggiungono almeno il 90% di purezza, 8 hanno una label dominante sotto il 50% e 16 label dominano almeno un cluster. La copertura massima può essere elevata per label poco numerose o concentrate, ma deve essere interpretata insieme al numero di cluster occupati e alla dimensione del subset. Non vengono introdotti nuovi fit matched a 1000 eventi per mini75, mini112 o full224.

## Risultati principali

1. Sui medesimi 8583 eventi, le metriche comuni osservate crescono da mini75 a mini112 a full224; il risultato descrive tre pipeline complete e non identifica un effetto causale della risoluzione.
2. L'ARI mostra il maggiore incremento numerico tra mini75 e full224 nella propria scala, senza che ciò autorizzi confronti di importanza scientifica tra delta di metriche diverse.
3. I valori globali devono essere integrati con occupazione, dominanza, copertura e purezza: nessuna di queste quantità è un sinonimo delle altre.
4. Blip e Koi Fish formano più nuclei prototipici in tutti i rami; nel full224 la lettura corretta dei prototipi Blip distingue il cluster 21, relativamente più largo e intenso, dai cluster 5, 10 e 26, prevalentemente stretti e compatti.
5. Violin Mode e Whistle mostrano una separazione più distribuita e spesso mista; il dato è descrittivo della rappresentazione, non una conclusione fisica.
6. Tomte presenta una forte copertura massima nei tre rami da 8583 eventi, mentre Scratchy raggiunge una copertura particolarmente alta nel full224; queste osservazioni non dimostrano stabilità tra esperimenti.
7. I top-5 sono strumenti di interpretazione dei nuclei prototipici. Le tabelle complete, non le sole immagini, definiscono composizione e frammentazione dei cluster.
8. Mini224_1000 deve restare in una sezione separata: popolazione, capacità e rappresentazione differiscono, quindi non si usa per ordinare causalmente i profili per risoluzione.

## Paragrafi utilizzabili nella tesi

Sulla popolazione transduttiva comune di 8583 eventi, le partizioni mini75, mini112 e full224 PRIMARY mostrano un incremento osservazionale di purity pesata, NMI, ARI, homogeneity e completeness. La condivisione della popolazione rende possibile un confronto descrittivo uniforme mediante lo stesso evaluator, ma non consente di attribuire le differenze alla sola risoluzione delle immagini: cambiano anche capacità del modello, dimensionalità latente e rappresentazione appresa. I risultati vanno quindi presentati come confronto fra pipeline complete congelate.

L'analisi per label evidenzia che copertura, occupazione, dominanza e purezza descrivono aspetti distinti. Una label può occupare molti cluster pur dominandone pochi, oppure avere un cluster di alta copertura e contemporaneamente presenze minoritarie altrove. Blip, Light Modulation, Violin Mode e Whistle rendono particolarmente visibile questa distinzione. Per questo motivo la frammentazione non viene dedotta dal solo numero di cluster dominati né dalla sola copertura massima.

Le contact sheet confermano la presenza di nuclei morfologici interpretabili, ma rappresentano i cinque eventi più simili al prototipo costruito nello spazio delle mu centrali normalizzate. Non sono campioni casuali dei cluster. Nel full224, ad esempio, i prototipi Blip dei cluster 5, 10 e 26 sono prevalentemente stretti e compatti, mentre quelli del cluster 21 sono relativamente più larghi e intensi; questa osservazione resta una descrizione visiva e non definisce nuove sottoclassi fisiche. La composizione completa dei cluster è valutata mediante le contingenze ufficiali.

Mini224_1000 costituisce un esperimento separato su 1000 eventi, comprendente insieme gli 800 elementi train e i 200 validation. Le sue metriche e morfologie non sono confrontate come se provenissero dalla stessa popolazione degli altri tre rami e lo split non è interpretato come test indipendente dell'IIC transduttivo. Non sono richiesti né prodotti nuovi fit matched a 1000 eventi per mini75, mini112 o full224.

## Relazione con Sakai e distinzione delle metriche

Il confronto con Sakai riguarda la sequenza scientifica generale e le scelte dichiarate del PROJECT, non un'identità numerica. In particolare:

- la purity standard ufficiale è pesata per la numerosità dei cluster;
- l'indicatore Sakai-like macro ricostruito nel PROJECT è una quantità separata;
- il 90,9% del PAPER è riportato nel contesto e nel protocollo originali del lavoro di Sakai.

Questi numeri non devono essere accostati come stime equivalenti della stessa grandezza. Analogamente, le configurazioni locali non dimostrano le sottoclassi fisiche discusse nel PAPER: eventuali somiglianze morfologiche restano ipotesi interpretative prudenti.

## Limiti e lavoro interpretativo residuo

- Le differenze tra mini75, mini112 e full224 sono confondenti rispetto a risoluzione, capacità, latent dimension e training; non supportano inferenze causali isolate.
- Mini224_1000 usa una popolazione differente e non entra nel confronto quantitativo a parità di N.
- Cluster ID uguali tra rami non implicano allineamento semantico.
- Le label Gravity Spy sono impiegate post-hoc e possono contenere ambiguità o classi morfologicamente sovrapposte.
- Le contact sheet privilegiano i prototipi centrali e possono nascondere code, sottogruppi o contaminazioni visibili soltanto nelle contingenze complete.
- Le categorie di coerenza visiva sono descrittive e non costituiscono una metrica addestrata o un test statistico.
- La variazione fra seed, fit o campioni non è stimata da queste singole partizioni congelate.
- Nessuna nuova sottoclasse fisica è dichiarata e nessuna equivalenza numerica con Sakai è rivendicata.
- La discussione t-SNE completa del full224 non è completata né sostituita da questo documento; resta un'attività interpretativa separata.

## Conclusione

Le evidenze congelate supportano una lettura coerente: sulla popolazione condivisa, full224 ottiene i valori globali più alti, mini112 si colloca fra full224 e mini75 e numerose famiglie conservano nuclei prototipici riconoscibili. Questa regolarità non riduce però la struttura dei risultati a una graduatoria di risoluzioni. Le differenze di popolazione, rappresentazione, capacità e protocollo impongono una formulazione descrittiva e prudente. Mini224_1000 rimane un ramo scientificamente valido ma separato; top-5, metriche globali e tabelle di composizione vengono usati con ruoli distinti e complementari.
