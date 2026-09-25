# Contratto architetturale v1 del mini-VAE C Sakai-like

## 1. Scopo e stato

Questo documento congela il contratto architetturale v1 del mini-VAE C.
Non costituisce un'implementazione e non modifica le pipeline dati o le
baseline già validate.

Le informazioni sono separate in modo esplicito:

- **Paper**: elementi verificati direttamente nella Figura 8 di Sakai et al.
  2022 o appartenenti alla formulazione VAE descritta nel paper.
- **Progetto**: decisioni operative adottate per il mini-VAE C.
- **Aperto**: estensioni o varianti non ancora approvate.

Mini75 e mini112 sono adattamenti scalati del riferimento, non repliche
numeriche dell'architettura full 224 pubblicata.

## 2. Figura 8: architettura full 224 pubblicata

La trascrizione di questa sezione è stata verificata direttamente sulla
Figura 8 del paper Sakai et al. 2022.

### 2.1 Encoder full 224 - Paper

| Operazione | Output |
|---|---|
| Input | `(M, 4, 224, 224)` |
| Convolution + BatchNorm + ReLU | `(M, 64, 112, 112)` |
| Max-Pooling | `(M, 64, 56, 56)` |
| Convolution + BatchNorm + ReLU | `(M, 128, 28, 28)` |
| Convolution + BatchNorm + ReLU | `(M, 256, 14, 14)` |
| Convolution + BatchNorm + ReLU | `(M, 512, 7, 7)` |
| Average-Pooling | `(M, 512, 1, 1)` |
| Fully-Connected | `(M, 1024)` |
| Separazione: media | `(M, 512)` |
| Separazione: varianza | `(M, 512)` |

La tabella del paper parla di media e varianza. L'impiego di `logvar`
nel mini-VAE C è una decisione del progetto, non una trascrizione letterale
del paper.

### 2.2 Decoder full 224 - Paper

| Operazione | Output |
|---|---|
| Input | `(M, 512)` |
| Fully-Connected + BatchNorm + ReLU | `(M, 512 * 7 * 7)` |
| Reshape implicito necessario all'implementazione, non mostrato come riga separata nella Figura 8 | `(M, 512, 7, 7)` |
| Upsampling-Nearest | `(M, 512, 14, 14)` |
| Upsampling-Nearest | `(M, 512, 28, 28)` |
| Convolution + BatchNorm + ReLU | `(M, 256, 28, 28)` |
| Upsampling-Nearest | `(M, 256, 56, 56)` |
| Convolution + BatchNorm + ReLU | `(M, 128, 56, 56)` |
| Upsampling-Nearest | `(M, 128, 112, 112)` |
| Convolution + BatchNorm + ReLU | `(M, 64, 112, 112)` |
| Upsampling-Nearest | `(M, 64, 224, 224)` |
| Convolution + BatchNorm + ReLU | `(M, 4, 224, 224)` |

La Figura 8 non elenca il reshape come layer autonomo. Il reshape è
dedotto operativamente dall'output fully connected
`(M, 512 * 7 * 7)` e dall'input spaziale necessario al primo upsampling.
Non deve quindi essere presentato come una riga pubblicata letteralmente
dagli autori.

Dopo il fully connected sono presenti due upsampling consecutivi,
`7 -> 14` e `14 -> 28`, prima della convolution `512 -> 256`.
Non deve essere interposta una convolution fra questi due upsampling.

La riga finale della Figura 8 è marcata con BatchNorm e ReLU. Questa
indicazione è registrata come fatto della figura, anche se il percorso
principale del progetto adotta un'uscita diversa, definita nella sezione 8.

### 2.3 Dettagli non specificati dal paper

La Figura 8 non specifica:

- dimensione dei kernel;
- stride;
- padding;
- presenza dei bias;
- inizializzazione dei pesi;
- epsilon e momentum della BatchNorm;
- semantica esatta delle running statistics;
- rappresentazione numerica esatta del parametro di varianza;
- implementazione probabilistica esatta dell'ultimo layer.

Tali dettagli non devono essere attribuiti agli autori.

## 3. Convenzioni comuni mini-VAE C - Progetto

- linguaggio C11 procedurale;
- nessun framework deep-learning esterno;
- tensori contigui `float32`;
- layout NCHW;
- indice logico:

```text
((n * C + c) * H + y) * W + x
```

- convolution implementata come cross-correlation;
- kernel convolution `3 x 3`;
- convolution encoder di downsampling con stride 2 e padding 1;
- MaxPool `2 x 2`, stride 2, padding 0;
- convolution decoder nascosta con stride 1 e padding 1;
- nearest-neighbour upsampling verso una shape target esplicita;
- Global Average Pooling prima dell'uscita encoder;
- encoder FC con output `2 * latent_dim`;
- prima metà dell'output encoder uguale a `mu`;
- seconda metà uguale a `logvar`;
- reparameterization:

```text
z = mu + exp(0.5 * logvar) * epsilon
```

- label escluse dalle API forward, loss e optimizer;
- nessuna PCA nella pipeline VAE.

Kernel, stride, padding, cross-correlation e gestione dell'uscita sono
decisioni del progetto; non sono presentati come specifiche pubblicate.

## 4. Topologia mini75 v1 - Progetto

La profondità encoder è 3.

### 4.1 Encoder

```text
(M, 4, 75, 75)
-> Conv s2 p1:       (M, C1, 38, 38)
-> BatchNorm + ReLU: (M, C1, 38, 38)
-> MaxPool 2 s2:     (M, C1, 19, 19)
-> Conv s2 p1:       (M, C2, 10, 10)
-> BatchNorm + ReLU: (M, C2, 10, 10)
-> Conv s2 p1:       (M, C3, 5, 5)
-> BatchNorm + ReLU: (M, C3, 5, 5)
-> GlobalAvgPool:    (M, C3)
-> FC:               (M, 2 * L)
-> mu:               (M, L)
-> logvar:           (M, L)
```

### 4.2 Decoder

```text
(M, L)
-> FC:               (M, C3 * 5 * 5)
-> BatchNorm + ReLU: (M, C3 * 5 * 5)
-> reshape:          (M, C3, 5, 5)
-> nearest target:   (M, C3, 10, 10)
-> nearest target:   (M, C3, 19, 19)
-> Conv s1 p1:       (M, C2, 19, 19)
-> BatchNorm + ReLU: (M, C2, 19, 19)
-> nearest target:   (M, C2, 38, 38)
-> Conv s1 p1:       (M, C1, 38, 38)
-> BatchNorm + ReLU: (M, C1, 38, 38)
-> nearest target:   (M, C1, 75, 75)
-> Conv s1 p1:       (M, 4, 75, 75) logits
```

## 5. Topologia mini112 v1 - Progetto

La profondità encoder è 3.

### 5.1 Encoder

```text
(M, 4, 112, 112)
-> Conv s2 p1:       (M, C1, 56, 56)
-> BatchNorm + ReLU: (M, C1, 56, 56)
-> MaxPool 2 s2:     (M, C1, 28, 28)
-> Conv s2 p1:       (M, C2, 14, 14)
-> BatchNorm + ReLU: (M, C2, 14, 14)
-> Conv s2 p1:       (M, C3, 7, 7)
-> BatchNorm + ReLU: (M, C3, 7, 7)
-> GlobalAvgPool:    (M, C3)
-> FC:               (M, 2 * L)
-> mu:               (M, L)
-> logvar:           (M, L)
```

### 5.2 Decoder

```text
(M, L)
-> FC:               (M, C3 * 7 * 7)
-> BatchNorm + ReLU: (M, C3 * 7 * 7)
-> reshape:          (M, C3, 7, 7)
-> nearest target:   (M, C3, 14, 14)
-> nearest target:   (M, C3, 28, 28)
-> Conv s1 p1:       (M, C2, 28, 28)
-> BatchNorm + ReLU: (M, C2, 28, 28)
-> nearest target:   (M, C2, 56, 56)
-> Conv s1 p1:       (M, C1, 56, 56)
-> BatchNorm + ReLU: (M, C1, 56, 56)
-> nearest target:   (M, C1, 112, 112)
-> Conv s1 p1:       (M, 4, 112, 112) logits
```

I due upsampling iniziali delle topologie mini conservano il principio
visibile nel decoder della Figura 8.

## 6. Profili congelati v1

Il catalogo machine-readable ufficiale è
`sakai_multires_c/configs/vae_c_profiles_v1.csv`.

| Profilo | Ruolo | Side | Depth | C1/C2/C3/C4 | L | Bottleneck | Parametri BN1d | MAC forward | Stato |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| `mini75_smoke_v1` | `engineering_smoke` | 75 | 3 | 8/16/32/0 | 32 | 5 | 41572 | 5620896 | `primary_smoke` |
| `mini75_experiment_v1` | `optional_full_split_experiment` | 75 | 3 | 16/32/64/0 | 32 | 5 | 106116 | 18356544 | `optional_after_smoke` |
| `mini112_primary_v1` | `primary_c_experiment` | 112 | 3 | 16/32/64/0 | 64 | 7 | 262852 | 39948288 | `primary` |
| `mini112_scaleup_v1` | `optional_scaleup` | 112 | 3 | 32/64/128/0 | 128 | 7 | 1035652 | 141729792 | `optional_after_benchmark` |

`forward_mac_per_sample` include esclusivamente convolution e fully
connected. Un MAC corrisponde a una moltiplicazione più una somma
accumulata. Il conteggio non include BatchNorm, ReLU, pooling, upsampling,
sigmoid, BCE, KL o copie di memoria. Questi valori sono budget
architetturali e non misure di tempo.

`mini75_smoke_v1` serve per primitive, forward/backward, Adam, gradient
check e overfit su 8-32 eventi. È un profilo ingegneristico e non è
automaticamente il risultato scientifico mini75 finale.

`mini112_primary_v1` è il profilo principale del VAE C.
`mini112_scaleup_v1` può essere usato soltanto dopo un benchmark reale
del profilo primario.

I conteggi `trainable_parameters_bn1d` adottano BatchNorm1d su tutte le
feature del fully connected decoder, come definito nella sezione 7.

## 7. Batch Normalization v1 - Progetto

### 7.1 Posizionamento

- Dopo le convolution nascoste: BatchNorm2d per canale, con statistiche
  calcolate su `N * H * W`.
- Dopo il fully connected decoder: BatchNorm1d sulle
  `D = C3 * bottleneck_side * bottleneck_side` feature.
- BatchNorm1d è l'interpretazione di progetto più letterale della riga
  fully connected marcata `B` nella Figura 8.
- La convolution finale dei logits non usa BatchNorm.

### 7.2 Semantica

- `epsilon = 1e-5`;
- `momentum = 0.1`, con convenzione PyTorch:

```text
running_new = (1 - momentum) * running_old
              + momentum * batch_estimate
```

- affine gamma/beta abilitato;
- running statistics abilitate;
- gamma iniziale uguale a 1;
- beta iniziale uguale a 0;
- running mean iniziale uguale a 0;
- running variance iniziale uguale a 1;
- in training, varianza biased per la normalizzazione del batch;
- running variance aggiornata con la stima unbiased;
- in evaluation, uso di running mean e running variance;
- BatchNorm1d in training richiede batch size almeno 2.

Questa semantica è una decisione del progetto per facilitare il confronto
futuro con PyTorch. I suoi valori non sono specificati nel paper.

### 7.3 Bias

- nessun bias nei layer immediatamente seguiti da BatchNorm;
- bias presente nel fully connected encoder `2L`;
- bias presente nella convolution finale dei logits.

## 8. Uscita e loss v1

### 8.1 Scelta principale - Progetto

- convolution finale lineare;
- nessuna BatchNorm finale;
- nessuna ReLU finale;
- output uguale ai logits;
- sigmoid applicata soltanto per ottenere probabilità o PNG;
- BCE-with-logits in forma numericamente stabile;
- KL gaussiana parametrizzata tramite `mu` e `logvar`;
- `beta_KL = 1`;
- nessun KL warm-up iniziale;
- nessun clamp iniziale di `logvar`;
- errore diagnostico e interruzione in presenza di NaN o Inf.

Per ogni campione:

```text
reconstruction_sum =
    somma BCE-with-logits su canali, righe e colonne

KL_sum =
    -0.5 * somma_j(
        1 + logvar_j - mu_j^2 - exp(logvar_j)
    )

loss_sample =
    reconstruction_sum + KL_sum

loss_batch =
    media dei loss_sample sul batch
```

La forma stabile della BCE-with-logits è:

```text
max(logit, 0)
- logit * target
+ log1p(exp(-abs(logit)))
```

Logging futuro obbligatorio:

- reconstruction media per campione;
- reconstruction per pixel;
- KL media per campione;
- KL per dimensione latente;
- loss totale.

### 8.2 Paper e progetto

**Paper**:

- likelihood Bernoulli;
- termine di ricostruzione;
- divergenza KL verso il prior gaussiano.

**Progetto**:

- parametro `logvar`;
- logits e BCE-with-logits;
- riduzione sum per campione e mean sul batch;
- `beta_KL = 1`;
- nessun warm-up iniziale.

### 8.3 Ambiguità dell'ultimo layer

La Figura 8 mostra BatchNorm e ReLU anche sull'ultimo layer decoder,
mentre la formulazione Bernoulli richiede un output interpretabile come
probabilità.

Il percorso principale usa logits lineari e BCE-with-logits per coerenza
matematica e stabilità numerica. Una versione che segua letteralmente
BatchNorm e ReLU finali potrà essere soltanto una futura ablation
separatamente documentata.

## 9. Inizializzazione v1 - Progetto

Hidden convolution e decoder FC seguiti da ReLU:

```text
He/Kaiming uniform, fan_in
bound = sqrt(6 / fan_in)
```

Encoder FC `2L` e convolution finale dei logits:

```text
Xavier uniform
bound = sqrt(6 / (fan_in + fan_out))
```

Inoltre:

- bias iniziali uguali a 0;
- BatchNorm gamma uguale a 1;
- BatchNorm beta uguale a 0;
- seed di inizializzazione separato da ogni altro seed.

Queste inizializzazioni sono decisioni del progetto e non sono attribuite
a Sakai et al.

## 10. Numerica e backend v1 - Progetto

- parametri, attivazioni e gradienti in `float32`;
- accumulo della loss in `double`;
- statistiche diagnostiche in `double`;
- gradient-check di riferimento in `double`;
- prima Conv2D tramite loop diretti e deterministici;
- nessun im2col nella prima versione;
- nessun BLAS nella prima versione;
- nessun OpenMP nella prima versione;
- nessun SIMD esplicito nella prima versione.

Un backend ottimizzato potrà essere aggiunto successivamente dietro la
stessa API. Ogni backend ottimizzato dovrà essere confrontato
numericamente con quello diretto deterministico.

## 11. Randomness e modalità v1 - Progetto

Devono esistere stream o seed separati per:

- inizializzazione;
- shuffle;
- augmentation;
- epsilon latente di training;
- epsilon latente di validation.

Vincoli:

- nessun `rand()` o `srand()` globale nelle nuove API;
- i valori casuali devono poter dipendere da `base_seed`, `epoch` e
  `manifest_index`;
- il risultato per un evento non deve dipendere dalla composizione o
  dall'ordine del batch;
- training: un epsilon per evento e dimensione latente;
- validation loss: epsilon fisso e riproducibile per `manifest_index`,
  indipendente dall'epoca;
- embedding ufficiali: `z = mu`;
- ricostruzioni ufficiali: `z = mu`;
- nessun campione casuale per embedding o ricostruzioni ufficiali.

## 12. Supporto futuro 224

Il futuro modello o configuratore:

- il modello/configuratore C non deve essere hardcoded alla profondità 3;
- deve supportare profondità 3 per mini75 e mini112;
- deve poter supportare in C la profondità 4 per l'eventuale mini224_1000.

Lo stato dei profili 224 è:

- `mini224_1000`: configurazione rinviata fino al benchmark mini112;
- `sakai224_full`: segue concettualmente la profondità 4 e la Figura 8
  completa, con latent 512, ma sarà implementato e addestrato
  separatamente nel progetto PyTorch/Colab;
- il runner C non è obbligato ad addestrare `sakai224_full`;
- il contratto dati resta identico fra C e PyTorch.

Nessun profilo definitivo mini224_1000 è incluso nel contratto v1.

## 13. Ruoli e limiti

- `mini75_smoke_v1` è un profilo ingegneristico.
- `mini75_experiment_v1` è opzionale dopo la chiusura dello smoke.
- `mini112_primary_v1` è il VAE C principale.
- `mini112_scaleup_v1` dipende da benchmark reali.
- mini224_1000 sarà deciso dopo mini112.
- sakai224_full userà la Figura 8 completa e latent 512 su PyTorch/Colab.
- nessuna PCA entra nel VAE.
- le label restano escluse da forward, loss e optimizer.
- gli embedding ufficiali futuri sono `mu`.

## 14. Decisioni non comprese nel contratto v1

Restano fuori dal congelamento corrente:

- configurazione mini224_1000;
- backend ottimizzati;
- parallelizzazione;
- im2col;
- KL warm-up;
- beta-VAE con beta diverso da 1;
- clamp di `logvar`;
- ablation con BatchNorm/ReLU sull'ultimo layer.
