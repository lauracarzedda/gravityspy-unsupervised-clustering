# Specifica dell'implementazione VAE in C11

## Scopo

Questo documento descrive l'implementazione VAE in C11 utilizzata nei
profili mini del progetto e distingue le indicazioni ricavate dal lavoro di
Sakai et al. dalle scelte implementative introdotte nella presente
ricostruzione.

I profili mini sono adattamenti della configurazione di riferimento e non
costituiscono repliche numeriche dell'architettura full224. Il ramo full224 è
un riferimento separato, realizzato in Python/PyTorch. La PCA appartiene alle
baseline lineari e non alla pipeline VAE.

## Architettura di riferimento descritta da Sakai et al.

La Figura 8 di Sakai et al. descrive un encoder che riceve un tensore
`(M, 4, 224, 224)` e produce in sequenza:

| Operazione | Output |
|---|---|
| Convolution + BatchNorm + ReLU | `(M, 64, 112, 112)` |
| Max-Pooling | `(M, 64, 56, 56)` |
| Convolution + BatchNorm + ReLU | `(M, 128, 28, 28)` |
| Convolution + BatchNorm + ReLU | `(M, 256, 14, 14)` |
| Convolution + BatchNorm + ReLU | `(M, 512, 7, 7)` |
| Average-Pooling | `(M, 512, 1, 1)` |
| Fully-Connected | `(M, 1024)` |
| Media | `(M, 512)` |
| Varianza | `(M, 512)` |

Il decoder parte da un vettore latente di dimensione 512. Il fully connected
produce `512 * 7 * 7` valori, che vengono interpretati come un tensore
`(M, 512, 7, 7)`. Seguono due upsampling nearest consecutivi, `7 -> 14` e
`14 -> 28`, una convolution con 256 canali, quindi le sequenze upsampling a
56 e convolution a 128 canali, upsampling a 112 e convolution a 64 canali,
upsampling a 224 e convolution finale a 4 canali.

Il reshape dopo il fully connected è una deduzione implementativa necessaria
per ottenere l'ingresso spaziale del decoder; non è presentato come layer
autonomo nella figura. La figura non specifica in modo completo kernel,
stride, padding, bias, inizializzazione e semantica della Batch Normalization.
Questi dettagli, quando indicati nelle sezioni seguenti, appartengono
all'implementazione del progetto e non sono attribuiti agli autori.

## Scelte implementative dei profili C11

Il motore mini adotta C11 procedurale senza framework di deep learning. I
tensori, i parametri e i gradienti sono contigui in `float32`, con layout
NCHW. Le convolution sono cross-correlation dirette con kernel `3 x 3` e
padding zero implicito ai bordi: nell'encoder usano stride 2 e padding 1,
mentre nel decoder nascosto usano stride 1 e padding 1.

Dopo la prima convolution dell'encoder è applicato un MaxPool `2 x 2`,
stride 2, senza padding. Le restanti convolution dell'encoder completano il
downsampling. Un Global Average Pooling riduce l'ultima mappa a un valore per
canale; il fully connected successivo produce `2 * latent_dim` valori,
suddivisi in `mu` e `logvar`. Il campionamento latente usa

```text
z = mu + exp(0.5 * logvar) * epsilon
```

Il decoder espande il vettore latente con un fully connected, applica
BatchNorm e ReLU, reinterpreta i valori nella forma spaziale del bottleneck e
usa upsampling nearest verso dimensioni target esplicite. I primi due
upsampling sono consecutivi; le convolution nascoste successive riducono i
canali e sono seguite da BatchNorm e ReLU. La convolution finale produce
quattro mappe di logits.

Le etichette non compaiono nelle API di forward, loss o ottimizzazione e
restano escluse dall'apprendimento. La pipeline VAE non include PCA.

## Configurazioni utilizzate

I valori seguenti provengono dai cataloghi di profilo e dai runner associati.
Il conteggio dei MAC comprende convolution e fully connected per un singolo
forward; non comprende BatchNorm, ReLU, pooling, upsampling, sigmoid o loss.

| Esperimento | Profilo C11 | Input | Canali encoder | Latent | Bottleneck | Parametri addestrabili | MAC/forward | Popolazione VAE |
|---|---|---|---|---:|---|---:|---:|---|
| mini75 | `mini75_experiment_v1` | `4 x 75 x 75` | 16 / 32 / 64 | 32 | `5 x 5` | 106116 | 18356544 | 6866 train + 1717 validation |
| mini112 | `mini112_primary_v1` | `4 x 112 x 112` | 16 / 32 / 64 | 64 | `7 x 7` | 262852 | 39948288 | 6866 train + 1717 validation |
| mini224_1000 | `mini224_bridge_v1` | `4 x 224 x 224` | 16 / 32 / 64 / 128 | 64 | `7 x 7` | 625860 | 220790784 | 800 train + 200 validation |

`mini75_smoke_v1` e `mini112_scaleup_v1`, ancora presenti nei cataloghi,
sono rispettivamente un profilo tecnico di verifica e una configurazione
ausiliaria. Non rappresentano le configurazioni scientifiche principali
riportate nella tabella.

mini224_1000 è un esperimento separato eseguito su 1000 eventi. Mantiene la
risoluzione `224 x 224`, ma usa un modello C11 ridotto e uno spazio latente di
dimensione 64. La popolazione 800/200 è distinta da quella di 8583 eventi dei
rami mini75, mini112 e full224; mini224_1000 non costituisce quindi un
confronto a parità di popolazione con tali rami.

Il riferimento full224 utilizza Python/PyTorch, input `4 x 224 x 224`, canali
64 / 128 / 256 / 512 e spazio latente di dimensione 512. È una ricostruzione
separata della configurazione completa e non fa parte del motore C11 dei
profili mini.

## Batch Normalization

Le convolution nascoste sono seguite da BatchNorm2d per canale, con
statistiche calcolate su `N * H * W`. Il fully connected del decoder è seguito
da BatchNorm1d sulle feature. L'implementazione usa `epsilon = 1e-5`,
`momentum = 0.1`, trasformazione affine gamma/beta e running statistics.

Durante il training la normalizzazione usa la varianza biased del batch,
mentre la running variance è aggiornata con la stima unbiased. La modalità di
valutazione usa running mean e running variance; BatchNorm1d in training
richiede un batch di almeno due eventi. Gamma è inizializzata a 1,
beta e running mean a 0, running variance a 1. La convolution finale dei
logits non ha BatchNorm. Questa semantica è una scelta implementativa del
progetto.

## Uscita e funzione obiettivo

La convolution finale è lineare e restituisce logits, senza BatchNorm o ReLU.
La sigmoid è applicata quando servono probabilità o anteprime; la loss usa
direttamente una BCE-with-logits numericamente stabile. Il termine KL descrive
un posterior gaussiano diagonale parametrizzato da `mu` e `logvar`, con
`beta_KL = 1`.

Ricostruzione e KL sono sommate sulle rispettive dimensioni per ogni campione;
la loss del batch è la media delle loss per campione. La Figura 8 riporta
BatchNorm e ReLU anche sulla riga finale del decoder, mentre questa
implementazione adotta logits lineari per la likelihood Bernoulli. La
differenza è una scelta numerica del progetto, non una correzione della
figura.

## Inizializzazione e numerica

I pesi dei layer nascosti seguiti da ReLU usano l'inizializzazione uniforme
He/Kaiming con `sqrt(6 / fan_in)`. Il fully connected encoder che produce
`2 * latent_dim` e la convolution finale dei logits usano Xavier uniforme con
`sqrt(6 / (fan_in + fan_out))`. I bias sono inizializzati a 0; i parametri
BatchNorm partono da gamma 1 e beta 0.

Parametri, attivazioni e gradienti sono in `float32`. Le loss aggregate e
alcune statistiche diagnostiche sono accumulate in `double`. Il backend di
riferimento implementa la cross-correlation con loop diretti e deterministici.

## Randomness e riproducibilità

I runner utilizzano seed distinti per inizializzazione, campionamento
latente, augmentation e shuffle. Le trasformazioni casuali per evento sono
derivate in modo stateless da dominio, seed di base, epoca,
`manifest_index` e indice di stream, secondo l'operazione considerata. In
questo modo l'esito associato a un evento non dipende dalla composizione del
batch.

Il training usa epsilon gaussiani per il campionamento latente; la validation
stocastica deriva epsilon riproducibili dall'indice di manifest. Gli embedding
di riferimento sono costituiti da `mu`, senza campionamento. Le ricostruzioni
deterministiche usano la modalità latente media, cioè `z = mu`.

## Provenance

La specifica architetturale v1 redatta durante lo sviluppo è preservata nella
sua forma originale: [Specifica architetturale storica v1](provenance/vae_c_architecture_contract_v1.md).
