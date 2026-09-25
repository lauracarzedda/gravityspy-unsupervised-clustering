# Valutazione post-hoc comune delle partizioni

## Scopo

Le partizioni prodotte dai diversi rami sono valutate mediante un protocollo
post-hoc comune dopo che le assegnazioni sono state fissate. Il protocollo
permette di descrivere in modo uniforme la concordanza fra cluster e classi
note, mantenendo separati apprendimento non supervisionato e valutazione
esterna.

## Label barrier

Le label Gravity Spy sono escluse dal VAE, dalle rappresentazioni latenti
`μ`, dalla IIC, dall'over-clustering, dall'ensemble, dalla rappresentazione
aggregata `H`, dalla matrice di affinity e dal clustering spectral. Sono
introdotte soltanto nella valutazione e nell'interpretazione successive. Le
metriche post-hoc non modificano le partizioni e non determinano selezione
retrospettiva, retraining o reclustering.

## Popolazioni valutate

| Ramo | Ruolo | Popolazione |
|---|---|---:|
| mini75 | PRIMARY | 8583 eventi |
| mini112 | PRIMARY | 8583 eventi |
| full224 `symmetric_figure` / `symmetric_v1` | PRIMARY | 8583 eventi |
| full224 `legacy_text` / `legacy_text_v1` | SENSITIVITY | 8583 eventi |
| mini224_1000 | PRIMARY | 1000 eventi: 800 train e 200 validation nello split VAE |

`mini224_1000` usa una popolazione differente e non costituisce un confronto
a parità di numero di eventi con mini75, mini112 o full224.

## Ruoli full224

Il ramo `symmetric_figure` / `symmetric_v1` è la configurazione PRIMARY. Il
ramo `legacy_text` / `legacy_text_v1` è mantenuto separatamente come
SENSITIVITY e non sostituisce retroattivamente la configurazione primaria
sulla base delle metriche osservate.

## Metriche comuni

Per ciascuna partizione sono considerate la matrice di contingenza, la purity,
la normalized mutual information con media aritmetica, l'adjusted Rand index,
la homogeneity e la completeness. L'associazione fra assegnazioni e label
rispetta l'ordine e l'identità degli eventi definiti per ciascun ramo.

## Natura transduttiva

La IIC e il clustering spectral operano sulla popolazione analizzata secondo
un protocollo transduttivo. Le metriche descrivono quindi la concordanza
post-hoc su tale popolazione e non una capacità di generalizzazione misurata
su un test set indipendente.

## Provenance

Il [record storico della decisione](provenance/project_decision_common_posthoc_evaluation_v1_2026-09-19.md) conserva lo stato del progetto nel momento in cui il protocollo comune fu formalizzato.
