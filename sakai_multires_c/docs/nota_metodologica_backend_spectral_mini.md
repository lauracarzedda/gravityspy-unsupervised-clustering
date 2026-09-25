# Backend spectral dei profili mini

## Implementazione adottata

Per i profili mini75, mini112 e mini224_1000, la pipeline è implementata in
C11 fino alla costruzione della matrice di affinità inclusa. Il clustering
spectral finale utilizza un Python reference bridge e applica la stessa
procedura spectral di riferimento adottata per full224.

## Motivazione metodologica

Il lavoro di Sakai non prescrive C11 per il clustering spectral. L'estensione
del requisito C11 anche alla fase spectral era una scelta iniziale del
progetto; il bridge Python riduce una sorgente di differenza implementativa
nella fase finale e rende più diretto il confronto con full224. Questa scelta
non implica identità numerica con il software o con l'ambiente originale di
Sakai.

## Pipeline scientifica

La sequenza metodologica resta:

```text
VAE
-> μ
-> IIC
-> over-clustering
-> ensemble K = 5
-> H
-> affinity
-> spectral
```

La configurazione utilizza `C = 36` classi principali, `W = 250` unità nel
ramo di over-clustering e `K = 5` classificatori nell'ensemble. Il simbolo
`W` usato per la matrice di affinità resta distinto da `W = 250`.

## Procedura spectral

Quando previsto dalla procedura corrente, la diagonale della matrice di
affinità è esclusa dal grafo. Il degree è calcolato come
`d_i = sum_j W_ij` e l'operatore normalizzato come
`M = D^(-1/2) W D^(-1/2)`. La decomposizione utilizza `eigsh`; gli
autovettori sono ordinati e sottoposti a sign-fix, quindi l'embedding spectral
è normalizzato L2 per riga. L'assegnazione finale usa k-means con 36 cluster.

## Popolazioni

| Ramo | Popolazione |
|---|---:|
| mini75 | 8583 eventi |
| mini112 | 8583 eventi |
| full224 | 8583 eventi |
| mini224_1000 | 1000 eventi |

La popolazione di `mini224_1000` è distinta e non costituisce un confronto a
parità di numero di eventi con gli altri rami.

## Bridge di riferimento

| Profilo | Bridge | Commit produttore storico |
|---|---|---|
| mini75 | `run_vae_c_iic_spectral_mini75.py` | `3513717d5c52f48fc9b0bb95f167ed74d98067a1` |
| mini112 | `run_vae_c_iic_spectral_mini112.py` | `6da682ede0eda10fbec3e0ff87b76897fceb77fc` |
| mini224_1000 | `run_vae_c_iic_spectral_mini224.py` | `9127789928b4827593cbae1dcbee350ae13151a5` |

I metadata dei singoli artefatti specificano versioni dell'ambiente,
checksum dei producer e parametri completi delle rispettive esecuzioni.

## Provenance

Il [record storico della decisione](provenance/project_decision_spectral_backend_v1_2026-09-19.md) conserva la transizione dalla precedente ipotesi di backend C11 con BLAS/LAPACK al bridge Python effettivamente adottato.
