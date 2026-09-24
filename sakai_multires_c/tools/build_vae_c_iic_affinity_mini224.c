/* Canonical mini224_1000 source identity; the frozen affinity core is shared. */
#include <string.h>

#define RUN_ID "mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5"
#define H_SHA "134e9a2efd12dfff57c3400fc3680134c83659c0f3314ced2a8468ed1941bea4"
#define ROWS_SHA "d7f5101cfe3f3ae31f24b12833f0125873888ff9ed33c4c33c428b90955db4c2"
#define H_SPEC_SHA "01211e4507045320a6ae47bb868f1cf82e3a81930a964bed69a773e7b8647280"
#define H_MANIFEST_SHA "a53d28504ecbadfe15fb7f17d1a695c6cae29fc21365d48fae1c744bfa9108bd"
#define FINAL_SHA "e98dd0dd6513bc9636fe22af0341c41266bc00ce2629499e3aea2b5e50d83fe4"
#define SOURCE_MODE "canonical_mini224"
#define CANONICAL_ROWS 1000u
#define CANONICAL_ROW_VALID(manifest_index, event_index, split, split_pos, row) \
    ((event_index) == (manifest_index) + 1u && \
     ((row) < 800u ? \
      (strcmp((split), "train") == 0 && (split_pos) == (row)) : \
      (strcmp((split), "validation") == 0 && \
       (split_pos) == (row) - 800u)))

#include "build_vae_c_iic_affinity_mini75.c"
