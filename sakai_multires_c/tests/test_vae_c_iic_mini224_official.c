#define main mini224_runner_entry_for_test
#include "../tools/run_vae_c_iic_mini224.c"
#undef main

#define CHECK(test) do { \
    if (!(test)) { \
        fprintf(stderr, "mini224 official state test line %d: %s\n", \
                __LINE__, #test); \
        return 1; \
    } \
} while (0)

static bool fixture_row(FILE *file, unsigned int epoch, bool complete)
{
    size_t index;
    if (fprintf(file, "%u,1.0,8,1000", epoch) < 0) return false;
    for (index = 0u; index < 5u; ++index)
        if (fprintf(file, ",%u", 8u * (epoch + 1u)) < 0) return false;
    if (!complete) return fputs(",0.5", file) >= 0;
    for (index = 0u; index < 15u; ++index)
        if (fputs(",0.5", file) < 0) return false;
    return fputc('\n', file) != EOF;
}

static bool fixture_metrics(const char *path, unsigned int rows,
                            bool duplicate, bool incomplete)
{
    FILE *file = fopen(path, "wb");
    unsigned int index;
    bool ok = file != NULL;
    if (!ok) return false;
    ok = fputs(OFFICIAL_METRICS_HEADER, file) >= 0;
    for (index = 0u; ok && index < rows; ++index)
        ok = fixture_row(file, duplicate ? 0u : index,
                         !(incomplete && index + 1u == rows));
    if (fclose(file) != 0) ok = false;
    return ok;
}

static bool fixture_file(const char *path, const char *text)
{
    FILE *file = fopen(path, "wb");
    bool ok = file != NULL;
    if (!ok) return false;
    ok = fwrite(text, 1u, strlen(text), file) == strlen(text);
    if (fclose(file) != 0) ok = false;
    return ok;
}

static int checkpoint_fixture(const char *directory)
{
    char latest_path[PATH_CAP], final_path[PATH_CAP];
    char p0[VAE_C_SHA256_HEX_SIZE], o0[VAE_C_SHA256_HEX_SIZE];
    char b0[VAE_C_SHA256_HEX_SIZE], c0[VAE_C_SHA256_HEX_SIZE];
    char p1[VAE_C_SHA256_HEX_SIZE], o1[VAE_C_SHA256_HEX_SIZE];
    char b1[VAE_C_SHA256_HEX_SIZE], c1[VAE_C_SHA256_HEX_SIZE];
    VAECIICEnsemble *source = NULL, *loaded = NULL;
    VAECIICCheckpointIdentity identity, wrong;
    VAECIICCheckpointMetadata metadata, observed;
    size_t pair;
    FILE *file;
    int byte;
    CHECK(join(latest_path, directory, "latest_fixture.ckpt"));
    CHECK(join(final_path, directory, "final_fixture.ckpt"));
    source = (VAECIICEnsemble *)malloc(sizeof(*source));
    loaded = (VAECIICEnsemble *)malloc(sizeof(*loaded));
    CHECK(source != NULL && loaded != NULL);
    vae_c_iic_ensemble_init(source);
    vae_c_iic_ensemble_init(loaded);
    CHECK(vae_c_iic_ensemble_allocate_v1(source, 64u, 128u) == VAE_C_OP_OK);
    CHECK(vae_c_iic_ensemble_allocate_v1(loaded, 64u, 128u) == VAE_C_OP_OK);
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair) {
        VAECAdam *adam = &source->optimizers[pair];
        uint64_t step;
        adam->step = 800u;
        adam->beta1_power = 1.0;
        adam->beta2_power = 1.0;
        for (step = 0u; step < 800u; ++step) {
            adam->beta1_power *= adam->beta1;
            adam->beta2_power *= adam->beta2;
        }
        source->pairs[pair].main_head.num_batches_tracked = 1600u;
        source->pairs[pair].overcluster_head.num_batches_tracked = 1600u;
    }
    vae_c_iic_checkpoint_identity_init(&identity);
    identity.input_dim = 64u;
    identity.capacity = 128u;
    identity.batch_size = 128u;
    identity.train_rows = 1000u;
    identity.max_epochs = 100u;
    identity.shuffle_seed = VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        identity.seeds[pair] = VAE_C_IIC_ENSEMBLE_FIRST_SEED + pair;
    (void)strcpy(identity.run_id, "mini224-official-synthetic-only");
    (void)strcpy(identity.base_commit,
                 "b2746bc9e041f3ae3dd98ef8e23a37e9f2db0a67");
    (void)strcpy(identity.producer_state, "SYNTHETIC_TEST_ONLY");
    memset(identity.producer_binary_sha256, 'a', 64u);
    identity.producer_binary_sha256[64] = '\0';
    vae_c_iic_checkpoint_metadata_init(&metadata);
    metadata.kind = VAE_C_IIC_CHECKPOINT_LATEST;
    metadata.last_completed_epoch = 99u;
    CHECK(vae_c_iic_checkpoint_save_atomic(
        latest_path, &identity, &metadata, source) ==
        VAE_C_IIC_CHECKPOINT_OK);
    CHECK(vae_c_iic_checkpoint_load(
        latest_path, &identity, loaded, &observed) ==
        VAE_C_IIC_CHECKPOINT_OK);
    CHECK(observed.kind == VAE_C_IIC_CHECKPOINT_LATEST &&
          observed.last_completed_epoch == 99u);
    CHECK(ensemble_digests(source, p0, o0, b0, c0));
    CHECK(ensemble_digests(loaded, p1, o1, b1, c1));
    CHECK(strcmp(p0, p1) == 0 && strcmp(o0, o1) == 0 &&
          strcmp(b0, b1) == 0 && strcmp(c0, c1) == 0);
    metadata.kind = VAE_C_IIC_CHECKPOINT_FINAL;
    CHECK(vae_c_iic_checkpoint_save_atomic(
        final_path, &identity, &metadata, loaded) ==
        VAE_C_IIC_CHECKPOINT_OK);
    CHECK(vae_c_iic_checkpoint_load(
        final_path, &identity, source, &observed) ==
        VAE_C_IIC_CHECKPOINT_OK);
    CHECK(observed.kind == VAE_C_IIC_CHECKPOINT_FINAL &&
          observed.last_completed_epoch == 99u);
    CHECK(ensemble_digests(source, p1, o1, b1, c1));
    CHECK(strcmp(p0, p1) == 0 && strcmp(o0, o1) == 0 &&
          strcmp(b0, b1) == 0 && strcmp(c0, c1) == 0);
    wrong = identity;
    wrong.train_rows = 1001u;
    CHECK(vae_c_iic_checkpoint_load(
        latest_path, &wrong, loaded, &observed) !=
        VAE_C_IIC_CHECKPOINT_OK);
    file = fopen(latest_path, "r+b");
    CHECK(file != NULL);
    byte = fgetc(file);
    CHECK(byte != EOF && fseek(file, 0, SEEK_SET) == 0 &&
          fputc(byte ^ 1, file) != EOF && fclose(file) == 0);
    CHECK(vae_c_iic_checkpoint_load(
        latest_path, &identity, loaded, &observed) !=
        VAE_C_IIC_CHECKPOINT_OK);
    vae_c_iic_ensemble_free(source);
    vae_c_iic_ensemble_free(loaded);
    free(source);
    free(loaded);
    return 0;
}

int main(int argc, char **argv)
{
    char directory[PATH_CAP], metrics[PATH_CAP], partial[PATH_CAP];
    char config[PATH_CAP], latest[PATH_CAP], final[PATH_CAP];
    char extra[PATH_CAP], before[VAE_C_SHA256_HEX_SIZE];
    char after[VAE_C_SHA256_HEX_SIZE];
    RunInput run;
    Options parsed;
    char *fresh_args[] = {
        "runner", "--official-run", "--repository-root", "C:/fixture",
        "--code-commit", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    };
    char *resume_args[] = {
        "runner", "--official-run", "--resume", "--repository-root",
        "C:/fixture", "--code-commit",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    };
    char *bad_args[] = {
        "runner", "--official-preflight", "--resume", "--repository-root",
        "C:/fixture", "--code-commit",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    };
    uint64_t free_bytes = 0u;
    bool has_latest, has_final;
    int written;
    if (argc != 2 || !official_disk_free(argv[1], &free_bytes))
        return 1;
    CHECK(parse_options(6, fresh_args, &parsed) &&
          parsed.official_run && !parsed.official_resume);
    CHECK(parse_options(7, resume_args, &parsed) &&
          parsed.official_run && parsed.official_resume);
    CHECK(!parse_options(7, bad_args, &parsed));
    CHECK(free_bytes >= UINT64_C(268435456));
    written = snprintf(directory, sizeof(directory),
                       "%s/mini224_official_state_%lu", argv[1],
#ifdef _WIN32
                       (unsigned long)GetCurrentProcessId()
#else
                       (unsigned long)getpid()
#endif
    );
    CHECK(written > 0 && (size_t)written < sizeof(directory));
    CHECK(!path_exists(directory) && official_make_directory(directory));
    CHECK(join(metrics, directory, "metrics.csv"));
    CHECK(join(config, directory, "run_config.json"));
    CHECK(join(latest, directory, "latest.ckpt"));
    CHECK(join(final, directory, "final.ckpt"));
    CHECK(join(extra, directory, "unexpected.tmp"));
    written = snprintf(partial, sizeof(partial), "%s.partial", metrics);
    CHECK(written > 0 && (size_t)written < sizeof(partial));

    CHECK(official_action(false, false, false, false, false, 0u) ==
          OFFICIAL_FRESH);
    CHECK(official_action(false, true, true, false, true, 0u) ==
          OFFICIAL_REJECT);
    CHECK(official_action(true, true, false, false, true, 0u) ==
          OFFICIAL_RESTART_ZERO);
    CHECK(official_action(true, true, true, false, true, 0u) ==
          OFFICIAL_CONTINUE);
    CHECK(official_action(true, true, true, false, true, 98u) ==
          OFFICIAL_CONTINUE);
    CHECK(official_action(true, true, true, false, true, 99u) ==
          OFFICIAL_FINALIZE);
    CHECK(official_action(true, true, true, false, true, 100u) ==
          OFFICIAL_REJECT);
    CHECK(official_action(true, true, true, true, true, 99u) ==
          OFFICIAL_REJECT);
    CHECK(official_action(true, true, true, false, false, 1u) ==
          OFFICIAL_REJECT);
    CHECK(official_action(true, false, true, false, true, 1u) ==
          OFFICIAL_REJECT);
    CHECK(!official_future_epoch("0,0.1\n", 0u));
    CHECK(official_future_epoch("1,incomplete", 0u));
    CHECK(!official_future_epoch("bad", 0u));

    CHECK(fixture_metrics(metrics, 2u, false, false));
    CHECK(official_reconcile_metrics(metrics, true, 0u));
    CHECK(fixture_file(config, "config"));
    memset(&run, 0, sizeof(run));
    CHECK(strlen(directory) < sizeof(run.official_output_dir));
    (void)strcpy(run.official_output_dir, directory);
    CHECK(official_inventory(&run, &has_latest, &has_final));
    CHECK(!has_latest && !has_final);
    CHECK(remove(metrics) == 0);
    CHECK(official_inventory(&run, &has_latest, &has_final));
    CHECK(!has_latest && !has_final);
    CHECK(official_write_atomic(metrics, OFFICIAL_METRICS_HEADER));
    CHECK(fixture_file(latest, "fixture marker"));
    CHECK(official_inventory(&run, &has_latest, &has_final));
    CHECK(has_latest && !has_final);
    CHECK(fixture_file(final, "fixture marker"));
    CHECK(official_inventory(&run, &has_latest, &has_final));
    CHECK(has_latest && has_final);
    CHECK(fixture_file(extra, "fixture marker"));
    CHECK(!official_inventory(&run, &has_latest, &has_final));

    CHECK(fixture_metrics(metrics, 2u, true, false));
    CHECK(vae_c_mu_file_sha256(metrics, before) == VAE_C_MU_OK);
    CHECK(!official_reconcile_metrics(metrics, true, 1u));
    CHECK(vae_c_mu_file_sha256(metrics, after) == VAE_C_MU_OK);
    CHECK(strcmp(before, after) == 0);
    CHECK(fixture_metrics(metrics, 1u, false, true));
    CHECK(!official_reconcile_metrics(metrics, true, 0u));
    CHECK(fixture_metrics(metrics, 2u, false, true));
    CHECK(official_reconcile_metrics(metrics, true, 0u));
    CHECK(fixture_metrics(metrics, 2u, false, false));
    CHECK(official_reconcile_metrics(metrics, false, 0u));
    CHECK(debug_file_equals(metrics, OFFICIAL_METRICS_HEADER));
    CHECK(fixture_file(partial, "collision"));
    CHECK(!official_write_atomic(metrics, OFFICIAL_METRICS_HEADER));
    CHECK(fixture_file(metrics, "wrong header\n"));
    CHECK(!official_reconcile_metrics(metrics, false, 0u));
    CHECK(checkpoint_fixture(directory) == 0);
    puts("MINI224_OFFICIAL_STATE_MACHINE_PASS=true");
    puts("NEURAL_FORWARD_CALLS=0\nOPTIMIZER_STEPS=0");
    return 0;
}
