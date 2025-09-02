#ifndef FASTTEXT_C_API_H
#define FASTTEXT_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

    #include <stdbool.h>
    #include <stdint.h>
    #include <stddef.h>

    typedef struct fasttext_args_t fasttext_args_t;
    typedef struct fasttext_t fasttext_t;

    fasttext_args_t* fasttext_args_new();
    void fasttext_args_delete(const fasttext_args_t* args);

    void fasttext_args_parse(const fasttext_args_t* args, int argc, char** argv);

    const char* fasttext_args_get_input(const fasttext_args_t* args);
    void fasttext_args_set_input(const fasttext_args_t* args, const char* input);

    const char* fasttext_args_get_output(const fasttext_args_t* args);
    void fasttext_args_set_output(const fasttext_args_t* args, const char* output);

    double fasttext_args_get_lr(const fasttext_args_t* args);
    void fasttext_args_set_lr(const fasttext_args_t* args, double lr);

    int fasttext_args_get_lr_update_rate(const fasttext_args_t* args);
    void fasttext_args_set_lr_update_rate(const fasttext_args_t* args, int rate);

    int fasttext_args_get_dim(const fasttext_args_t* args);
    void fasttext_args_set_dim(const fasttext_args_t* args, int dim);

    int fasttext_args_get_ws(const fasttext_args_t* args);
    void fasttext_args_set_ws(const fasttext_args_t* args, int ws);

    int fasttext_args_get_epoch(const fasttext_args_t* args);
    void fasttext_args_set_epoch(const fasttext_args_t* args, int epoch);

    int fasttext_args_get_min_count(const fasttext_args_t* args);
    void fasttext_args_set_min_count(const fasttext_args_t* args, int min_count);

    int fasttext_args_get_min_count_label(const fasttext_args_t* args);
    void fasttext_args_set_min_count_label(const fasttext_args_t* args, int min_count_label);

    int fasttext_args_get_neg(const fasttext_args_t* args);
    void fasttext_args_set_neg(const fasttext_args_t* args, int neg);

    int fasttext_args_get_word_ngrams(const fasttext_args_t* args);
    void fasttext_args_set_word_ngrams(const fasttext_args_t* args, int word_ngrams);

    int fasttext_args_get_loss(const fasttext_args_t* args);
    void fasttext_args_set_loss(const fasttext_args_t* args, int loss);

    int fasttext_args_get_model(const fasttext_args_t* args);
    void fasttext_args_set_model(const fasttext_args_t* args, int model);

    int fasttext_args_get_bucket(const fasttext_args_t* args);
    void fasttext_args_set_bucket(const fasttext_args_t* args, int bucket);

    int fasttext_args_get_minn(const fasttext_args_t* args);
    void fasttext_args_set_minn(const fasttext_args_t* args, int minn);

    int fasttext_args_get_maxn(const fasttext_args_t* args);
    void fasttext_args_set_maxn(const fasttext_args_t* args, int maxn);

    int fasttext_args_get_thread(const fasttext_args_t* args);
    void fasttext_args_set_thread(const fasttext_args_t* args, int thread);

    double fasttext_args_get_t(const fasttext_args_t* args);
    void fasttext_args_set_t(const fasttext_args_t* args, double t);

    const char* fasttext_args_get_label(const fasttext_args_t* args);
    void fasttext_args_set_label(const fasttext_args_t* args, const char* label);

    int fasttext_args_get_verbose(const fasttext_args_t* args);
    void fasttext_args_set_verbose(const fasttext_args_t* args, int verbose);

    const char* fasttext_args_get_pretrained_vectors(const fasttext_args_t* args);
    void fasttext_args_set_pretrained_vectors(const fasttext_args_t* args, const char* pretrained_vectors);

    int fasttext_args_get_save_output(const fasttext_args_t* args);
    void fasttext_args_set_save_output(const fasttext_args_t* args, int save_output);

    int fasttext_args_get_qout(const fasttext_args_t* args);
    void fasttext_args_set_qout(const fasttext_args_t* args, int qout);

    int fasttext_args_get_retrain(const fasttext_args_t* args);
    void fasttext_args_set_retrain(const fasttext_args_t* args, int retrain);

    int fasttext_args_get_qnorm(const fasttext_args_t* args);
    void fasttext_args_set_qnorm(const fasttext_args_t* args, int qnorm);

    size_t fasttext_args_get_cutoff(const fasttext_args_t* args);
    void fasttext_args_set_cutoff(const fasttext_args_t* args, size_t cutoff);

    size_t fasttext_args_get_dsub(const fasttext_args_t* args);
    void fasttext_args_set_dsub(const fasttext_args_t* args, size_t dsub);

    fasttext_t* fasttext_new();
    void fasttext_delete(const fasttext_t* ft);

    void fasttext_load_model(const fasttext_t* ft, const char* path);
    void fasttext_load_model_from_buffer(const fasttext_t* ft, const void* data, size_t size);

    typedef struct fasttext_get_float_char_pair_t {
        float first;
        char* second;
    } fasttext_get_float_char_pair_t;
    void fasttext_free_float_char_pair(const fasttext_get_float_char_pair_t* nns, size_t n_nn);

    fasttext_get_float_char_pair_t* fasttext_get_nn(const fasttext_t* ft, const char* word, int32_t k, size_t* n_neighbors);
    fasttext_get_float_char_pair_t* fasttext_get_analogies(
        const fasttext_t* ft,
        int32_t k,
        const char* wordA,
        const char* wordB,
        const char* wordC,
        size_t* n_analogies
    );
    int32_t fasttext_get_word_id(const fasttext_t *ft, const char *word);
    int32_t fasttext_get_subword_id(const fasttext_t *ft, const char *word);

    void fasttext_save_model(const fasttext_t* ft, const char* path);

    int fasttext_get_dimension(const fasttext_t* ft);

    void fasttext_get_word_vector(const fasttext_t* ft, const char* word, float* vec);
    void fasttext_get_sentence_vector(const fasttext_t* ft, const char* text, float* vec);

    typedef struct fasttext_prediction_t {
        float probability;
        char* label;
    } fasttext_prediction_t;

    fasttext_prediction_t* fasttext_predict(const fasttext_t* ft, const char* text, int32_t k, float threshold, size_t* n_predictions);
    void fasttext_free_predictions(const fasttext_prediction_t* predictions, size_t n_predictions);

    void fasttext_train(const char* input, const char* output, const char* model_name, bool retrain, bool qout, int thread);

#ifdef __cplusplus
}
#endif

#endif  // FASTTEXT_C_API_H
