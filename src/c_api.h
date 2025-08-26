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
void fasttext_args_delete(fasttext_args_t* args);

void fasttext_args_parse(fasttext_args_t* args, int argc, char** argv);

const char* fasttext_args_get_input(fasttext_args_t* args);
void fasttext_args_set_input(fasttext_args_t* args, const char* input);

const char* fasttext_args_get_output(fasttext_args_t* args);
void fasttext_args_set_output(fasttext_args_t* args, const char* output);

double fasttext_args_get_lr(fasttext_args_t* args);
void fasttext_args_set_lr(fasttext_args_t* args, double lr);

int fasttext_args_get_lr_update_rate(fasttext_args_t* args);
void fasttext_args_set_lr_update_rate(fasttext_args_t* args, int rate);

int fasttext_args_get_dim(fasttext_args_t* args);
void fasttext_args_set_dim(fasttext_args_t* args, int dim);

int fasttext_args_get_ws(fasttext_args_t* args);
void fasttext_args_set_ws(fasttext_args_t* args, int ws);

int fasttext_args_get_epoch(fasttext_args_t* args);
void fasttext_args_set_epoch(fasttext_args_t* args, int epoch);

int fasttext_args_get_min_count(fasttext_args_t* args);
void fasttext_args_set_min_count(fasttext_args_t* args, int min_count);

int fasttext_args_get_min_count_label(fasttext_args_t* args);
void fasttext_args_set_min_count_label(fasttext_args_t* args, int min_count_label);

int fasttext_args_get_neg(fasttext_args_t* args);
void fasttext_args_set_neg(fasttext_args_t* args, int neg);

int fasttext_args_get_word_ngrams(fasttext_args_t* args);
void fasttext_args_set_word_ngrams(fasttext_args_t* args, int word_ngrams);

int fasttext_args_get_loss(fasttext_args_t* args);
void fasttext_args_set_loss(fasttext_args_t* args, int loss);

int fasttext_args_get_model(fasttext_args_t* args);
void fasttext_args_set_model(fasttext_args_t* args, int model);

int fasttext_args_get_bucket(fasttext_args_t* args);
void fasttext_args_set_bucket(fasttext_args_t* args, int bucket);

int fasttext_args_get_minn(fasttext_args_t* args);
void fasttext_args_set_minn(fasttext_args_t* args, int minn);

int fasttext_args_get_maxn(fasttext_args_t* args);
void fasttext_args_set_maxn(fasttext_args_t* args, int maxn);

int fasttext_args_get_thread(fasttext_args_t* args);
void fasttext_args_set_thread(fasttext_args_t* args, int thread);

double fasttext_args_get_t(fasttext_args_t* args);
void fasttext_args_set_t(fasttext_args_t* args, double t);

const char* fasttext_args_get_label(fasttext_args_t* args);
void fasttext_args_set_label(fasttext_args_t* args, const char* label);

int fasttext_args_get_verbose(fasttext_args_t* args);
void fasttext_args_set_verbose(fasttext_args_t* args, int verbose);

const char* fasttext_args_get_pretrained_vectors(fasttext_args_t* args);
void fasttext_args_set_pretrained_vectors(fasttext_args_t* args, const char* pretrained_vectors);

int fasttext_args_get_save_output(fasttext_args_t* args);
void fasttext_args_set_save_output(fasttext_args_t* args, int save_output);

int fasttext_args_get_qout(fasttext_args_t* args);
void fasttext_args_set_qout(fasttext_args_t* args, int qout);

int fasttext_args_get_retrain(fasttext_args_t* args);
void fasttext_args_set_retrain(fasttext_args_t* args, int retrain);

int fasttext_args_get_qnorm(fasttext_args_t* args);
void fasttext_args_set_qnorm(fasttext_args_t* args, int qnorm);

size_t fasttext_args_get_cutoff(fasttext_args_t* args);
void fasttext_args_set_cutoff(fasttext_args_t* args, size_t cutoff);

size_t fasttext_args_get_dsub(fasttext_args_t* args);
void fasttext_args_set_dsub(fasttext_args_t* args, size_t dsub);

fasttext_t* fasttext_new();
void fasttext_delete(fasttext_t* ft);

void fasttext_load_model(fasttext_t* ft, const char* path);
void fasttext_load_model_from_buffer(fasttext_t* ft, const void* data, size_t size);
void fasttext_save_model(fasttext_t* ft, const char* path);

int fasttext_get_dimension(fasttext_t* ft);

void fasttext_get_word_vector(fasttext_t* ft, const char* word, float* vec);
void fasttext_get_sentence_vector(fasttext_t* ft, const char* text, float* vec);

typedef struct fasttext_prediction_t {
    float probability;
    char* label;
} fasttext_prediction_t;

fasttext_prediction_t* fasttext_predict(fasttext_t* ft, const char* text, int32_t k, float threshold, int* n_predictions);
void fasttext_free_predictions(fasttext_prediction_t* predictions, int n_predictions);

void fasttext_train(const char* input, const char* output, const char* model_name, bool retrain, bool qout, int thread);

#ifdef __cplusplus
}
#endif

#endif  // FASTTEXT_C_API_H
