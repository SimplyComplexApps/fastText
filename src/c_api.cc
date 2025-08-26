#include "c_api.h"
#include "fasttext.h"
#include <string>
#include <vector>
#include <sstream>

extern "C" {

struct fasttext_args_t {
    fasttext::Args* args;
};

struct fasttext_t {
    fasttext::FastText* ft;
};

fasttext_args_t* fasttext_args_new() {
    fasttext_args_t* c_args = new fasttext_args_t();
    c_args->args = new fasttext::Args();
    return c_args;
}

void fasttext_args_delete(fasttext_args_t* c_args) {
    delete c_args->args;
    delete c_args;
}

void fasttext_args_parse(fasttext_args_t* c_args, int argc, char** argv) {
    std::vector<std::string> args_vec(argv, argv + argc);
    c_args->args->parseArgs(args_vec);
}

const char* fasttext_args_get_input(fasttext_args_t* c_args) { return c_args->args->input.c_str(); }
void fasttext_args_set_input(fasttext_args_t* c_args, const char* input) { c_args->args->input = input; }

const char* fasttext_args_get_output(fasttext_args_t* c_args) { return c_args->args->output.c_str(); }
void fasttext_args_set_output(fasttext_args_t* c_args, const char* output) { c_args->args->output = output; }

double fasttext_args_get_lr(fasttext_args_t* c_args) { return c_args->args->lr; }
void fasttext_args_set_lr(fasttext_args_t* c_args, double lr) { c_args->args->lr = lr; }

int fasttext_args_get_lr_update_rate(fasttext_args_t* c_args) { return c_args->args->lrUpdateRate; }
void fasttext_args_set_lr_update_rate(fasttext_args_t* c_args, int rate) { c_args->args->lrUpdateRate = rate; }

int fasttext_args_get_dim(fasttext_args_t* c_args) { return c_args->args->dim; }
void fasttext_args_set_dim(fasttext_args_t* c_args, int dim) { c_args->args->dim = dim; }

int fasttext_args_get_ws(fasttext_args_t* c_args) { return c_args->args->ws; }
void fasttext_args_set_ws(fasttext_args_t* c_args, int ws) { c_args->args->ws = ws; }

int fasttext_args_get_epoch(fasttext_args_t* c_args) { return c_args->args->epoch; }
void fasttext_args_set_epoch(fasttext_args_t* c_args, int epoch) { c_args->args->epoch = epoch; }

int fasttext_args_get_min_count(fasttext_args_t* c_args) { return c_args->args->minCount; }
void fasttext_args_set_min_count(fasttext_args_t* c_args, int min_count) { c_args->args->minCount = min_count; }

int fasttext_args_get_min_count_label(fasttext_args_t* c_args) { return c_args->args->minCountLabel; }
void fasttext_args_set_min_count_label(fasttext_args_t* c_args, int min_count_label) { c_args->args->minCountLabel = min_count_label; }

int fasttext_args_get_neg(fasttext_args_t* c_args) { return c_args->args->neg; }
void fasttext_args_set_neg(fasttext_args_t* c_args, int neg) { c_args->args->neg = neg; }

int fasttext_args_get_word_ngrams(fasttext_args_t* c_args) { return c_args->args->wordNgrams; }
void fasttext_args_set_word_ngrams(fasttext_args_t* c_args, int word_ngrams) { c_args->args->wordNgrams = word_ngrams; }

int fasttext_args_get_loss(fasttext_args_t* c_args) { return (int)c_args->args->loss; }
void fasttext_args_set_loss(fasttext_args_t* c_args, int loss) { c_args->args->loss = (fasttext::loss_name)loss; }

int fasttext_args_get_model(fasttext_args_t* c_args) { return (int)c_args->args->model; }
void fasttext_args_set_model(fasttext_args_t* c_args, int model) { c_args->args->model = (fasttext::model_name)model; }

int fasttext_args_get_bucket(fasttext_args_t* c_args) { return c_args->args->bucket; }
void fasttext_args_set_bucket(fasttext_args_t* c_args, int bucket) { c_args->args->bucket = bucket; }

int fasttext_args_get_minn(fasttext_args_t* c_args) { return c_args->args->minn; }
void fasttext_args_set_minn(fasttext_args_t* c_args, int minn) { c_args->args->minn = minn; }

int fasttext_args_get_maxn(fasttext_args_t* c_args) { return c_args->args->maxn; }
void fasttext_args_set_maxn(fasttext_args_t* c_args, int maxn) { c_args->args->maxn = maxn; }

int fasttext_args_get_thread(fasttext_args_t* c_args) { return c_args->args->thread; }
void fasttext_args_set_thread(fasttext_args_t* c_args, int thread) { c_args->args->thread = thread; }

double fasttext_args_get_t(fasttext_args_t* c_args) { return c_args->args->t; }
void fasttext_args_set_t(fasttext_args_t* c_args, double t) { c_args->args->t = t; }

const char* fasttext_args_get_label(fasttext_args_t* c_args) { return c_args->args->label.c_str(); }
void fasttext_args_set_label(fasttext_args_t* c_args, const char* label) { c_args->args->label = label; }

int fasttext_args_get_verbose(fasttext_args_t* c_args) { return c_args->args->verbose; }
void fasttext_args_set_verbose(fasttext_args_t* c_args, int verbose) { c_args->args->verbose = verbose; }

const char* fasttext_args_get_pretrained_vectors(fasttext_args_t* c_args) { return c_args->args->pretrainedVectors.c_str(); }
void fasttext_args_set_pretrained_vectors(fasttext_args_t* c_args, const char* pretrained_vectors) { c_args->args->pretrainedVectors = pretrained_vectors; }

int fasttext_args_get_save_output(fasttext_args_t* c_args) { return c_args->args->saveOutput; }
void fasttext_args_set_save_output(fasttext_args_t* c_args, int save_output) { c_args->args->saveOutput = save_output; }

int fasttext_args_get_qout(fasttext_args_t* c_args) { return c_args->args->qout; }
void fasttext_args_set_qout(fasttext_args_t* c_args, int qout) { c_args->args->qout = qout; }

int fasttext_args_get_retrain(fasttext_args_t* c_args) { return c_args->args->retrain; }
void fasttext_args_set_retrain(fasttext_args_t* c_args, int retrain) { c_args->args->retrain = retrain; }

int fasttext_args_get_qnorm(fasttext_args_t* c_args) { return c_args->args->qnorm; }
void fasttext_args_set_qnorm(fasttext_args_t* c_args, int qnorm) { c_args->args->qnorm = qnorm; }

size_t fasttext_args_get_cutoff(fasttext_args_t* c_args) { return c_args->args->cutoff; }
void fasttext_args_set_cutoff(fasttext_args_t* c_args, size_t cutoff) { c_args->args->cutoff = cutoff; }

size_t fasttext_args_get_dsub(fasttext_args_t* c_args) { return c_args->args->dsub; }
void fasttext_args_set_dsub(fasttext_args_t* c_args, size_t dsub) { c_args->args->dsub = dsub; }

fasttext_t* fasttext_new() {
    fasttext_t* c_ft = new fasttext_t();
    c_ft->ft = new fasttext::FastText();
    return c_ft;
}

void fasttext_delete(fasttext_t* c_ft) {
    delete c_ft->ft;
    delete c_ft;
}

void fasttext_load_model(fasttext_t* c_ft, const char* path) {
    c_ft->ft->loadModel(std::string(path));
}

void fasttext_load_model_from_buffer(fasttext_t* c_ft, const void* data, size_t size) {
    c_ft->ft->loadModelFromBuffer(data, size);
}

void fasttext_save_model(fasttext_t* c_ft, const char* path) {
    c_ft->ft->saveModel(std::string(path));
}

int fasttext_get_dimension(fasttext_t* c_ft) {
    return c_ft->ft->getDimension();
}

void fasttext_get_word_vector(fasttext_t* c_ft, const char* word, float* vec) {
    fasttext::Vector v(c_ft->ft->getDimension());
    c_ft->ft->getWordVector(v, std::string(word));
    for (int i = 0; i < v.size(); ++i) {
        vec[i] = v[i];
    }
}

void fasttext_get_sentence_vector(fasttext_t* c_ft, const char* text, float* vec) {
    fasttext::Vector v(c_ft->ft->getDimension());
    std::stringstream ss(text);
    c_ft->ft->getSentenceVector(ss, v);
    for (int i = 0; i < v.size(); ++i) {
        vec[i] = v[i];
    }
}

fasttext_prediction_t* fasttext_predict(fasttext_t* c_ft, const char* text, int32_t k, float threshold, int* n_predictions) {
    std::stringstream ss(text);
    std::vector<std::pair<fasttext::real, std::string>> predictions;
    c_ft->ft->predictLine(ss, predictions, k, threshold);

    *n_predictions = predictions.size();
    fasttext_prediction_t* c_predictions = new fasttext_prediction_t[*n_predictions];

    for (int i = 0; i < *n_predictions; ++i) {
        c_predictions[i].probability = predictions[i].first;
        c_predictions[i].label = new char[predictions[i].second.length() + 1];
        strcpy(c_predictions[i].label, predictions[i].second.c_str());
    }

    return c_predictions;
}

void fasttext_free_predictions(fasttext_prediction_t* predictions, int n_predictions) {
    for (int i = 0; i < n_predictions; ++i) {
        delete[] predictions[i].label;
    }
    delete[] predictions;
}

fasttext::model_name _stringToModel(std::string mn) {
  // Remember this is ok because this is C++, otherwise, strcmp should be used
  if (mn == "cbow") {
    return fasttext::model_name::cbow;
  } else if (mn == "sg") {
    return fasttext::model_name::sg;
  } else if (mn == "sup") {
    return fasttext::model_name::sup;
  }
  return fasttext::model_name::sg; // should never happen
}

void fasttext_train(const char* input, const char* output, const char* model_name, bool retrain, bool qout, int thread) {
    fasttext::Args args = fasttext::Args();
    args.input = std::string(input);
    args.output = std::string(output);
    args.model = _stringToModel(std::string(model_name));
    args.retrain = retrain;
    args.qout = qout;
    args.thread = thread;

    fasttext::FastText ft;
    ft.train(args);
}

}
