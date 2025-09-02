#include "c_api.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "fasttext.h"

extern "C" {
    /* --- Args --- */

    struct fasttext_args_t {
        fasttext::Args* args;
    };

    fasttext_args_t* fasttext_args_new() {
        auto* c_args = new fasttext_args_t();
        c_args->args = new fasttext::Args();
        return c_args;
    }

    void fasttext_args_delete(const fasttext_args_t* args) {
        delete args->args;
        delete args;
    }

    void fasttext_args_parse(const fasttext_args_t* args, const int argc, char** argv) {
        const std::vector<std::string> args_vec(argv, argv + argc);
        args->args->parseArgs(args_vec);
    }

    const char* fasttext_args_get_input(const fasttext_args_t* args) { return args->args->input.c_str(); }
    void fasttext_args_set_input(const fasttext_args_t* args, const char* input) { args->args->input = input; }

    const char* fasttext_args_get_output(const fasttext_args_t* args) { return args->args->output.c_str(); }
    void fasttext_args_set_output(const fasttext_args_t* args, const char* output) { args->args->output = output; }

    double fasttext_args_get_lr(const fasttext_args_t* args) { return args->args->lr; }
    void fasttext_args_set_lr(const fasttext_args_t* args, const double lr) { args->args->lr = lr; }

    int fasttext_args_get_lr_update_rate(const fasttext_args_t* args) { return args->args->lrUpdateRate; }
    void fasttext_args_set_lr_update_rate(const fasttext_args_t* args, const int rate) { args->args->lrUpdateRate = rate; }

    int fasttext_args_get_dim(const fasttext_args_t* args) { return args->args->dim; }
    void fasttext_args_set_dim(const fasttext_args_t* args, const int dim) { args->args->dim = dim; }

    int fasttext_args_get_ws(const fasttext_args_t* args) { return args->args->ws; }
    void fasttext_args_set_ws(const fasttext_args_t* args, const int ws) { args->args->ws = ws; }

    int fasttext_args_get_epoch(const fasttext_args_t* args) { return args->args->epoch; }
    void fasttext_args_set_epoch(const fasttext_args_t* args, const int epoch) { args->args->epoch = epoch; }

    int fasttext_args_get_min_count(const fasttext_args_t* args) { return args->args->minCount; }
    void fasttext_args_set_min_count(const fasttext_args_t* args, const int min_count) { args->args->minCount = min_count; }

    int fasttext_args_get_min_count_label(const fasttext_args_t* args) { return args->args->minCountLabel; }
    void fasttext_args_set_min_count_label(const fasttext_args_t* args, const int min_count_label) { args->args->minCountLabel = min_count_label; }

    int fasttext_args_get_neg(const fasttext_args_t* args) { return args->args->neg; }
    void fasttext_args_set_neg(const fasttext_args_t* args, const int neg) { args->args->neg = neg; }

    int fasttext_args_get_word_ngrams(const fasttext_args_t* args) { return args->args->wordNgrams; }
    void fasttext_args_set_word_ngrams(const fasttext_args_t* args, const int word_ngrams) { args->args->wordNgrams = word_ngrams; }

    int fasttext_args_get_loss(const fasttext_args_t* args) { return static_cast<int>(args->args->loss); }
    void fasttext_args_set_loss(const fasttext_args_t* args, int loss) { args->args->loss = static_cast<fasttext::loss_name>(loss); }

    int fasttext_args_get_model(const fasttext_args_t* args) { return static_cast<int>(args->args->model); }
    void fasttext_args_set_model(const fasttext_args_t* args, int model) { args->args->model = static_cast<fasttext::model_name>(model); }

    int fasttext_args_get_bucket(const fasttext_args_t* args) { return args->args->bucket; }
    void fasttext_args_set_bucket(const fasttext_args_t* args, const int bucket) { args->args->bucket = bucket; }

    int fasttext_args_get_minn(const fasttext_args_t* args) { return args->args->minn; }
    void fasttext_args_set_minn(const fasttext_args_t* args, const int minn) { args->args->minn = minn; }

    int fasttext_args_get_maxn(const fasttext_args_t* args) { return args->args->maxn; }
    void fasttext_args_set_maxn(const fasttext_args_t* args, const int maxn) { args->args->maxn = maxn; }

    int fasttext_args_get_thread(const fasttext_args_t* args) { return args->args->thread; }
    void fasttext_args_set_thread(const fasttext_args_t* args, const int thread) { args->args->thread = thread; }

    double fasttext_args_get_t(const fasttext_args_t* args) { return args->args->t; }
    void fasttext_args_set_t(const fasttext_args_t* args, const double t) { args->args->t = t; }

    const char* fasttext_args_get_label(const fasttext_args_t* args) { return args->args->label.c_str(); }
    void fasttext_args_set_label(const fasttext_args_t* args, const char* label) { args->args->label = label; }

    int fasttext_args_get_verbose(const fasttext_args_t* args) { return args->args->verbose; }
    void fasttext_args_set_verbose(const fasttext_args_t* args, const int verbose) { args->args->verbose = verbose; }

    const char* fasttext_args_get_pretrained_vectors(const fasttext_args_t* args) { return args->args->pretrainedVectors.c_str(); }
    void fasttext_args_set_pretrained_vectors(const fasttext_args_t* args, const char* pretrained_vectors) { args->args->pretrainedVectors = pretrained_vectors; }

    int fasttext_args_get_save_output(const fasttext_args_t* args) { return args->args->saveOutput; }
    void fasttext_args_set_save_output(const fasttext_args_t* args, const int save_output) { args->args->saveOutput = save_output; }

    int fasttext_args_get_qout(const fasttext_args_t* args) { return args->args->qout; }
    void fasttext_args_set_qout(const fasttext_args_t* args, const int qout) { args->args->qout = qout; }

    int fasttext_args_get_retrain(const fasttext_args_t* args) { return args->args->retrain; }
    void fasttext_args_set_retrain(const fasttext_args_t* args, const int retrain) { args->args->retrain = retrain; }

    int fasttext_args_get_qnorm(const fasttext_args_t* args) { return args->args->qnorm; }
    void fasttext_args_set_qnorm(const fasttext_args_t* args, const int qnorm) { args->args->qnorm = qnorm; }

    size_t fasttext_args_get_cutoff(const fasttext_args_t* args) { return args->args->cutoff; }
    void fasttext_args_set_cutoff(const fasttext_args_t* args, const size_t cutoff) { args->args->cutoff = cutoff; }

    size_t fasttext_args_get_dsub(const fasttext_args_t* args) { return args->args->dsub; }
    void fasttext_args_set_dsub(const fasttext_args_t* args, const size_t dsub) { args->args->dsub = dsub; }

    /* --- FastText --- */

    struct fasttext_t {
        fasttext::FastText* ft;
    };

    fasttext_t* fasttext_new() {
        auto* c_ft = new fasttext_t();
        c_ft->ft = new fasttext::FastText();
        return c_ft;
    }

    void fasttext_delete(const fasttext_t* ft) {
        delete ft->ft;
        delete ft;
    }

    void fasttext_load_model(const fasttext_t* ft, const char* path) {
        ft->ft->loadModel(std::string(path));
    }

    void fasttext_load_model_from_buffer(const fasttext_t* ft, const void* data, const size_t size) {
        ft->ft->loadModelFromBuffer(data, size);
    }

    void fasttext_free_float_char_pair(const fasttext_get_float_char_pair_t* nns, const size_t n_nn) {
        for (int i = 0; i < n_nn; ++i) {
            delete[] nns[i].second;
        }
        delete[] nns;
    }

    fasttext_get_float_char_pair_t* fasttext_get_nn(const fasttext_t* ft, const char* word, const int32_t k, size_t* n_neighbors) {
        const std::vector<std::pair<fasttext::real, std::string>> neighbors = ft->ft->getNN(std::string(word), k);

        *n_neighbors = neighbors.size();
        if (*n_neighbors == 0) {
            return nullptr;
        }

        auto* c_neighbors = new fasttext_get_float_char_pair_t[*n_neighbors];
        for (size_t i = 0; i < *n_neighbors; ++i) {
            c_neighbors[i].first = neighbors[i].first;
            c_neighbors[i].second = new char[neighbors[i].second.length() + 1];
            strcpy(c_neighbors[i].second, neighbors[i].second.c_str());
        }
        return c_neighbors;
    }

    fasttext_get_float_char_pair_t* fasttext_get_analogies(
        const fasttext_t* ft,
        const int32_t k,
        const char* wordA,
        const char* wordB,
        const char* wordC,
        size_t* n_analogies
    ) {
        const std::vector<std::pair<fasttext::real, std::string>> analogies = ft->ft->getAnalogies(
            k,
            std::string(wordA),
            std::string(wordB),
            std::string(wordC)
        );

        *n_analogies = analogies.size();
        if (*n_analogies == 0) {
            return nullptr;
        }

        auto* c_analogies = new fasttext_get_float_char_pair_t[*n_analogies];
        for (size_t i = 0; i < *n_analogies; ++i) {
            c_analogies[i].first = analogies[i].first;
            c_analogies[i].second = new char[analogies[i].second.length() + 1];
            strcpy(c_analogies[i].second, analogies[i].second.c_str());
        }
        return c_analogies;
    }

    int32_t fasttext_get_word_id(const fasttext_t *ft, const char *word) {
        return ft->ft->getWordId(std::string(word));
    }

    int32_t fasttext_get_subword_id(const fasttext_t *ft, const char *word) {
        return ft->ft->getSubwordId(std::string(word));
    }

    void fasttext_save_model(const fasttext_t* ft, const char* path) {
        ft->ft->saveModel(std::string(path));
    }

    int fasttext_get_dimension(const fasttext_t* ft) {
        return ft->ft->getDimension();
    }

    void fasttext_get_word_vector(const fasttext_t* ft, const char* word, float* vec) {
        fasttext::Vector v(ft->ft->getDimension());
        ft->ft->getWordVector(v, std::string(word));
        for (int i = 0; i < v.size(); ++i) {
            vec[i] = v[i];
        }
    }

    void fasttext_get_sentence_vector(const fasttext_t* ft, const char* text, float* vec) {
        fasttext::Vector v(ft->ft->getDimension());
        std::stringstream ss(text);
        ft->ft->getSentenceVector(ss, v);
        for (int i = 0; i < v.size(); ++i) {
            vec[i] = v[i];
        }
    }

    fasttext_prediction_t* fasttext_predict(const fasttext_t* ft, const char* text, const int32_t k, const float threshold, size_t* n_predictions) {
        std::stringstream ss(text);
        std::vector<std::pair<fasttext::real, std::string>> predictions;
        ft->ft->predictLine(ss, predictions, k, threshold);

        *n_predictions = predictions.size();
        auto* c_predictions = new fasttext_prediction_t[*n_predictions];

        for (int i = 0; i < *n_predictions; ++i) {
            c_predictions[i].probability = predictions[i].first;
            c_predictions[i].label = new char[predictions[i].second.length() + 1];
            strcpy(c_predictions[i].label, predictions[i].second.c_str());
        }

        return c_predictions;
    }

    void fasttext_free_predictions(const fasttext_prediction_t* predictions, const size_t n_predictions) {
        for (int i = 0; i < n_predictions; ++i) {
            delete[] predictions[i].label;
        }
        delete[] predictions;
    }

    fasttext::model_name _stringToModel(const std::string& mn) {
      // Remember this is ok because this is C++, otherwise, strcmp should be used
      if (mn == "cbow") {
        return fasttext::model_name::cbow;
      }
      if (mn == "sg") {
          return fasttext::model_name::sg;
      }
      if (mn == "sup") {
          return fasttext::model_name::sup;
      }
      return fasttext::model_name::sg; // should never happen
    }

    void fasttext_train(const char* input, const char* output, const char* model_name, const bool retrain, const bool qout, const int thread) {
        auto args = fasttext::Args();
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
