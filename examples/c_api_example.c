#include "../src/c_api.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <model> <text>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* text = argv[2];

    fasttext_t* ft = fasttext_new();
    fasttext_load_model(ft, model_path);

    int dim = fasttext_get_dimension(ft);
    printf("Model dimension: %d\n", dim);

    float* vec = (float*)malloc(dim * sizeof(float));
    fasttext_get_sentence_vector(ft, text, vec);
    printf("Sentence vector: [");
    for (int i = 0; i < dim; ++i) {
        printf("%f", vec[i]);
        if (i < dim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    free(vec);

    int n_predictions;
    fasttext_prediction_t* predictions = fasttext_predict(ft, text, 5, 0.0, &n_predictions);
    printf("Predictions:\n");
    for (int i = 0; i < n_predictions; ++i) {
        printf("  %s (prob: %f)\n", predictions[i].label, predictions[i].probability);
    }
    fasttext_free_predictions(predictions, n_predictions);

    fasttext_delete(ft);

    return 0;
}
