#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

//Here Defining the structure of Logistic Regression
typedef struct {
    int N;
    int n_in;
    int n_out;
    double **W;
    double *b;
} LR;

//Function Prototypes
void LR_construct(LR *this, int N, int n_in, int n_out);
void LR_destruct(LR *this);
void LR_train(LR *this, int *x, int *y, double lr);
void LR_softmax(LR *this, double *x);
void LR_predict(LR *this, int *x, double *y);

// Making the model 
void LR_construct(LR *this, int N, int n_in, int n_out) {
    int i, j;
    this->N = N;
    this->n_in = n_in;
    this->n_out = n_out;

    this->W = (double **)malloc(sizeof(double*) * n_out);
    this->W[0] = (double *)malloc(sizeof(double) * n_in * n_out);
    for(i=0; i<n_out; i++) this->W[i] = this->W[0] + i * n_in;
    this->b = (double *)malloc(sizeof(double) * n_out);

    // Initialize weights and biases randomly
    srand(time(NULL));
    for(i=0; i<n_out; i++) {
        for(j=0; j<n_in; j++) {
            this->W[i][j] = (double)rand() / RAND_MAX;
        }
        this->b[i] = (double)rand() / RAND_MAX;
    }
}

//Freeing up the memory 
void LR_destruct(LR *this) {
    free(this->W[0]);
    free(this->W);
    free(this->b);
}

//Function to train the model 
void LR_train(LR *this, int *x, int *y, double lr) {
    int i, j;
    double *p_y_given_x = (double *)malloc(sizeof(double) * this->n_out);
    double *dy = (double *)malloc(sizeof(double) * this->n_out);

    for (i = 0; i < this->n_out; i++) {
        p_y_given_x[i] = 0;
        for (j = 0; j < this->n_in; j++) {
            p_y_given_x[i] += this->W[i][j] * x[j];
        }
        p_y_given_x[i] += this->b[i];
    }
    LR_softmax(this, p_y_given_x);

    for (i = 0; i < this->n_out; i++) {
        dy[i] = y[i] - p_y_given_x[i];

        for (j = 0; j < this->n_in; j++) {
            this->W[i][j] += lr * dy[i] * x[j] / this->N;
        }

        this->b[i] += lr * dy[i] / this->N;
    }

    free(p_y_given_x);
    free(dy);
}

// Soft Max Function
void LR_softmax(LR *this, double *x) {
    int i;
    double max = 0.0;
    double sum = 0.0;

    for (i = 0; i < this->n_out; i++) {
        if (max < x[i]) max = x[i];
    }
    for (i = 0; i < this->n_out; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }

    for (i = 0; i < this->n_out; i++) { 
        x[i] /= sum; 
    }
}

// Prediction of the output 
void LR_predict(LR *this, int *x, double *y) {
    int i, j;

    for (i = 0; i < this->n_out; i++) {
        y[i] = 0;
        for (j = 0; j < this->n_in; j++) {
            y[i] += this->W[i][j] * x[j];
        }
        y[i] += this->b[i];
    }

    LR_softmax(this, y);
}

// This is just for debuging and reset of the stuff please ignore it 
void shuffle_data(int **train_X, int **train_Y, int train_N, int n_in, int n_out) {
    int i, j, temp;
    for (i = 0; i < train_N; i++) {
        int rand_index = rand() % train_N;
        // Swap train_X[i] and train_X[rand_index]
        for (j = 0; j < n_in; j++) {
            temp = train_X[i][j];
            train_X[i][j] = train_X[rand_index][j];
            train_X[rand_index][j] = temp;
        }
        // Swap train_Y[i] and train_Y[rand_index]
        for (j = 0; j < n_out; j++) {
            temp = train_Y[i][j];
            train_Y[i][j] = train_Y[rand_index][j];
            train_Y[rand_index][j] = temp;
        }
    }
}

// defining the test dataset and training the model
void test_lr(void) {
    int i, j, epoch;
    double learning_rate = 0.01;
    int n_epochs = 250;
    int n_in = 6;  
    int n_out = 2; 

    
    FILE *dataFile = fopen("data.csv", "r");
    if (dataFile == NULL) {
        printf("Error opening data.csv\n");
        return;
    }

    
    int train_N = 0;
    char line[1024];
    fgets(line, sizeof(line), dataFile); 
    while (fgets(line, sizeof(line), dataFile)) {
        train_N++;
    }
    rewind(dataFile); 
 
    
    int **train_X = malloc(train_N * sizeof(int*));
    for (i = 0; i < train_N; i++) {
        train_X[i] = malloc(n_in * sizeof(int));
    }
 
    int **train_Y = malloc(train_N * sizeof(int*));
    for (i = 0; i < train_N; i++) {
        train_Y[i] = malloc(n_out * sizeof(int));
    }

    
    fgets(line, sizeof(line), dataFile);  
    i = 0;
    while (fgets(line, sizeof(line), dataFile)) {
        char *token;
        j = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            if (j >= n_in) { 
                train_Y[i][0] = atoi(token); 
                train_Y[i][1] = 1 - train_Y[i][0]; 
            } else {
                train_X[i][j] = atoi(token);
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(dataFile);

    
    LR classifier;
    LR_construct(&classifier, train_N, n_in, n_out);

    int correct_predictions = 0;

    for(epoch=0; epoch<n_epochs; epoch++) {
        shuffle_data(train_X, train_Y, train_N, n_in, n_out);
        correct_predictions = 0;
        for(i=0; i<train_N; i++) {
            LR_train(&classifier, train_X[i], train_Y[i], learning_rate);
        }

        for(i=0; i<train_N; i++) {
            double predicted_Y[n_out];
            LR_predict(&classifier, train_X[i], predicted_Y);
            int max_index = 0;
            for(j=1; j<n_out; j++) {
                if (predicted_Y[j] > predicted_Y[max_index]) {
                    max_index = j;
                }
            }
            int predicted_class = max_index;
            int true_class = 0;
            for(j=0; j<n_out; j++) {
                if (train_Y[i][j] == 1) {
                    true_class = j;
                    break;
                }
            }
            if (predicted_class == true_class) {
                correct_predictions++;
            }
        }
        double accuracy = (double)correct_predictions / train_N * 100.0;
        printf("Epoch %d, Accuracy: %.2f%%\n", epoch+1, accuracy);
        //just for debuging 
        printf("Epoch %d, Correct Predictions: %d\n", epoch+1, correct_predictions);
        printf("Epoch %d, Total Examples: %d\n", epoch+1, train_N);
    }
    
    
    for (i = 0; i < train_N; i++) {
        free(train_X[i]);
        free(train_Y[i]);
    }
    free(train_X);
    free(train_Y);


    LR_destruct(&classifier);

    
}

//  Main Function
int main(void) {
    test_lr();
    return 0;
}
