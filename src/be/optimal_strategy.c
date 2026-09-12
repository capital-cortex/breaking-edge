// TODO: Switch from entry fees only to fees per trade (entry + exit)
// TODO: Add bar_seconds for fee/swap in %/a
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define NEG_INF (-1e18)

double randomNormal() {
    double sum = 0.0;
    for (int i = 0; i < 12; i++) {
        sum += (double)rand() / RAND_MAX;
    }
    return sum - 6.0;
}

double changeStd(double* arr, int n) {
    if (n < 2) return 0.0;
    double mean = 0.0;
    double m2 = 0.0;
    int count = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i - 1] == 0) continue;
        double x = (arr[i] - arr[i - 1]) / arr[i - 1];
        count++;
        double delta = x - mean;
        mean += delta / count;
        double delta2 = x - mean;
        m2 += delta * delta2;
    }
    if (count < 2) return 0.0;
    return sqrt(m2 / (count - 1));
}

size_t getCsvRowCnt(char* src_path) {
    FILE* file = fopen(src_path, "r");
    if (!file) return -1;
    size_t cnt = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        cnt += ch == '\n';
    }
    fclose(file);
    return cnt - 1; // -1 for header
}

bool readCsv(double* prices, size_t cnt, char* src_path) {
    FILE* file = fopen(src_path, "r");
    if (!file) return false;
    char price_open_str[32], price_close_str[32];
    int result;
    size_t i = 0;
    fscanf(file, "%*[^\n]\n"); // ignore header
    fscanf(file, "%*[^\n]\n"); // ignore first row
    for (; i < cnt - 1; i++) {
        // second column is price_open
        fscanf(file, "%*[^,],%[^,],%*[^,],%*[^,],%[^,],%*[^\n]\n", price_open_str, price_close_str);
        prices[i] = atof(price_open_str);
    }
    prices[i] = atof(price_close_str);
    fclose(file);
    return true;
}

void optimalStrategy(int* strategy, double* prices, size_t n, double fee, double swap, double* dp, int* prev) {
    for (size_t i = 0; i < n * 3; i++) {
        dp[i] = NEG_INF;
    }
    dp[1] = 0.0;  /* t=0, flat */
    for (size_t t = 0; t < n - 1; t++) {
        double price_change = (prices[t + 1] - prices[t]) / prices[t];
        for (int s = 0; s < 3; s++) {
            double curr = dp[t * 3 + s];
            if (curr < NEG_INF / 2) continue;
            for (int ns = 0; ns < 3; ns++) {
                double val = curr;
                if (ns != 1) {
                    val += (ns - 1) * price_change - swap;
                    if (ns != s)
                        val -= fee;
                }
                if (val > dp[(t + 1) * 3 + ns]) {
                    dp  [(t + 1) * 3 + ns] = val;
                    prev[(t + 1) * 3 + ns] = s;
                }
            }
        }
    }
    /* find best final state */
    int state = 0;
    for (int s = 1; s < 3; s++) {
        if (dp[(n - 1) * 3 + s] > dp[(n - 1) * 3 + state])
            state = s;
    }
    for (size_t t = n - 1; t > 0; t--) {
        strategy[t - 1] = state - 1;
        state = prev[t * 3 + state];
    }
    strategy[n - 1] = 0;
}

void memFree(double* prices, double* noisy_prices, double* probs, double* dp, int* strategy, int* prev) {
    if (prices      ) free(prices      );
    if (noisy_prices) free(noisy_prices);
    if (probs       ) free(probs       );
    if (dp          ) free(dp          );
    if (strategy    ) free(strategy    );
    if (prev        ) free(prev        );
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        printf("Wrong argument count. Should be: `dst_path`, `src_path`, `fee`, `swap`, `noise`, `runs`");
        return -1;
    }
    char*  dst_path =      argv[1] ;
    char*  src_path =      argv[2] ;
    double fee      = atof(argv[3]);
    double swap     = atof(argv[4]);
    double noise    = atof(argv[5]);
    int    runs     = atoi(argv[6]);
    if (runs < 1) {
        printf("Argument 6 `runs` should be at least 1");
        return -1;
    }
    size_t cnt = getCsvRowCnt(src_path);
    if (cnt < 0) {
        printf("Could not read csv row cnt");
        return -1;
    }
    double* prices       = malloc(cnt     * sizeof *prices      );
    double* noisy_prices = malloc(cnt     * sizeof *noisy_prices);
    double* probs        = calloc(cnt * 3,  sizeof *probs       );
    double* dp           = malloc(cnt * 3 * sizeof *dp          );
    int*    strategy     = malloc(cnt     * sizeof *strategy    );
    int*    prev         = malloc(cnt * 3 * sizeof *prev        );
    if (!(prices && noisy_prices && probs && dp && strategy && prev)) {
        printf("Could not alloc memory.");
        memFree(prices, noisy_prices, probs, dp, strategy, prev);
        return -1;
    }
    if (!readCsv(prices, cnt, src_path)) {
        printf("Could not read csv");
        memFree(prices, noisy_prices, probs, dp, strategy, prev);
        return -1;
    }
    // get std of price change
    double noise_std = changeStd(prices, cnt);
    for (size_t i = 0; i < cnt; i++) {
        prices[i] = log(prices[i]);
    }
    for (int n = 0; n < runs - 1; n++) {
        // add noise
        for (size_t i = 0; i < cnt; i++) {
            noisy_prices[i] = exp(prices[i] + randomNormal() * noise_std * noise);
        }
        // get strategy
        optimalStrategy(strategy, noisy_prices, cnt, fee, swap, dp, prev);
        for (size_t i = 0; i < cnt; i++) {
            probs[i * 3 + strategy[i] + 1]++;
        }
    }
    // return strategy without noise
    for (size_t i = 0; i < cnt; i++) {
        noisy_prices[i] = exp(prices[i]);
    }
    optimalStrategy(strategy, noisy_prices, cnt, fee, swap, dp, prev);
    for (size_t i = 0; i < cnt; i++) {
        probs[i * 3 + strategy[i] + 1]++;
    }
    // write strategy
    FILE* file = fopen(dst_path, "w");
    for (size_t i = 0; i < cnt; i++) {
        fprintf(file, "%i, %f, %f, %f\n",
            strategy[i],
            probs[i * 3 + 0] / runs,
            probs[i * 3 + 1] / runs,
            probs[i * 3 + 2] / runs
        );
    }
    fclose(file);
    memFree(prices, noisy_prices, probs, dp, strategy, prev);

    return 0;
}
