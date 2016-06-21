

// input n   matrix is n by n
//       numGen  , simul times

// data structure,
// char[n][n]
// char[i][i]  & 1 == 1   : alive
// char[i][i] & 1 == 0 : dead
// char[i][i] & 3 == 1 : next alive
// char[i][i] & 3 == 0 : next alive

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define M(mat,x,y,n)  (*((mat) + (x) * (n) + (y)))

// random init data
void init_mat(char *mat, int n);

// base and the current state , calculate the next state;
void cal_next_state(char *mat, int n);

// shift state
void shift_state(char *mat, int n);

int main(int argc, char *argv[])
{
    int n = 10;
    int numGen = 10;

    // init data strucur
    char *mat = (char*)malloc(sizeof(char) * n * n);
    // init mat;
    init_mat(mat, n);

    for (int i = 0; i < numGen; i++) {
        // base and the current state , calculate the next state;
        cal_next_state(mat, n);
        // shift state
        shift_state(mat, n);
    }
    return 0;
}

void init_mat(char *mat, int n)
{
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j ++) {
            M(mat,i,j,n) = rand() % 2;
        }
    }
}

static inline int alive_now(char *mat, int i, int j, int n) {
    return ((M(mat, i, j, n) & 1)  == 1);
}

static inline int alive_ext(char *mat, int i, int j, int n) {
    if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
        return 0;
    else
        return alive_now(mat, i, j, n);
}

static inline int dead_now(char *mat, int i, int j, int n) {
    return !alive_now(mat, i, j, n);
}

static inline int count_neighbor_alive(char *mat, int i, int j, int n) {
    return
        alive_ext(mat, i-1, j-1, n)
        + alive_ext(mat, i-1, j, n)
        + alive_ext(mat, i-1, j+1, n)
        + alive_ext(mat, i, j-1, n)
        + alive_ext(mat, i, j+1, n)
        + alive_ext(mat, i+1, j-1, n)
        + alive_ext(mat, i+1, j, n)
        + alive_ext(mat, i+1, j+1, n);
}

static inline int alive_will(char *mat, int i, int j, int n)  {
    // alive :its neighbor alive is 2 or 3, alive,    otherwise dead.
    int n_alive = count_neighbor_alive(mat , i , j , n);
    if (alive_now(mat, i, j, n)) {
        if (n_alive == 2 || n_alive == 3)
            return 1;
        else
            return 0;
    } else {
        if (n_alive == 3)
            return 1;
        else
            return 0;
    }
}

// for one cell
void cal_next_state_1(char *mat, int i, int j, int n)
{
    if (alive_will(mat,i,j,n))
        M(mat,i,j,n) |= 0x3;
    else
        M(mat,i,j,n) &= ~0x3;
}

void cal_next_state(char *mat, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cal_next_state_1(mat, i, j, n);
}

void shift_state(char *mat, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M(mat,i,j,n) >>= 1;
}
