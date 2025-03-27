#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#define N 9

typedef struct {
    int board[N][N];
    int valid;
} Sudoku;

void print_board(Sudoku s) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (s.board[i][j] == -1) {
                printf("_ ", s.board[i][j]);
            }
            else {
                printf("%d ", s.board[i][j]);
            }
        }
        printf("\n");
    }
}

bool valid(Sudoku* s, int r, int c) {
    for (int i = 0; i < N; i++) {
        if ((s->board[r][i] == s->board[r][c] && i != c) || (s->board[i][c] == s->board[r][c] && i != r)) return false;
    }

    for (int br = r - r%3; br < r-r%3+3; br++) {
        for (int bc = c - c%3; bc < c-c%3+3; bc++) {
            if (s->board[br][bc] == s->board[r][c] && (br != r || bc != c)) {
                return false;
            }
        }
    }
    return true;
}

Sudoku place(Sudoku curr, int pos) {
	int r = pos / N;
	int c = pos % N;
	if (curr.board[r][c] != -1) return r==N-1&&c==N-1 ? curr : place(curr, pos+1);
	Sudoku ans = {.valid = 0};
#pragma omp taskgroup
	for (int i = 1; i < 10; i++) {
		Sudoku s = curr;
		s.board[r][c] = i;
#pragma omp task firstprivate(s, r, c) shared(ans)	
		if (valid(&s, r, c)) {
			if (r >= N-1 && c >= N-1) {
				ans = s; 
			#pragma omp cancel taskgroup
			}
			Sudoku result = place(s, pos+1);
			if (result.valid) {
				ans = result; 
			#pragma omp cancel taskgroup
			}
		}
	}
	return ans;
}

int main() {
	Sudoku s = { .board = {
                {5, -1, -1, -1, 7, -1, -1, -1, -1},
                {6, -1, -1, 1, -1, 5, -1, -1, -1},
                {-1, 9, -1, -1, -1, -1, -1, 6, -1},
                {8, -1, -1, -1, 6, -1, -1, -1, 3},
                {-1, -1, -1, -1, -1, 3, -1, -1, 1},
                {7, -1, -1, -1, 2, -1, -1, -1, -1},
                {-1, 6, -1, -1, -1, -1, 2, 8, -1},
                {-1, -1, -1, 4, -1, -1, -1, -1, 5},
                {-1, -1, -1, -1, 8, -1, -1, -1, 9}},
                .valid = 1
		};
//	Sudoku s = { .board = {
//		{5, 3, 1},
//		{-1, -1, -1},
//		{2, 4, 8}},
//	       	.valid = 1
//	};
    print_board(s);
    printf("\n");

    double tstart = omp_get_wtime();
Sudoku result;
#pragma omp parallel
#pragma omp single
    result = place(s, 0);
    print_board(result);
    double duration = omp_get_wtime()-tstart;    
    printf("t= %8.5f sec\n",duration);
    return 0;
}
