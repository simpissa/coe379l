#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>


int max(int x, int y, int distance) {
	int squared_dist = pow(x,2) + pow(y,2);
	return squared_dist > distance ? squared_dist : distance;
}
int main() {
	int n = 100;
	int coordinates[n][2];
	int range = 1000;
	for (int i = 0; i < n; i++) {
		coordinates[i][0] = rand() % range - range/2;
		coordinates[i][1] = rand() % range - range/2;
		printf("(%d,%d) ", coordinates[i][0], coordinates[i][1]);
	}
	
	
	int max_distance;
#pragma omp parallel for reduction(max:max_distance)
	for (int i = 0; i < n ; i++) {
		
		max_distance = max(coordinates[i][0], coordinates[i][1], max_distance);
	}
	printf("\nMaximum Cartesian Distance: %f\n",sqrt(max_distance));

	return 0;
}
