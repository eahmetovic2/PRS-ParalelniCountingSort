#define COUNT_H
#include <cstdio>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <sys\timeb.h>

#define MAX_VRIJEDNOST 512
#define BLOCKS 16
#define THREADS 128

void ispis(const char*, int*, int);
void ispis2(const char *, int *, int , int );

void seqCount(int *A, int *C, int n) {
	int i, j = 0, counter[MAX_VRIJEDNOST];
	memset(counter, 0, sizeof(counter));
	// Broji ponavljanja elemenata i upisuje ih u niz counter
	for (i = 0; i<n; i++)
		counter[A[i]]++;

	i = 0;
	// upisuje izbrojane elemente u niz C
	for (i = 0; i < MAX_VRIJEDNOST; i++) {
		while (counter[i] > 0) {
			C[j] = i;
			j++;
			counter[i]--;
		}
	}
}


__global__ void parCount(int *A, int *B, int n) {
	int block_id = blockIdx.x,
		block_num = gridDim.x,
		block_size,
		block_offset,
		thread_id = threadIdx.x,
		thread_num = blockDim.x,
		thread_size,
		thread_offset,
		offset;

	__shared__ int count[MAX_VRIJEDNOST];

	//postavljanje inicijalnih vrijednosti svakog threada na 0
	thread_size = (thread_num > MAX_VRIJEDNOST ? 1 : MAX_VRIJEDNOST / thread_num);
	offset = thread_id * thread_size;
	for (int i = offset; i < offset + thread_size && i < MAX_VRIJEDNOST; ++i)
		count[i] = 0;

	__syncthreads();

	//brojanje ponavljanja svih clanova niza. Svaki thread broji svoj dio nesortiranog niza
	block_size = (block_num > n ? 1 : n / block_num);
	block_offset = block_id * block_size;

	thread_size = (thread_num > block_size ? 1 : block_size / thread_num);

	offset = block_offset + thread_id * thread_size;

	//brojanje elemenata
	for (int i = offset; i < offset + thread_size && i < block_offset + block_size && i < n; ++i)
		atomicAdd(&count[A[i]], 1);

	__syncthreads();

	//svaki thread kopira svoj dio u globalnu memoriju
	thread_size = (thread_num > MAX_VRIJEDNOST ? 1 : MAX_VRIJEDNOST / thread_num);
	thread_offset = thread_id * thread_size;
	offset = block_id * MAX_VRIJEDNOST + thread_offset;

	if (offset + thread_size <= (block_id + 1) * MAX_VRIJEDNOST)
		memcpy(&B[offset], &count[thread_offset], sizeof(int) * thread_size);
}

__global__ void merge(int *B) {
	int block_id = blockIdx.x,
		block_num = gridDim.x,
		block_size,
		block_offset,
		thread_id = threadIdx.x,
		thread_num = blockDim.x,
		thread_size,
		thread_offset,
		offset;

	// prolazi kroz veliki niz B i sabire sve elemente podnizova u prvi podniz
	for (int i = block_num, j = 2; i != 1; i /= 2, j *= 2) {
		// racunanje granice bloka (velicinu)
		block_size = i * MAX_VRIJEDNOST / block_num / 2;
		block_offset = (block_id / j) * (j * MAX_VRIJEDNOST) + block_size * (block_id % j);

		thread_size = (thread_num > block_size ? 1 : block_size / thread_num);

		// racunanje offseta gdje pocinje thread i sabiranje countova
		offset = block_offset + thread_id * thread_size;
		for (int k = offset, l = offset + (MAX_VRIJEDNOST * (j / 2));
			k < offset + thread_size && k < block_offset + block_size; ++k, ++l)
			B[k] += B[l];

		__syncthreads();
	}
}

int main(int argc, const char **argv) {

	int	n = pow(2, 20);
	printf("Broj elemenata: %d\n", n);
	printf("Max element: %d\n", MAX_VRIJEDNOST);
	printf("Broj threadova: %d\n", THREADS);
	printf("Broj blokova: %d\n\n", BLOCKS);

	int	*dA, *dB;

	int *A = (int*)calloc(n, sizeof(int));
	int *B = (int*)calloc(MAX_VRIJEDNOST, sizeof(int));
	int *C = (int*)calloc(n, sizeof(int));

	// alokacija memorije na grafickoj
	cudaMalloc((void**)&dA, sizeof(int) * n);
	cudaMalloc((void**)&dB, sizeof(int) * BLOCKS * MAX_VRIJEDNOST);
	
	srand(time(NULL));

	// upisivanje random vrijednosti u niz
	for (int i = 0; i < n; ++i)
		A[i] = rand() % MAX_VRIJEDNOST;

	// kopiranje niza u graficku memoriju
	cudaMemcpy(dA, A, sizeof(int) * n, cudaMemcpyHostToDevice);

	//Paralelni
	struct timeb start, end, start1;
	int diff;
	ftime(&start1);

	// OVO JE POTENCIJALNO O(1)
	printf("Paralelni: \n");
	parCount << <BLOCKS, THREADS >> >(dA, dB, n);
	merge << <BLOCKS, THREADS >> >(dB);
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start1.time)
		+ (end.millitm - start1.millitm));
	printf("\Paralelno prebrojavanje elemenata je trajalo %u milisekundi", diff);

	ftime(&start);
	cudaMemcpy(B, dB, sizeof(int) * MAX_VRIJEDNOST, cudaMemcpyDeviceToHost);
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start.time)
		+ (end.millitm - start.millitm));
	printf("\nKopiranje iz graficke u radnu memoriju je trajalo %u milisekundi", diff);

	ftime(&start);
	int j = 0;
	for (int i = 0; i < MAX_VRIJEDNOST; i++) {
		while (B[i] > 0) {
			C[j] = i;
			j++;
			B[i]--;
		}
	}
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start.time)
		+ (end.millitm - start.millitm));
	printf("\nKreiranje sortiranog niza trajalo %u milisekundi", diff);

	diff = (int)(1000.0 * (end.time - start1.time)
		+ (end.millitm - start1.millitm));
	printf("\nUkupno je trajalo %u milisekundi\n", diff);
	//ispis2("Paralelni", C, n, 100000);
	//ispis("Paralelni", C, n);


	//Sekvencijalni
	ftime(&start);
	seqCount(A, C, n);
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start.time)
		+ (end.millitm - start.millitm));
	printf("\nSekvencijalni: ");
	printf("\nSortiranje je trajalo %u milisekundi\n", diff);
	//ispis2("Sekvencijalni", C, n, 1000);
	//ispis("Sekvencijalni", C, n);


	char str[60];
	fgets(str, 60, stdin);
	cudaFree(dA);
	cudaFree(dB);
	delete[] A;
	delete[] B;
	delete[] C;

	return EXIT_SUCCESS;
}


void ispis(const char *naziv, int *niz, int velicina) {
	printf("%s = [%d", naziv, niz[0]);
	for (int i = 0; i < velicina; ++i) printf(", %d", niz[i]);
	printf("]\n");
}

void ispis2(const char *naziv, int *niz, int velicina, int jump) {
	printf("%s = [%d", naziv, niz[0]);
	for (int i = 0; i < velicina; i=i+jump) printf(", %d", niz[i]);
	printf("]\n");
}