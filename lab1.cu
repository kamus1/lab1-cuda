#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

/*
 *  Lectura Archivo
 */
void Read(float** R, float** G, float** B,int *L, int *M, int *N, const char *filename) {    
	FILE *fp;
	fp = fopen(filename, "r");
	fscanf(fp, "%d %d %d\n", L ,M, N); //ahora tambien leemos L

	//guardar todas los R,G,B de todas las imagenes en un solo array para R, G y B
	int imsize_total = (*L) * (*M) * (*N);
	int imsize = (*M) * (*N); //tamaño original de 1 imagen

	float* R1 = new float[imsize_total];
	float* G1 = new float[imsize_total];
	float* B1 = new float[imsize_total];

	for( int j=0; j < (*L); ++j){ //recorrer cantidad de imagenes

		//por cada bloque de imagenes asignar los indices
		for(int i = 0; i < imsize; i++)
			fscanf(fp, "%f ", &(R1[i + imsize*j]));
		for(int i = 0; i < imsize; i++)
			fscanf(fp, "%f ", &(G1[i + imsize*j]));
		for(int i = 0; i < imsize; i++)
			fscanf(fp, "%f ", &(B1[i + imsize*j]));
	} 
	fclose(fp);
	*R = R1; *G = G1; *B = B1;
}

/*
 *  Escritura Archivo
 */
void Write(float* R, float* G, float* B, int M, int N, const char *filename) {
	//no es neceario modificar nada, se asume R,G,B tamaño M*N con valores promediados
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", M, N);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", R[i]);
    fprintf(fp, "%f\n", R[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", G[i]);
    fprintf(fp, "%f\n", G[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", B[i]);
    fprintf(fp, "%f\n", B[M*N-1]);
    fclose(fp);
}

/*
 *  Procesamiento Imagen CPU
 */
void funcionCPU(float *R, float *G, float *B, float *Rout, float *Gout, float *Bout,int L, int M, int N){
	//con CPU iría dando saltos en los bloques de colores y sumando los valores y luego dividir por L
		int imsize = M*N;

		for( int i=0; i< imsize; ++i){ //por cada pixel en la imagen
			Rout[i] = 0;
			Gout[i] = 0;
			Bout[i] = 0;
			for( int j=0; j < L; ++j){ //recorrer cantidad de imagenes
				Rout[i] += R[i + j*imsize];
				Gout[i] += G[i + j*imsize];
				Bout[i] += B[i + j*imsize];
			}
			Rout[i] /= L;
			Gout[i] /= L;
			Bout[i] /= L;
		}
}


 /*

 R [i1 i2..... in, in+1 ] Rout 
 
 
 G [i1..... in, ]
 B [i1..... in, ]

 *  Procesamiento Imagen GPU
 */
__global__ void kernelGPU(float *R, float *G, float *B, float *Rout, float *Gout, float *Bout, int L,int M, int N){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	int imsize = M*N;
	if( tid < imsize){
		Rout[tid] = 0;
		Gout[tid] = 0;
		Bout[tid] = 0;
		for (int i = 0; i < L; i++){
			Rout[tid] += R[imsize*i + tid];
			Gout[tid] += G[imsize*i + tid];
			Bout[tid] += B[imsize*i + tid];
		}
		Rout[tid] /= L;
		Gout[tid] /= L;
		Bout[tid] /= L;
	}
	
}

/*
 *  Codigo Principal
 */
int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
	int L, M, N;
    float *Rhost, *Ghost, *Bhost;
    float *Rhostout, *Ghostout, *Bhostout;
    float *Rdev, *Gdev, *Bdev;
    float *Rdevout, *Gdevout, *Bdevout;
    char names[6][3][30] = {
		//Nombres

		//-lee [0]------ escribe CPU [1]----- escribe GPU [2]
		{"images1.txt\0", "images1CPU.txt\0", "images1GPU.txt\0"},
		{"images2.txt\0", "images2CPU.txt\0", "images2GPU.txt\0"},
		{"images3.txt\0", "images3CPU.txt\0", "images3GPU.txt\0"},
		{"images4.txt\0", "images4CPU.txt\0", "images4GPU.txt\0"},
		{"images5.txt\0", "images5CPU.txt\0", "images5GPU.txt\0"},
		{"images6.txt\0", "images6CPU.txt\0", "images6GPU.txt\0"},
	
	};

    for (int i=0; i<6; i++){
	    Read(&Rhost, &Ghost, &Bhost, &L ,&M, &N, names[i][0]); // los ColorHost van a quedar de tamaño L*M*N

	    /*
	     *  CPU
	     */
		
		//los tamaños de estos no los modificamos porque vamos a hacer el promedio y quedan del tamaño original de 1 imagen
	    Rhostout = (float*)malloc(M*N*sizeof(float));
	    Ghostout = (float*)malloc(M*N*sizeof(float));
	    Bhostout = (float*)malloc(M*N*sizeof(float));

		/*
		Rhost -> L*M*N
		Ghost -> L*M*N
		Bhost -> L*M*N

		Rhostout -> M*N
		Ghostout -> M*N
		Bhostout -> M*N
		*/
		clock_t t1, t2;
		t1 = clock();
	    funcionCPU(Rhost, Ghost, Bhost, Rhostout, Ghostout, Bhostout, L,M, N);  //pasarle L
		t2 = clock();

		double ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
		printf("Tiempo CPU Imagen %d : %f [ms]\n", i+1, ms); 

	    Write(Rhostout, Ghostout, Bhostout, M, N, names[i][1]);

	    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	    



		//--------------------------------------- GPU -----------------------------------------------//
	    int grid_size, block_size = 256;
	    grid_size = (int)ceil((float) M * N / block_size);
		
		//es necesario ajustar los ColorDev (device) a tamaños L*M*N
		//pero los out no, quedan como M*N

		//reserva memoria
	    cudaMalloc((void**)&Rdev, L * M * N * sizeof(float));
	    cudaMalloc((void**)&Gdev, L * M * N * sizeof(float));
	    cudaMalloc((void**)&Bdev, L * M * N * sizeof(float));

		//copia datos de ColorHost --> ColorDev
	    cudaMemcpy(Rdev, Rhost, L * M * N * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(Gdev, Ghost, L * M * N * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(Bdev, Bhost, L * M * N * sizeof(float), cudaMemcpyHostToDevice);
	        
	    cudaMalloc((void**)&Rdevout, M * N * sizeof(float));
	    cudaMalloc((void**)&Gdevout, M * N * sizeof(float));
	    cudaMalloc((void**)&Bdevout, M * N * sizeof(float));
	    
		// tiempos
		cudaEvent_t ct1, ct2;
		float dt;
		cudaEventCreate(&ct1);
		cudaEventCreate(&ct2);
		cudaEventRecord(ct1);

		//GPU
	    kernelGPU<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout, Gdevout, Bdevout, L, M, N); 

		cudaEventRecord(ct2);
		cudaEventSynchronize(ct2);
		cudaEventElapsedTime(&dt, ct1, ct2);
		printf("Tiempo GPU Imagen %d: %f [ms]\n\n",i+1, dt);


	    Rhostout = (float*)malloc(M*N*sizeof(float));
	    Ghostout = (float*)malloc(M*N*sizeof(float));
	    Bhostout = (float*)malloc(M*N*sizeof(float));
	    cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	    cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	    cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	    Write(Rhostout, Ghostout, Bhostout, M, N, names[i][2]);

    	cudaFree(Rdev); cudaFree(Gdev); cudaFree(Bdev);
    	cudaFree(Rdevout); cudaFree(Gdevout); cudaFree(Bdevout);
    	free(Rhost); free(Ghost); free(Bhost);
    	free(Rhostout); free(Ghostout); free(Bhostout);
		//------------------------------------------------------------------------------------------//
	}
	return 0;
}