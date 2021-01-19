#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sched.h>
#include <sys/time.h>
#include <sicm_low.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>

#define BILLION  1000000000L;

int main(int argc, char *argv[]){

	double **a, **b, **c, **temp;
//	double ***d, ***e;
	
	clock_t start, end;
        struct timespec begin, stop, obegin, ostop;
	long double accum;
//	size*=1024*8;

	MPI_Init(&argc, &argv);
	sicm_class_init();

	int rank;
	int size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int proc_rt;
	proc_rt = (int)sqrt(size);
	if((proc_rt*proc_rt) != size)
	{
		MPI_Finalize();
		return -1;
	}
	unsigned long r_size = (unsigned long)strtoul(argv[1], NULL, 10)/proc_rt;
	int chunks = sicm_class_get_max_chunks();
	int sum_buffer;
        MPI_Allreduce(&chunks, &sum_buffer, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        chunks = sum_buffer/size;
	int fast_part = sicm_class_fast_partition();
	MPI_Allreduce(&fast_part, &sum_buffer, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        fast_part = sum_buffer/size;
	int slow_part = sicm_class_slow_partition();
	MPI_Allreduce(&slow_part, &sum_buffer, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        slow_part = sum_buffer/size;
	int slowest_part = sicm_class_slowest_partition();
	MPI_Allreduce(&slowest_part, &sum_buffer, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        slowest_part = sum_buffer/size;
	unsigned long long total_capacity = (unsigned long long)sicm_class_get_max_capacity();
	unsigned long long sum_capacity;
	MPI_Allreduce(&total_capacity, &sum_capacity, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        total_capacity = sum_capacity/size;
	int k1_no_of_streams = 3;
	int pg_size = 4096;
	printf("Ratio %1.9lf\n", (double)(k1_no_of_streams*r_size*r_size)/(fast_part*total_capacity));
	if(((double)(k1_no_of_streams*r_size*r_size)/(fast_part*total_capacity)) > 0.02) 
		chunks = (int)(((double)(k1_no_of_streams*r_size*r_size)/(fast_part*total_capacity))*chunks);
	else
		chunks = 0.02*chunks;
	chunks = floor(sqrt(chunks))*floor(sqrt(chunks));
	printf("Rank %d Chunks %d Fast partition %d Slow Partition %d Slowest Partition %d Total Capacity %lu\n", rank, chunks, fast_part, slow_part, slowest_part, total_capacity);
	r_size /= sqrt(chunks);
	unsigned long c_size = r_size;
//	unsigned long l_size = r_size;
//	unsigned long b_size = r_size;
//	unsigned long w_size = r_size;
	unsigned long row = floor(sqrt(r_size*c_size/sizeof(double)));
	unsigned long col = row;
	unsigned long chunk_size = (c_size*r_size + (pg_size - ((c_size*r_size)%pg_size)));
	unsigned long chunk_elements = chunk_size/sizeof(double);

	printf("Rank %d Chunksize %d Row %d Col %d Mat element size %d element size %d\n", rank, chunk_size, row, col, row*col*sizeof(double), r_size*c_size);
//	unsigned long len = l_size/sizeof(double**);
//	unsigned long bdh = b_size/sizeof(double*);
//	unsigned long wid = w_size/sizeof(double);
	int tag1 = 1;
	int tag2 = 2;
	int prev, next;
	MPI_Request reqs [4] ; 
	MPI_Status stats [4];

	srand(74102);
	int i = 0, j = 0, k = 0, p = 0;
	int step;
	a = (double **)sicm_class_alloc(0, chunks*sizeof(double *),chunks+1);
	b = (double **)sicm_class_alloc(0, chunks*sizeof(double *), chunks+ 1);
	c = (double **)sicm_class_alloc(0, chunks*sizeof(double *), chunks+1);
	for(i = 0; i < chunks; i++){
		if(i%slowest_part < fast_part)
			j = 0;
		else if((i%slowest_part >= fast_part)&&(i%slowest_part < slow_part))
			j = 1;
		else
			j = 2;
		a[i] = (double *)sicm_class_alloc(j, chunk_size, i);
		b[i] = (double *)sicm_class_alloc(j, chunk_size, i);
		c[i] = (double *)sicm_class_alloc(j, chunk_size, i);
	}
	printf("Allocated\n");
	for(p = 0; p < chunks; p++){
		for(i = 0; i < row; i++){
			for(j = 0; j < col; j++){
				a[p][i*row+j] = rand()*rand()/(rand()+1);
				b[p][i*row+j] = rand()*rand()/(rand()+1);
				c[p][i*row+j] = 0.0;
			}
		}
	}

  	int *low, *high;
	int *threadorder;
	int total_work_chunks = chunks;
	int work_chunks_assigned = 0;
	int chunk_increament = total_work_chunks;
	int numthreads;
	int to = 0;
  	int flag_move = -1;
#pragma omp parallel
{
  	numthreads = omp_get_num_threads();
}
	low = (int*)malloc(sizeof(int)*numthreads);
	high = (int*)malloc(sizeof(int)*numthreads);
	threadorder = (int*)malloc(sizeof(int)*numthreads);
	MPI_Barrier(MPI_COMM_WORLD);
			clock_gettime( CLOCK_MONOTONIC, &begin);



#pragma omp parallel shared(work_chunks_assigned, low, high, threadorder, to, flag_move) private(i,j,k,p)
{
  int threadnum = omp_get_thread_num();

  #pragma omp critical
  {
	threadorder[to%numthreads] = threadnum;
	to++;
  }
//for(i = 0; i < numthreads; i++)
//printf("Rank: %d It: %d Thread Id: %d Numthreads: %d TO %d\n", rank, i, threadnum, numthreads, threadorder[i]);

  while(work_chunks_assigned < total_work_chunks){
  if(to <= (3*numthreads)){
	  #pragma omp barrier
	  #pragma omp master
	  {
  		sicm_class_chunk_scheduler_ie(&work_chunks_assigned, total_work_chunks, chunk_size,  k1_no_of_streams, numthreads, threadnum, threadorder, low, high, &flag_move);
  	  }
	  #pragma omp barrier
  }
  else{
#pragma omp critical
{
	sicm_class_chunk_scheduler_ie(&work_chunks_assigned, total_work_chunks, chunk_size,  k1_no_of_streams, numthreads, threadnum, threadorder, low, high, &flag_move);
}
  }
/*#pragma omp critical
{
	printf("Rank: %d Thread Id: %d Low: %d, High %d\n", rank, threadnum, low[threadnum], high[threadnum]);
}*/
  	if(flag_move == 1){
#pragma omp critical
{
  		for(p = low[threadnum]; p < high[threadnum]; p++){
 //       printf("MI Rank: %d Thread Id: %d Low: %d, High: %d I: %d\n", rank, threadnum, low[threadnum], high[threadnum], p);
  		        sicm_class_move(0,a[p]);
  		        sicm_class_move(0,b[p]);
  			sicm_class_move(0,c[p]);
  		}
}
  	}
 	 for(p = low[threadnum]; p < high[threadnum]; p++){
/*#pragma omp critical
{
        printf("LK Rank: %d Thread Id: %d Low: %d, High: %d I: %d\n", rank, threadnum, low[threadnum], high[threadnum], p);
}*/
                for(i = 0; i < row; i++){
                        for(j = 0; j < col; j++){
                                double tmp = 0.0;
                                for(k = 0; k < row; k++){

                                        tmp += a[p][i*col+k]*b[p][k*col+j];
                                }
                                c[p][i*col+j] += tmp;
                        }
                }
	}
        if(flag_move == 1){
#pragma omp critical
{
	        for(p = low[threadnum]; p < high[threadnum]; p++){
 //       printf("MB Rank: %d Thread Id: %d Low: %d, High: %d I: %d\n", rank, threadnum, low[threadnum], high[threadnum], p);
        	        sicm_class_move(2,a[p]);
                	sicm_class_move(2,b[p]);
                	sicm_class_move(2,c[p]);
        	}
}
        }
  	#pragma omp critical
  	{
		threadorder[to%numthreads] = threadnum;
		to++;
  	}
  }
}
	free(low);
	free(high);
	free(threadorder);

//printf("Rank: %d\n", rank);

	for(step = 0; step < (proc_rt - 1); step++)	{
//printf("Send:\n");
for(i = 0; i < chunks; i++){
MPI_Isend(b[i], chunk_elements, MPI_DOUBLE, (rank+proc_rt)%size, tag1, MPI_COMM_WORLD, &reqs [0]);
MPI_Isend(a[i], chunk_elements, MPI_DOUBLE, (rank+1)%size, tag2, MPI_COMM_WORLD, &reqs [1]);
//printf("Recv\n");
if(rank < proc_rt){
	MPI_Irecv(b[i], chunk_elements, MPI_DOUBLE, (size+rank-proc_rt)%size, tag1, MPI_COMM_WORLD, &reqs [2]);
	MPI_Irecv(a[i], chunk_elements, MPI_DOUBLE, (size+rank-1)%size, tag2, MPI_COMM_WORLD, &reqs [3]);
}else{
	MPI_Irecv(b[i], chunk_elements, MPI_DOUBLE, (rank-proc_rt)%size, tag1, MPI_COMM_WORLD, &reqs [2]);
	MPI_Irecv(a[i], chunk_elements, MPI_DOUBLE, (rank-1)%size, tag2, MPI_COMM_WORLD, &reqs [3]);
}
//printf("Wait\n");
	MPI_Waitall (4 , reqs , stats );
}


	total_work_chunks = chunks;
        work_chunks_assigned = 0;
	chunk_increament = total_work_chunks;
	k1_no_of_streams = 3;
	to = 0;
	low = (int*)malloc(sizeof(int)*numthreads);
	high = (int*)malloc(sizeof(int)*numthreads);
	threadorder = (int*)malloc(sizeof(int)*numthreads);
  	flag_move = -1;

#pragma omp parallel shared(work_chunks_assigned, low, high, threadorder, to, flag_move)   private(i,j,k,p) //OpenMP region
{ 
  int threadnum = omp_get_thread_num();

  #pragma omp critical			//Decides the order in which the threads should be allocated chunks
  {     
        threadorder[to%numthreads] = threadnum;
        to++;
  }
  
  while(work_chunks_assigned < total_work_chunks){ 	//Work till all chunks are assigned
  if(to <= (3*numthreads)){		//Condition to determine if only main thread assigns chunks or all threads steal
	  #pragma omp barrier
          #pragma omp master
          {     
                sicm_class_chunk_scheduler_ie(&work_chunks_assigned, total_work_chunks, chunk_size,  k1_no_of_streams, numthreads, threadnum, threadorder, low, high, &flag_move);
          }
          #pragma omp barrier
  }
  else{
#pragma omp critical
{       
        sicm_class_chunk_scheduler_ie(&work_chunks_assigned, total_work_chunks, chunk_size,  k1_no_of_streams, numthreads, threadnum, threadorder, low, high, &flag_move);
} 
  }     
        if(flag_move == 1){				//Move the blocks in thee fast memory if available
#pragma omp critical
{
                for(p = low[threadnum]; p < high[threadnum]; p++){
 //       printf("MI Rank: %d Thread Id: %d Low: %d, High: %d I: %d\n", rank, threadnum, low[threadnum], high[threadnum], p);
                        sicm_class_move(0,a[p]);
                        sicm_class_move(0,b[p]);
                        sicm_class_move(0,c[p]);
                }
}
        }
         for(p = low[threadnum]; p < high[threadnum]; p++){		//Loop to traverse over assigned chunks
/*#pragma omp critical
{
        printf("LK Rank: %d Thread Id: %d Low: %d, High: %d, I: %d\n", rank, threadnum, low[threadnum], high[threadnum], p);
}*/
                for(i = 0; i < row; i++){				//The compute kernel
                        for(j = 0; j < col; j++){
                                double tmp = 0.0;
                                for(k = 0; k < row; k++){
                                        
                                        tmp += a[p][i*col+k]*b[p][k*col+j];
                                }
                                c[p][i*col+j] += tmp;
                        }
                }
        }
        if(flag_move == 1){			//Move the computed blocks into the slowest memory
#pragma omp critical
{
                for(p = low[threadnum]; p < high[threadnum]; p++){
//        printf("MB Rank: %d Thread Id: %d Low: %d, High: %d, I: %d\n", rank, threadnum, low[threadnum], high[threadnum], p);
                        sicm_class_move(2,a[p]);
                        sicm_class_move(2,b[p]);
                        sicm_class_move(2,c[p]);
                }
}
        }
        #pragma omp critical		//Decides the order in which the threads should be allocated chunks for the next scheduling iteration
        {
                threadorder[to%numthreads] = threadnum;
                to++;
        }
  }
}

}
	free(low);
	free(high);
	free(threadorder);

	MPI_Barrier(MPI_COMM_WORLD);
			clock_gettime( CLOCK_MONOTONIC, &stop);
			accum = ( stop.tv_sec - begin.tv_sec ) + (long double)( stop.tv_nsec - begin.tv_nsec ) / (long double)BILLION;
			if(rank == 0){
				printf("MatMul\n");
				printf("%d %lu %LF\n", rank, size*chunks*r_size*c_size, accum);
			}
	sicm_class_free(a);
	sicm_class_free(b);
	sicm_class_free(c);
	sicm_fini();
	MPI_Finalize();
	return 0;
}
