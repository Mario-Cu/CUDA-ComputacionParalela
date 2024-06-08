/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * CUDA version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.3
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the CUDA assignment versions */
#include<cuda.h>

/* Example of macros for error checking in CUDA */
#define CUDA_CHECK_FUNCTION( call )	{ cudaError_t check = call; if ( check != cudaSuccess ) fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check) ); }
#define CUDA_CHECK_KERNEL( )	{ cudaError_t check = cudaGetLastError(); if ( check != cudaSuccess ) fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check) ); }

/* Arbitrary value to indicate that no matches are found */
#define	NOT_FOUND	-1

/* Arbitrary value to restrict the checksums period */
#define CHECKSUM_MAX	65535


/* 
 * Utils: Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Utils: Random generator
 */
#include "rng.c"

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */
/* ADD KERNELS AND OTHER FUNCTIONS HERE */

/**
__global__ void kerPatMatches(int *d_pat_matches,int pat_number, unsigned long *d_pat_found){
	unsigned long tidg = blockIdx.x * blockDim.x + threadIdx.x;
	if(tidg<pat_number){
		if(d_pat_found[tidg] != NOT_FOUND){
			atomicAdd(d_pat_matches,1);
		}
	}
}
*/

__global__ void reduceMatches(int *d_reduce_matches, int *seq_matches, int seq_length){

	extern __shared__ int sdata2[];

	unsigned long blocksize = blockDim.x * blockDim.y;
	unsigned long tidb = threadIdx.x;
	unsigned long tidg = blockIdx.x *blockDim.x + threadIdx.x;

	if(tidg>=seq_length){
		sdata2[tidb] =  0;
	}
	sdata2[tidb] = seq_matches[tidg];
	
	__syncthreads();

	for(unsigned long s = blockDim.x/2; s > 0; s >>= 1) {
		if (tidb < s) {
			sdata2[tidb] += sdata2[tidb+s];
			__syncthreads();
		}
		
	}
	
	if(tidb == 0){
		d_reduce_matches[blockIdx.x] = sdata2[0];
		
	}
}




__global__ void reduceFound(unsigned long *vector_d, unsigned long *reduce_d, int n) {
    extern __shared__ unsigned long sdata[];
    unsigned long tidb = threadIdx.x;
    unsigned long tidg = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tidb] = (tidg < n && vector_d[tidg] != NOT_FOUND) ? vector_d[tidg] : 0;
    __syncthreads();

    for (unsigned long s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidb < s) {
            sdata[tidb] += sdata[tidb + s];
        }
        __syncthreads();
    }

    if (tidb == 0) {
        reduce_d[blockIdx.x] = sdata[0];
    }
}




__global__ void increment_matches(int pat_number, unsigned long *d_pat_found, unsigned long *d_pat_length, int *d_seq_matches ) {
    unsigned long tidg = blockIdx.x * blockDim.x + threadIdx.x;	
    for(unsigned long pat = 0; pat < pat_number; pat++){
        if(d_pat_found[pat] != NOT_FOUND && tidg < d_pat_length[pat]) {
            atomicAdd(&d_seq_matches[d_pat_found[pat] + tidg], 1);
        }
    }
}

__global__ void createSeq(unsigned long random, float prob_G, float prob_C, float prob_A, unsigned long seq_length, char *d_sequence) {
    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < seq_length) {
        unsigned long random_local = random;
        rng_skip(&random_local, tid);
        double prob = rng_next(&random_local);
        if(prob < prob_G) d_sequence[tid] = 'G';
        else if(prob < prob_C) d_sequence[tid] = 'C';
        else if(prob < prob_A) d_sequence[tid] = 'A';
        else d_sequence[tid] = 'T';
    }
}

__global__ void searchPatterns(int *pat_matches, unsigned long seq_length, unsigned long pat_number, unsigned long *d_pat_length, char * d_sequence, char **d_pattern, unsigned long *d_pat_found, int *d_seq_matches) {
    
	unsigned long tid_block = (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned long pat,lind;
	
	if(tid_block < seq_length){
		for( pat=0; pat < pat_number; pat++){
			for(lind=0;lind < d_pat_length[pat] && tid_block+lind<seq_length && d_pat_length[pat]<=seq_length;lind++ ){
				if(d_sequence[tid_block+lind]!=d_pattern[pat][lind]){
					break;
				}
			}
			if(lind==d_pat_length[pat]){
				atomicMin((unsigned long long *)&d_pat_found[pat], tid_block);
			}
		}
	}	
}

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

/*
 * Function: Allocate new patttern
 */
char *pattern_allocate( rng_t *random, unsigned long pat_rng_length_mean, unsigned long pat_rng_length_dev, unsigned long seq_length, unsigned long *new_length ) {

	/* Random length */
	unsigned long length = (unsigned long)rng_next_normal( random, (double)pat_rng_length_mean, (double)pat_rng_length_dev );
	if ( length > seq_length ) length = seq_length;
	if ( length <= 0 ) length = 1;

	/* Allocate pattern */
	char *pattern = (char *)malloc( sizeof(char) * length );
	if ( pattern == NULL ) {
		fprintf(stderr,"\n-- Error allocating a pattern of size: %lu\n", length );
		exit( EXIT_FAILURE );
	}

	/* Return results */
	*new_length = length;
	return pattern;
}

/*
 * Function: Fill random sequence or pattern
 */
void generate_rng_sequence( rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length) {
	unsigned long ind; 
	for( ind=0; ind<length; ind++ ) {
		double prob = rng_next( random );
		if( prob < prob_G ) seq[ind] = 'G';
		else if( prob < prob_C ) seq[ind] = 'C';
		else if( prob < prob_A ) seq[ind] = 'A';
		else seq[ind] = 'T';
	}
}

/*
 * Function: Copy a sample of the sequence
 */
void copy_sample_sequence( rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Copy sample */
	unsigned long ind; 
	for( ind=0; ind<length; ind++ )
		pattern[ind] = sequence[ind+location];
}

/*
 * Function: Regenerate a sample of the sequence
 */
void generate_sample_sequence( rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length ) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Regenerate sample */
	rng_t local_random = random_seq;
	rng_skip( &local_random, location );
	generate_rng_sequence( &local_random, prob_G, prob_C, prob_A, pattern, length);
}


/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<seq_length> <prob_G> <prob_C> <prob_A> <pat_rng_num> <pat_rng_length_mean> <pat_rng_length_dev> <pat_samples_num> <pat_samp_length_mean> <pat_samp_length_dev> <pat_samp_loc_mean> <pat_samp_loc_dev> <pat_samp_mix:B[efore]|A[fter]|M[ixed]> <long_seed>\n");
	fprintf(stderr,"\n");
}



/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	/* 0. Default output and error without buffering, forces to write immediately */
	setbuf(stdout, NULL);
	setbuf(stderr, NULL);

	/* 1. Read scenary arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 15) {
		fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	unsigned long seq_length = atol( argv[1] );
	float prob_G = atof( argv[2] );
	float prob_C = atof( argv[3] );
	float prob_A = atof( argv[4] );
	if ( prob_G + prob_C + prob_A > 1 ) {
		fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}
	prob_C += prob_G;
	prob_A += prob_C;

	int pat_rng_num = atoi( argv[5] );
	unsigned long pat_rng_length_mean = atol( argv[6] );
	unsigned long pat_rng_length_dev = atol( argv[7] );
	
	int pat_samp_num = atoi( argv[8] );
	unsigned long pat_samp_length_mean = atol( argv[9] );
	unsigned long pat_samp_length_dev = atol( argv[10] );
	unsigned long pat_samp_loc_mean = atol( argv[11] );
	unsigned long pat_samp_loc_dev = atol( argv[12] );

	char pat_samp_mix = argv[13][0];
	if ( pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M' ) {
		fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	unsigned long seed = atol( argv[14] );

#ifdef DEBUG
	/* DEBUG: Print arguments */
	printf("\nArguments: seq_length=%lu\n", seq_length );
	printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A );
	printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev );
	printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev );
	printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed );
	printf("\n");
#endif // DEBUG

        CUDA_CHECK_FUNCTION( cudaSetDevice(0) );

	/* 2. Initialize data structures */
	/* 2.1. Skip allocate and fill sequence */
	rng_t random = rng_new( seed );
	rng_skip( &random, seq_length );

	/* 2.2. Allocate and fill patterns */
	/* 2.2.1 Allocate main structures */
	int pat_number = pat_rng_num + pat_samp_num;
	unsigned long *pat_length = (unsigned long *)malloc( sizeof(unsigned long) * pat_number );
	char **pattern = (char **)malloc( sizeof(char*) * pat_number );
	if ( pattern == NULL || pat_length == NULL ) {
		fprintf(stderr,"\n-- Error allocating the basic patterns structures for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}

	/* 2.2.2 Allocate and initialize ancillary structure for pattern types */
	int ind;
	unsigned long lind;
	#define PAT_TYPE_NONE	0
	#define PAT_TYPE_RNG	1
	#define PAT_TYPE_SAMP	2
	char *pat_type = (char *)malloc( sizeof(char) * pat_number );
	if ( pat_type == NULL ) {
		fprintf(stderr,"\n-- Error allocating ancillary structure for pattern of size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_NONE;

	/* 2.2.3 Fill up pattern types using the chosen mode */
	switch( pat_samp_mix ) {
	case 'A':
		for( ind=0; ind<pat_rng_num; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		for( ; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		break;
	case 'B':
		for( ind=0; ind<pat_samp_num; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		for( ; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		break;
	default:
		if ( pat_rng_num == 0 ) {
			for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		}
		else if ( pat_samp_num == 0 ) {
			for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		}
		else if ( pat_rng_num < pat_samp_num ) {
			int interval = pat_number / pat_rng_num;
			for( ind=0; ind<pat_number; ind++ ) 
				if ( (ind+1) % interval == 0 ) pat_type[ind] = PAT_TYPE_RNG;
				else pat_type[ind] = PAT_TYPE_SAMP;
		}
		else {
			int interval = pat_number / pat_samp_num;
			for( ind=0; ind<pat_number; ind++ ) 
				if ( (ind+1) % interval == 0 ) pat_type[ind] = PAT_TYPE_SAMP;
				else pat_type[ind] = PAT_TYPE_RNG;
		}
	}

	/* 2.2.4 Generate the patterns */
	for( ind=0; ind<pat_number; ind++ ) {
		if ( pat_type[ind] == PAT_TYPE_RNG ) {
			pattern[ind] = pattern_allocate( &random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind] );
			generate_rng_sequence( &random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind] );
		}
		else if ( pat_type[ind] == PAT_TYPE_SAMP ) {
			pattern[ind] = pattern_allocate( &random, pat_samp_length_mean, pat_samp_length_dev, seq_length, &pat_length[ind] );
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
			rng_t random_seq_orig = rng_new( seed );
			generate_sample_sequence( &random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#else
			copy_sample_sequence( &random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#endif
		}
		else {
			fprintf(stderr,"\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind );
			exit( EXIT_FAILURE );
		}
	}
	free( pat_type );

	/* Allocate and move the patterns to the GPU */
	unsigned long *d_pat_length;
	char **d_pattern;
	CUDA_CHECK_FUNCTION( cudaMalloc( &d_pat_length, sizeof(unsigned long) * pat_number ) );
	CUDA_CHECK_FUNCTION( cudaMalloc( &d_pattern, sizeof(char *) * pat_number ) );

	char **d_pattern_in_host = (char **)malloc( sizeof(char*) * pat_number );
	if ( d_pattern_in_host == NULL ) {
		fprintf(stderr,"\n-- Error allocating the patterns structures replicated in the host for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	for( ind=0; ind<pat_number; ind++ ) {
		CUDA_CHECK_FUNCTION( cudaMalloc( &(d_pattern_in_host[ind]), sizeof(char *) * pat_length[ind] ) );
        	CUDA_CHECK_FUNCTION( cudaMemcpy( d_pattern_in_host[ind], pattern[ind], pat_length[ind] * sizeof(char), cudaMemcpyHostToDevice ) );
	}
	CUDA_CHECK_FUNCTION( cudaMemcpy( d_pattern, d_pattern_in_host, pat_number * sizeof(char *), cudaMemcpyHostToDevice ) );

	/* Avoid the usage of arguments to take strategic decisions
	 * In a real case the user only has the patterns and sequence data to analize
	 */
	argc = 0;
	argv = NULL;
	pat_rng_num = 0;
	pat_rng_length_mean = 0;
	pat_rng_length_dev = 0;
	pat_samp_num = 0;
	pat_samp_length_mean = 0;
	pat_samp_length_dev = 0;
	pat_samp_loc_mean = 0;
	pat_samp_loc_dev = 0;
	pat_samp_mix = '0';

	/* 2.3. Other result data and structures */
	int pat_matches = 0;

	/* 2.3.1. Other results related to patterns */
	unsigned long *pat_found;
	pat_found = (unsigned long *)malloc( sizeof(unsigned long) * pat_number );
	if ( pat_found == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux pattern structure for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	
	/* 3. Start global timer */
        CUDA_CHECK_FUNCTION( cudaDeviceSynchronize() );
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */
	/* 2.1. Allocate and fill sequence */

	char *sequence = (char *)malloc( sizeof(char) * seq_length );
	if ( sequence == NULL ) {
		fprintf(stderr,"\n-- Error allocating the sequence for size: %lu\n", seq_length );
		exit( EXIT_FAILURE );
	}

	unsigned long nElem = seq_length;
	unsigned long bSize = 128;
	unsigned long nBlock = 0;
	nBlock = nElem / bSize;

	if(nElem % bSize != 0){
		nBlock++;
	}
	
	unsigned long nElem2 = pat_number;
	unsigned long bSize2 = 1024;
	unsigned long nBlock2 = 0;
	nBlock2 = nElem2 / bSize2;

	if(nElem2 % bSize2 != 0){
		nBlock2++;
	}

	dim3 grid(nBlock,1);
	dim3 block(bSize,1);

	dim3 grid2(nBlock2,1);
	dim3 block2(bSize2,1);


	char *d_sequence = NULL;
	CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_sequence, sizeof(char) * seq_length));
	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_length, pat_length, sizeof(unsigned long) * pat_number, cudaMemcpyHostToDevice));
	random = rng_new( seed );
	unsigned long random_local = random;
	createSeq<<<grid, block>>>(random_local, prob_G, prob_C, prob_A, seq_length, d_sequence);
	CUDA_CHECK_KERNEL();
	
		

#ifdef DEBUG
	/* DEBUG: Print sequence and patterns */
	printf("-----------------\n");
	printf("Sequence: ");
	for( lind=0; lind<seq_length; lind++ ) 
		printf( "%c", sequence[lind] );
	printf("\n-----------------\n");
	printf("Patterns: %d ( rng: %d, samples: %d )\n", pat_number, pat_rng_num, pat_samp_num );
	int debug_pat;
	for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		printf( "Pat[%d]: ", debug_pat );
		for( lind=0; lind<pat_length[debug_pat]; lind++ ) 
			printf( "%c", pattern[debug_pat][lind] );
		printf("\n");
	}
	printf("-----------------\n\n");
#endif // DEBUG

	/* 2.3.2. Other results related to the main sequence */
	int *seq_matches;
	seq_matches = (int *)malloc( sizeof(int) * seq_length );
	if ( seq_matches == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux sequence structures for size: %lu\n", seq_length );
		exit( EXIT_FAILURE );
	}

	
	/* 4. Initialize ancillary structures */

	for( ind=0; ind<pat_number; ind++) {
		pat_found[ind] = NOT_FOUND;
	}
	memset(seq_matches,0,sizeof(int)*seq_length);

	unsigned long *d_pat_found = NULL;
	int *d_seq_matches = NULL;

	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_seq_matches, sizeof(int) * seq_length));
	CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_pat_found, sizeof(unsigned long) * pat_number));
	cudaMemset(d_seq_matches, 0, seq_length*sizeof(int));
	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_found, pat_found, sizeof(unsigned long)*pat_number, cudaMemcpyHostToDevice));

	
	/* 5. Search for each pattern */
	searchPatterns<<<grid,block>>>(&pat_matches, seq_length, pat_number,d_pat_length,d_sequence,d_pattern,d_pat_found,d_seq_matches);
	CUDA_CHECK_KERNEL();
				

/**
	int *h_pat_matches = (int *) malloc (sizeof(int));
	h_pat_matches[0] = 0;
	int *d_pat_matches;
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_pat_matches, sizeof(int)));
	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_matches,h_pat_matches, sizeof(int), cudaMemcpyHostToDevice));

	kerPatMatches<<<grid,block>>>(d_pat_matches,pat_number, d_pat_found);
	CUDA_CHECK_KERNEL();
	CUDA_CHECK_FUNCTION(cudaMemcpy(h_pat_matches, d_pat_matches, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK_FUNCTION(cudaMemcpy(pat_found, d_pat_found, sizeof(unsigned long) * pat_number, cudaMemcpyDeviceToHost));

	pat_matches = h_pat_matches[0];

*/





	unsigned long checksum_found = 0;
	unsigned long *h_reduce_checksum_found = (unsigned long *) malloc(nBlock2 * sizeof(unsigned long));
	memset(h_reduce_checksum_found,0,sizeof(unsigned long) * nBlock2);
	unsigned long *d_reduce_checksum_found = NULL;
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_reduce_checksum_found, sizeof(unsigned long) * nBlock2));
	
	reduceFound<<< grid2, block2, bSize2*sizeof(unsigned long)>>>(d_pat_found,d_reduce_checksum_found, pat_number);
	CUDA_CHECK_KERNEL();

	CUDA_CHECK_FUNCTION(cudaMemcpy(h_reduce_checksum_found, d_reduce_checksum_found, sizeof(unsigned long) * nBlock2 , cudaMemcpyDeviceToHost););
	

	for(int i = 1; i < nBlock2; i++) {h_reduce_checksum_found[0] += h_reduce_checksum_found[i]; }
	checksum_found = h_reduce_checksum_found[0] % CHECKSUM_MAX;

	int k = 1;
	// Unrolling by a factor of 8
	for(; k + 7 < nBlock2; k += 8) {
		h_reduce_checksum_found[0] += h_reduce_checksum_found[k] 
							+ h_reduce_checksum_found[k + 1]
							+ h_reduce_checksum_found[k + 2]
							+ h_reduce_checksum_found[k + 3]
							+ h_reduce_checksum_found[k + 4]
							+ h_reduce_checksum_found[k + 5]
							+ h_reduce_checksum_found[k + 6]
							+ h_reduce_checksum_found[k + 7];
	}
	// Handle the remaining elements
	for(; k < nBlock2; k++) {
		h_reduce_checksum_found[0] += h_reduce_checksum_found[k];
	}

	CUDA_CHECK_FUNCTION(cudaMemcpy(pat_found, d_pat_found, sizeof(unsigned long)*pat_number, cudaMemcpyDeviceToHost));

	
	for(unsigned long pat = 0; pat<pat_number; pat++){
		if(pat_found[pat] != NOT_FOUND){
			pat_matches++;
		}
	}

	increment_matches<<<grid,block>>>(pat_number, d_pat_found, d_pat_length, d_seq_matches);
	CUDA_CHECK_KERNEL();


	unsigned long checksum_matches = 0;

	int *h_reduce_matches = (int *) malloc(sizeof(int) * nBlock);
	int *d_reduce_matches = NULL;
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_reduce_matches, sizeof(int) * nBlock));
	reduceMatches<<<grid,block,bSize*sizeof(int)>>>(d_reduce_matches,d_seq_matches,seq_length);
	CUDA_CHECK_KERNEL();

	CUDA_CHECK_FUNCTION(cudaMemcpy(h_reduce_matches,d_reduce_matches,sizeof(int)*nBlock,cudaMemcpyDeviceToHost));
		
	int z = 1;
	// Unrolling by a factor of 8
	for(; z + 7 < nBlock; z += 8) {
		h_reduce_matches[0] += h_reduce_matches[z] 
							+ h_reduce_matches[z + 1]
							+ h_reduce_matches[z + 2]
							+ h_reduce_matches[z + 3]
							+ h_reduce_matches[z + 4]
							+ h_reduce_matches[z + 5]
							+ h_reduce_matches[z + 6]
							+ h_reduce_matches[z + 7];
	}
	// Handle the remaining elements
	for(; z < nBlock; z++) {
		h_reduce_matches[0] += h_reduce_matches[z];
	}

		checksum_matches = h_reduce_matches[0] % CHECKSUM_MAX;
		

#ifdef DEBUG
	/* DEBUG: Write results */
	printf("-----------------\n");
	printf("Found start:");
	for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		printf( " %lu", pat_found[debug_pat] );
	}
	printf("\n");
	printf("-----------------\n");
	printf("Matches:");
	for( lind=0; lind<seq_length; lind++ ) 
		printf( " %d", seq_matches[lind] );
	printf("\n");
	printf("-----------------\n");
#endif // DEBUG

	/* Free local resources */	
	free( sequence );
	free( seq_matches );
	cudaFree(d_pat_found);
	cudaFree(d_pat_length);
	cudaFree(d_pattern);
	cudaFree(d_sequence);

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 8. Stop global timer */
    CUDA_CHECK_FUNCTION( cudaDeviceSynchronize() );
	ttotal = cp_Wtime() - ttotal;

	/* 9. Output for leaderboard */
	printf("\n");
	/* 9.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	/* 9.2. Results: Statistics */
	printf("Result: %d, %lu, %lu\n\n", 
			pat_matches,
			checksum_found,
			checksum_matches );
		
	/* 10. Free resources */	
	int i;
	for( i=0; i<pat_number; i++ ) free( pattern[i] );
	free( pattern );
	free( pat_length );
	free( pat_found );

	/* 11. End */
	return 0;
}
