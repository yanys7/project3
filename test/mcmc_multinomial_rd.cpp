#include <tuple>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <hmlp_blas_lapack.h>

#include <data.hpp>
#include <mixture_multinomial_rd.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

/* Include R headers. */
#include <R.h>
#include <Rembedded.h>
#include <Rdefines.h>

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace hmlp;

int main( int argc, char *argv[] )
{
  using T = double;

  size_t n = 800;
  size_t w1 = 3;
  size_t w2 = 7;
  size_t q = 755;
  size_t q1 = 100;
  size_t q2 = 100;
  size_t burnIn = 30000;
  size_t niter = 50000;
  size_t permute = 1;
  size_t n_mixtures = 4;
  size_t d = 2;

  /* Must initialize R package before calling any R functions!!! */
  Rf_initEmbeddedR(argc, argv);
  GetRNGstate();
  PutRNGstate();
  
  //for (int32_t iter = 0; iter < 10; iter ++)
  //{
  //  GetRNGstate();
  //  std::cout << "runif(0, 1):"  << runif(0, 1) << std::endl;
  //  PutRNGstate();
  //}

	if ( argc == 20 )
	{
		/** read parameters */
		sscanf( argv[ 1 ], "%lu", &n );
		sscanf( argv[ 2 ], "%lu", &w1 );
		sscanf( argv[ 3 ], "%lu", &w2 );
		sscanf( argv[ 4 ], "%lu", &q );
		sscanf( argv[ 5 ], "%lu", &burnIn );
		sscanf( argv[ 6 ], "%lu", &niter );
    sscanf( argv[ 7 ], "%lu", &q1 );
    sscanf( argv[ 8 ], "%lu", &q2 );
	}
	else
	{
		printf( "\n[usage] ./mcmc.x <n> <w> <q> <q1> <niter>\n\n" );
  }

  std::string Y_filename(       argv[ 9 ] );
  std::string M_filename(       argv[ 10 ] );
  std::string A_filename(       argv[ 11 ] );
  std::string C1_filename(      argv[ 12 ]  );
  std::string C2_filename(      argv[ 13 ] ); 
  std::string beta_m_filename(  argv[ 14 ]  );
  std::string alpha_a_filename( argv[ 15 ] );
  std::string pi_mixtures_filename( argv[ 16 ]  );
  std::string Psi_filename(     argv[ 17 ] );
  std::string D_filename(       argv[ 18 ] );

  hmlp::Data<T> Y( n, 1 );
  hmlp::Data<T> M( n, q ); 
  hmlp::Data<T> A( n, 1 );
  hmlp::Data<T> C1( n, w1 );
  hmlp::Data<T> C2( n, w2 );

  hmlp::Data<T> beta_m( 1, q );
  hmlp::Data<T> alpha_a( 1, q ); 
  hmlp::Data<T> pi_mixtures( q, n_mixtures );
  hmlp::Data<T> Psi( d, d );
  hmlp::Data<T> corrD_orig( q, q );

  Y.readmatrix( n, 1, Y_filename );
  M.readmatrix( n, q, M_filename );
  A.readmatrix( n, 1, A_filename );
  C1.readmatrix( n, w1, C1_filename );
  C2.readmatrix( n, w2, C2_filename );

  beta_m.readmatrix( 1, q, beta_m_filename );
  alpha_a.readmatrix( 1, q, alpha_a_filename );
  pi_mixtures.readmatrix( q, n_mixtures, pi_mixtures_filename );
  Psi.readmatrix( d, d, Psi_filename );
  corrD_orig.readmatrix( q, q, D_filename );


  mcmc::mcmc<T>( Y, A, M, C1, C2, beta_m, alpha_a, pi_mixtures, Psi, corrD_orig, n, w1, w2, q, q1, q2, burnIn, niter );

  Rf_endEmbeddedR(0);
  return 0;
};
