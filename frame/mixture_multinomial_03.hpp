#ifndef MCMC_HPP
#define MCMC_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <random>
#include <limits>
#include <cstddef>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <ctime>

# include "wishart.hpp"
# include "pdflib.hpp"
# include "rnglib.hpp"
# include "PolyaGamma.h"
# include "polyagamma_wrapper.h"
# include <mvn.hpp>

using namespace std;
using namespace hmlp;


namespace hmlp
{
namespace mcmc
{
template <typename RealType = double>
class beta_distribution
{
  public:
    typedef RealType result_type;

    class param_type
    {
      public:
        typedef beta_distribution distribution_type;

        explicit param_type(RealType a = 2.0, RealType b = 2.0)
          : a_param(a), b_param(b) { }

        RealType a() const { return a_param; }
        RealType b() const { return b_param; }

        bool operator==(const param_type& other) const
        {
          return (a_param == other.a_param &&
              b_param == other.b_param);
        }

        bool operator!=(const param_type& other) const
        {
          return !(*this == other);
        }

      private:
        RealType a_param, b_param;
    };

    explicit beta_distribution(RealType a = 2.0, RealType b = 2.0)
      : a_gamma(a), b_gamma(b) { }
    explicit beta_distribution(const param_type& param)
      : a_gamma(param.a()), b_gamma(param.b()) { }

    void reset() { }

    param_type param() const
    {
      return param_type(a(), b());
    }

    void param(const param_type& param)
    {
      a_gamma = gamma_dist_type(param.a());
      b_gamma = gamma_dist_type(param.b());
    }

    template <typename URNG>
    result_type operator()(URNG& engine)
    {
      return generate(engine, a_gamma, b_gamma);
    }

    template <typename URNG>
    result_type operator()(URNG& engine, const param_type& param)
    {
      gamma_dist_type a_param_gamma(param.a()),
                      b_param_gamma(param.b());
      return generate(engine, a_param_gamma, b_param_gamma); 
    }

    result_type min() const { return 0.0; }
    result_type max() const { return 1.0; }

    result_type a() const { return a_gamma.alpha(); }
    result_type b() const { return b_gamma.alpha(); }

    bool operator==(const beta_distribution<result_type>& other) const
    {
      return (param() == other.param() &&
          a_gamma == other.a_gamma &&
          b_gamma == other.b_gamma);
    }

    bool operator!=(const beta_distribution<result_type>& other) const
    {
      return !(*this == other);
    }

  private:
    typedef std::gamma_distribution<result_type> gamma_dist_type;

    gamma_dist_type a_gamma, b_gamma;

    template <typename URNG>
    result_type generate(URNG& engine,
        gamma_dist_type& x_gamma,
        gamma_dist_type& y_gamma)
    {
      result_type x = x_gamma(engine);
      return x / (x + y_gamma(engine));
    }
};

  template <typename CharT, typename RealType>
    std::basic_ostream<CharT>& operator<<(std::basic_ostream<CharT>& os,
        const beta_distribution<RealType>& beta)
    {
      os << "~Beta(" << beta.a() << "," << beta.b() << ")";
      return os;
    }

  template <typename CharT, typename RealType>
    std::basic_istream<CharT>& operator>>(std::basic_istream<CharT>& is,
        beta_distribution<RealType>& beta)
    {
      std::string str;
      RealType a, b;
      if (std::getline(is, str, '(') && str == "~Beta" &&
          is >> a && is.get() == ',' && is >> b && is.get() == ')') {
        beta = beta_distribution<RealType>(a, b);
      } else {
        is.setstate(std::ios::failbit);
      }
      return is;
    }


template<typename T>
void Normalize( hmlp::Data<T> &A )
{
  size_t m = A.dim();
  size_t n = A.num();

  for ( size_t j = 0; j < n; j ++ )
  {
    T mean = 0.0;
    T stde = 0.0;

    /** mean */
    for ( size_t i = 0; i < m; i ++ ) mean += A( i, j );
    mean /= m;

    /** standard deviation */
    for ( size_t i = 0; i < m; i ++ ) 
    {
      T tmp = A( i, j ) - mean;
      stde += tmp * tmp;
    }
    stde /= m;
    stde = std::sqrt( stde );

    /** normalize */
    for ( size_t i = 0; i < m; i ++ ) 
    {
      A[ j * m + i ] = ( A( i, j ) - mean ) / stde;
    }
  }

}; /** end Normalize() */

template<typename T>
hmlp::Data<T> Mean( hmlp::Data<T> &A )
{
  size_t m = A.dim();
  size_t n = A.num();
  hmlp::Data<T> mean( 1, n, 0.0);

  for ( size_t j = 0; j < n; j ++ )
  {
    /** mean */
    for ( size_t i = 0; i < m; i ++ ) mean( 0, (int)j ) += A( i, j );
    mean( 0 , (int)j ) /= m;

  }

  return mean;
}; /** end Mean() */

template<typename T>
T sigmoid( T &value )
{
  return std::exp( value ) / ( static_cast<T>(1) + std::exp( value ) );
}

template<typename T>
T logit( T value )
{
  return std::log( value / ( static_cast<T>(1) - value ) );
}

template<typename T>
class Variables
{
  public:

	Variables(
		hmlp::Data<T> &userY,
		hmlp::Data<T> &userA,
		hmlp::Data<T> &userM,
		hmlp::Data<T> &userC1,
		hmlp::Data<T> &userC2,
 	  hmlp::Data<T> &userbeta_m,
		hmlp::Data<T> &useralpha_a,
		hmlp::Data<T> &userpi_mixtures,
    hmlp::Data<T> &userPsi,
    hmlp::Data<T> &usercorrD_orig,
		size_t n, size_t w1, size_t w2, size_t q, size_t q1, size_t q2 )
	  	: Y( userY ), A( userA ), M( userM ), C1( userC1 ), C2( userC2 ),
	    	beta_m( userbeta_m ), alpha_a( useralpha_a ), pi_mixtures( userpi_mixtures ), Psi( userPsi ),
        corrD_orig( usercorrD_orig )

  {
    this->n = n;
    this->w1 = w1;
    this->w2 = w2;
    this->q = q;
    this->q1 = q1;
    this->q2 = q2;

    corrD = corrD_orig;

    /** Initialize my_samples here. */
    my_samples.resize( 499, 3 * q + 5, 0.0 );
    my_probs.resize( 499, n_mixtures * q, 0.0 );

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed( seed );

    /** gamma distribution initialization */
    std::gamma_distribution<T>  dist_a(  ha, 1.0 / la );
    std::gamma_distribution<T>  dist_g(  hg, 1.0 / lg );
    std::gamma_distribution<T>  dist_e(  he, 1.0 / le );
    sigma_a  = 1.0 / dist_a ( generator );
    sigma_g  = 1.0 / dist_g ( generator );
    sigma_e  = 1.0 / dist_e ( generator );

    /** resize beta_a */
    beta_a.resize( 0, 0 ); 
    beta_a.resize( 1, 1 ); 
    beta_a[ 0 ] = 0.316;

    beta_c.resize( 1, w1, 0.0 );
    alpha_c.resize( w2, q, 0.0 );

    /** compute column 2-norm */
    A2norm.resize( 1, 1, 0.0 );
    for ( size_t i = 0; i < n; i ++ ) A2norm[ 0 ] += A[ i ] * A[ i ];

    M2norm.resize( 1, q, 0.0 );
    for ( size_t j = 0; j < q; j ++ )
      for ( size_t i = 0; i < n; i ++ )
        M2norm[ j ] += M[ j * n + i ] * M[ j * n + i ];

    C1_2norm.resize( 1, w1, 0.0);
    for ( size_t j = 0; j < w1; j ++ )
      for ( size_t i = 0; i < n; i ++ )
        C1_2norm[ j ] += C1[ j*n + i ] * C1[ j*n + i ];

    C2_2norm.resize( 1, w2, 0.0);
    for ( size_t j = 0; j < w2; j ++ )
      for ( size_t i = 0; i < n; i ++ )
        C2_2norm[ j ] += C2[ j*n + i ] * C2[ j*n + i ];


    /** Initialize */
    S_k.resize( 1, n_mixtures, (int)( 0.01 * q ) );
    S_k[ 1 ] = (int)( 0.05 * q );
    S_k[ 2 ] = (int)( 0.05 * q );
    S_k[ 3 ] = (int)( 0.89 * q );
    Vk_det.resize( 1, n_mixtures, 1.0 );
    Sigma_Mixture.resize( n_mixtures );
    mu_mixture.resize( n_mixtures );
    Vk_inv.resize( n_mixtures );
    Sigma_det.resize( 1, n_mixtures, 1.0 );
    prop.resize( 1, n_mixtures, 0.0 );
    w.resize( d, 1, 0.0 );

    /** resize Psi_0 */
    Psi_0.resize( d, d );
    Psi_0( 0, 0 ) = Psi( 0, 0 );
    Psi_0( 0, 1 ) = Psi( 0, 1 );
    Psi_0( 1, 0 ) = Psi( 1, 0 );
    Psi_0( 1, 1 ) = Psi( 1, 1 );

    for ( size_t k = 0; k < n_mixtures; k ++ ) 
    {
      Sigma_Mixture[ k ].resize( d, d, 0.0 );
      Vk_inv[ k ].resize( d, d, 0.0 );
      Vk_inv[ k ]( 0, 0 ) = 1.0; Vk_inv[ k ]( 1, 1 ) = 1.0;
      mu_mixture[ k ].resize( d, 1, 0.0 );
    }

    w_pg.resize( q, 3, 0.0 );
    b_multi.resize( q, 3, 0.0 ); 
    for ( int j = 0; j < q; j ++ )
    {
      b_multi( j, 0 ) = 1.5;
      b_multi( j, 1 ) = -0.2;
      b_multi( j, 2 ) = 1.5;
    }
    a_multi.resize( q, 3, 0.5 );

    r_jk.resize( q, 0 );
    sigma_multi.resize( 3, 1, 1.0 );

  };


  void Residual( size_t it )
  {
    /** res1 = Y - M * beta_m - beta_a * A */
    res1 = Y;
    for ( size_t i = 0; i < n; i ++ )
      res1[ i ] -= beta_a[ 0 ] * A[ i ];
    xgemm( "N", "N", n, 1, q, -1.0, M.data(), n,
        beta_m.data(), q, 1.0, res1.data(), n );

    //xgemm( "N", "N", n, 1, w1, -1.0, C1.data(), n,
    //    beta_c.data(), w1, 1.0, res1.data(), n );

    /** res2 = M - A * alpha_a - C * alpha_c */
    res2 = M;
    xgemm( "N", "N", n, q, 1, -1.0, A.data(), n,
        alpha_a.data(), 1, 1.0, res2.data(), n );

    //xgemm( "N", "N", n, q, w2, -1.0, C2.data(), n,
    //    alpha_c.data(), w2, 1.0, res2.data(), n );
  };


   void Iteration( size_t burnIn, size_t it )
   {

     if ( it % 10 == 0 )
     {
       //printf( "Iter %4lu sigma_multi %.3E sigma_g %.3E sigma_a %.3E pi_mixtures1 %.3E pi_mixtures2 %.3E pi_mixtures3 %.3E pi_mixtures4 %.3E \n", 
       //it, sigma_multi, sigma_g, sigma_a, pi_mixtures[ 0 ], pi_mixtures[ 1 ], pi_mixtures[ 2 ], pi_mixtures[ 3 ] ); fflush( stdout ); 
     }
     /** Update res1, res2 */
     if ( it == 0 )
     {
       Residual( it );
       
       for ( int j = 0; j < q; j ++ )
       {
         a_multi( j, 0 ) = logit( 0.89 );
         a_multi( j, 1 ) = logit( 0.05 / ( 1.0 - 0.89 ) );
         a_multi( j, 2 ) = logit( 0.05 / ( 1.0 - 0.89 - 0.05 ) );
       }

       corrD_a.resize( q, 3, 0 );
       xgemm( "N", "N", q, 3, q, 1.0, corrD_orig.data(), q,
         a_multi.data(), q, 1.0, corrD_a.data(), q );

       corrD_a_orig = corrD_a;
     }

     /** sigma_e and sigma_g */
     T he1 = he + n / 2.0;
     T hg1 = q * ( n / 2.0 ) + hg;
     T le1 = 0.0;
     T lg1 = 0.0;
     for ( size_t i = 0; i < n; i ++ )
       le1 += res1[ i ] * res1[ i ];
     for ( size_t i = 0; i < n * q; i ++ )
       lg1 += res2[ i ] * res2[ i ];
     le1 = 1.0 / ( le1 / 2.0 + le );
     lg1 = 1.0 / ( lg1 / 2.0 + lg );
     std::gamma_distribution<T> dist_e( he1, le1 );
     std::gamma_distribution<T> dist_g( hg1, lg1 );
     sigma_e  = 1.0 / dist_e( generator );
     sigma_g  = 1.0 / dist_g( generator );

     /** var_a */
     var_a.resize( 1, 1, 0.0 );
     var_a[ 0 ] = sigma_e / ( sigma_e / sigma_a + A2norm[ 0 ] ); 

     hmlp::Data<T> temp( d, 1, 0.0 );
     for ( size_t k = 0; k < 1; k ++ )
     {
       MultiVariableNormal<T> my_mvn( temp, Vk_inv[ k ] );
       Vk_det[ k ] = my_mvn.LogDeterminant();
     }

     Data<T> mu1( 1, 1, temp[ 0 ] );
     Data<T> mu2( 1, 1, temp[ 1 ] );
     Data<T> sigma1( 1, 1, Vk_inv[ 1 ]( 0, 0 ) );
     Data<T> sigma2( 1, 1, Vk_inv[ 2 ]( 1, 1 ) );

     MultiVariableNormal<T> my_mvn1( mu1, sigma1 );
     Vk_det[ 1 ] = my_mvn1.LogDeterminant();

     MultiVariableNormal<T >my_mvn2( mu2, sigma2 );
     Vk_det[ 2 ] = my_mvn2.LogDeterminant();

     if ( it % 100 == 0 )
     {
       printf( "sigma_multi0 %.4E, sigma_multi1 %.4E, sigma_multi2 %.4E \n", sigma_multi[ 0 ], sigma_multi[ 1 ], sigma_multi[ 2 ] ); fflush( stdout );
     }

     vector<Data<T>> Wishart_m( n_mixtures );
     for ( size_t k = 0; k < n_mixtures; k ++ ) 
     {
       Wishart_m[ k ] = Psi_0;
     }

     for ( size_t j = 0; j < q; j ++ )
     {
       /** update res1, res2 */
       T w_beta_mj = 0.0;
       T w_alpha_aj = 0.0;
       for ( size_t i = 0; i < n; i ++ )
       {
         w_beta_mj += M( i, j ) * ( res1[ i ] + M( i, j ) * beta_m[ j ] );
         w_alpha_aj += A[ i ] * M[ j * n + i ];
       }
       w_beta_mj /= sigma_e;
       w_alpha_aj /= sigma_g;
       w[ 0 ] = w_beta_mj; w[ 1 ] = w_alpha_aj;

       for ( size_t k = 0; k < n_mixtures; k ++ ) 
       {
         prop[ k ] = std::log( pi_mixtures( j, k ) );
       }

       vector<Data<T>> Sigma_temp( n_mixtures );
       for ( size_t k = 0; k < n_mixtures; k ++ ) 
       {
         Sigma_temp[ k ] = Vk_inv[ k ];
       }

       T old_beta_m = beta_m[ j ];
       T old_alpha_a = alpha_a[ j ];

       for ( size_t k = 0; k < 1; k ++ ) 
       {
         Sigma_temp[ k ]( 0, 0 ) += M2norm[ j ] / sigma_e;
         Sigma_temp[ k ]( 1, 1 ) += A2norm[ 0 ] / sigma_g;
         MultiVariableNormal<T> my_mvn3( temp, Sigma_temp[ k ] );
         Sigma_Mixture[ k ] = my_mvn3.Inverse();
         mu_mixture[ k ] = Sigma_Mixture[ k ] * w;
         Sigma_det[ k ] = my_mvn3.LogDeterminant();
         prop[ k ] += 0.5 * ( mu_mixture[ k ][ 0 ] * w[ 0 ] + mu_mixture[ k ][ 1 ] * w[ 1 ] ) + 0.5 * ( - Sigma_det[ k ] + Vk_det[ k ] );
       }


       Sigma_temp[ 1 ]( 0, 0 ) += M2norm[ j ] / sigma_e;
       Sigma_temp[ 1 ]( 1, 1 ) += A2norm[ 0 ] / sigma_g;
       Data<T> mu4( 1, 1, temp[ 0 ] );
       Data<T> sigma4( 1, 1, Sigma_temp[ 1 ]( 0, 0 ) );
       MultiVariableNormal<T> my_mvn4( mu4, sigma4 );
       Sigma_Mixture[ 1 ]( 0, 0 ) = 1.0 / sigma4[ 0 ];
       mu_mixture[ 1 ][ 0 ] = Sigma_Mixture[ 1 ]( 0, 0 ) * w[ 0 ];
       Sigma_det[ 1 ] = std::log( std::abs( Sigma_temp[ 1 ]( 0, 0 ) ) );
       prop[ 1 ] += 0.5 * ( mu_mixture[ 1 ][ 0 ] * w[ 0 ] ) + 0.5 * ( - Sigma_det[ 1 ] + Vk_det[ 1 ] );

       Sigma_temp[ 2 ]( 0, 0 ) += M2norm[ j ] / sigma_e;
       Sigma_temp[ 2 ]( 1, 1 ) += A2norm[ 0 ] / sigma_g;
       Data<T> sigma5( 1, 1, Sigma_temp[ 2 ]( 1, 1 ) );
       MultiVariableNormal<T> my_mvn5( mu4, sigma5 );
       Sigma_Mixture[ 2 ]( 1, 1 ) = 1.0 / sigma5[ 0 ];
       mu_mixture[ 2 ][ 1 ] = Sigma_Mixture[ 2 ]( 1, 1 ) * w[ 1 ];
       Sigma_det[ 2 ] = std::log( std::abs( Sigma_temp[ 2 ]( 1, 1 ) ) );
       prop[ 2 ] += 0.5 * ( mu_mixture[ 2 ][ 1 ] * w[ 1 ] ) + 0.5 * ( - Sigma_det[ 2 ] + Vk_det[ 2 ] );

       T max_prop = prop[ 0 ];
       for ( size_t k = 0; k < n_mixtures; k++ ) {
           if ( prop[ k ] > max_prop ) max_prop = prop[ k ]; 
       }

       for ( size_t k = 0; k < n_mixtures; k ++ ) {
           prop[ k ] = std::exp( prop[ k ] - max_prop );
           //prop[ k ] = std::exp( prop[ k ] );
        }

       T sum_prop = 0.0;
       for ( size_t k = 0; k < n_mixtures; k ++ ) {
           sum_prop += prop[ k ]; 
       }

       for ( size_t k = 0; k < n_mixtures; k ++ ) {
           prop[ k ] /= sum_prop;
        }

       std::discrete_distribution<int> dist_r ( { prop[ 0 ], prop[ 1 ], prop[ 2 ], prop[ 3 ] } );
       r_jk[ j ] = dist_r( generator );


       if ( r_jk[ j ] == 0 ) {
       	  MultiVariableNormal<T> my_mvn6( mu_mixture[ r_jk[ j ] ], Sigma_Mixture[ r_jk[ j ] ] );
          Data<T> bvn_sample = my_mvn6.SampleFrom( 1 );
          beta_m[ j ] = bvn_sample[ 0 ];
          alpha_a[ j ] = bvn_sample[ 1 ];
       }

       if ( r_jk[ j ] == 1 ) {
	        //MultiVariableNormal<T> my_mvn6( mu_mixture[ 1 ][ 0 ], Sigma_Mixture[ 1 ]( 0, 0 ) );
	        //Data<T> bvn_sample = my_mvn6.SampleFrom( 1 );
	        std::normal_distribution<T> dist_norm_m0( mu_mixture[ 1 ][ 0 ], std::sqrt( Sigma_Mixture[ 1 ]( 0, 0 ) ) );
          beta_m[ j ] = dist_norm_m0( generator );
          alpha_a[ j ] = 0;
       }

       if ( r_jk[ j ] == 2 ) {
          //MultiVariableNormal<T> my_mvn6( mu_mixture[ 2 ][ 1 ], Sigma_Mixture[ 2 ]( 1, 1 ) );
          //Data<T> bvn_sample = my_mvn6.SampleFrom( 1 );
          std::normal_distribution<T> dist_norm_m0( mu_mixture[ 2 ][ 1 ], std::sqrt( Sigma_Mixture[ 2 ]( 1, 1 ) ) );
          alpha_a[ j ] = dist_norm_m0( generator );
          beta_m[ j ] = 0;
       }

       if ( r_jk[ j ] == 3 ) {
          beta_m[ j ] = 0;
          alpha_a[ j ] = 0;
       }

       Wishart_m[ r_jk[ j ] ]( 0, 0 ) += beta_m[ j ] * beta_m[ j ];
       Wishart_m[ r_jk[ j ] ]( 0, 1 ) += beta_m[ j ] * alpha_a[ j ];
       Wishart_m[ r_jk[ j ] ]( 1, 0 ) += beta_m[ j ] * alpha_a[ j ];
       Wishart_m[ r_jk[ j ] ]( 1, 1 ) += alpha_a[ j ] * alpha_a[ j ];

       for ( size_t i = 0; i < n; i ++ )
       {
        res1[ i ] = res1[ i ] + ( old_beta_m - beta_m[ j ] ) * M( i, j );
        res2[ j*n + i ] = res2[ j*n + i ] + ( old_alpha_a - alpha_a[ j ] ) * A[ i ];
       }

       if ( it > burnIn && it % 10 == 0 ) {
	       my_probs( count, 4* (int)j ) = prop[ 0 ];
	       my_probs( count, 4* (int)j + 1 ) = prop[ 1 ];
	       my_probs( count, 4* (int)j + 2 ) = prop[ 2 ];
	       my_probs( count, 4* (int)j + 3 ) = prop[ 3 ];
       }

     } /** end for each j < q */

     /** update V_k */
     vector<size_t> r_count( n_mixtures, 0 );
     for ( size_t j = 0; j < q; j ++ )
     {
       r_count[ r_jk[ j ] ] += 1;
     }

     if ( it % 100 == 0 ) {
     printf( "r_jk0 %d r_count1 %d r_count2 %d r_count3 %d r_count4 %d \n", r_jk[ 0 ], r_count[ 0 ], r_count[ 1 ], r_count[ 2 ], r_count[ 3 ] ); fflush( stdout );
     }

     for ( size_t k = 0; k < 1; k ++ )
     {
       //beta_distribution<T> dist_pi_mixtures( r_count[ k ] + S_k[ k ] , 1 );
       //pi_mixtures[ k ] = dist_pi_mixtures( generator );

       //pi_mixtures[ k ] = ( r_count[ k ] + 0 ) / ( q + 4 * 0 );
       MultiVariableNormal<T> my_mvn( temp, Wishart_m[ k ] );
       Data<T> invWishart_m = my_mvn.Inverse();

       /** Catch the output with a pointer. */
       T *samples = wishart_sample( d, df + r_count[ k ], invWishart_m.data() );
       for ( size_t s = 0; s < Vk_inv[ k ].size(); s ++ )
         Vk_inv[ k ][ s ] = samples[ s ];
       /** Free the corresponding memory space allocated in wishart_sample. */
       free( samples );

       MultiVariableNormal<T> my_mvn9( temp, Vk_inv[ k ] );
       Data<T> Vk_temp = my_mvn9.Inverse();

       //Data<T> Vk_temp = Wishart_m[ k ];
       //for ( size_t s = 0; s < Wishart_m[ k ].size(); s++ )
       //  Vk_temp[ s ] /= ( r_count[ k ] + df + d + 1 );
       //MultiVariableNormal<T> my_mvn9( temp, Vk_temp );
       //Vk_inv[ k ] = my_mvn9.Inverse();

       if ( it % 10000 == 0 ) {
       Vk_temp.Print();
       printf("Wishart %d \n", k); fflush( stdout ); }
     }

     MultiVariableNormal<T> my_mvn10( temp, Wishart_m[ 1 ] );
     Data<T> invWishart_m = my_mvn10.Inverse();

     /** Catch the output with a pointer. */
     T *samples = wishart_sample( d, df + r_count[ 1 ], invWishart_m.data() );
     for ( size_t s = 0; s < Vk_inv[ 1 ].size(); s ++ )
       Vk_inv[ 1 ][ s ] = samples[ s ];
     /** Free the corresponding memory space allocated in wishart_sample. */
     free( samples );

     MultiVariableNormal<T> my_mvn11( temp, Vk_inv[ 1 ] );
     Data<T> Vk_temp = my_mvn11.Inverse();

     Data<T> sigma3( 1, 1, Vk_temp( 0, 0 ) );
     Vk_inv[ 1 ]( 0, 0 ) = 1.0 / sigma3[ 0 ];

     Vk_inv[ 1 ]( 1, 0 ) = 0.0;
     Vk_inv[ 1 ]( 0, 1 ) = 0.0;
     Vk_inv[ 1 ]( 1, 1 ) = 0.0;

     if ( it % 10000 == 0 )  {
     Vk_temp.Print();
     printf("Wishart %d \n", 1); fflush( stdout ); }

     MultiVariableNormal<T> my_mvn12( temp, Wishart_m[ 2 ] );
     invWishart_m = my_mvn12.Inverse();

     /** Catch the output with a pointer. */
     samples = wishart_sample( d, df + r_count[ 2 ], invWishart_m.data() );
     for ( size_t s = 0; s < Vk_inv[ 2 ].size(); s ++ )
       Vk_inv[ 2 ][ s ] = samples[ s ];
     /** Free the corresponding memory space allocated in wishart_sample. */
     free( samples );

     MultiVariableNormal<T> my_mvn13( temp, Vk_inv[ 2 ] );
     Vk_temp = my_mvn13.Inverse();

     Data<T> sigma4( 1, 1, Vk_temp( 1, 1 ) );
     Vk_inv[ 2 ]( 1, 1 ) = 1.0 / sigma4[ 0 ];

     Vk_inv[ 2 ]( 0, 0 ) = 0.0;
     Vk_inv[ 2 ]( 0, 1 ) = 0.0;
     Vk_inv[ 2 ]( 1, 0 ) = 0.0;

     if ( it % 10000 == 0 ) {
     Vk_temp.Print();
     printf("Wishart %d \n", 2); fflush( stdout ); }

     /** update beta_a */
     T mu_a = 0.0;
     old = beta_a[ 0 ];
     for ( size_t i = 0; i < n; i ++ )
       mu_a += A[ i ] * ( res1[ i ] + beta_a[ 0 ] * A[ i ] );
     mu_a *= ( var_a[ 0 ] / sigma_e );
     std::normal_distribution<T> dist_beta_a( mu_a, std::sqrt( var_a[ 0 ] ) );
     beta_a[ 0 ] = dist_beta_a( generator );
     for ( size_t i = 0; i < n; i ++ )
     {
       res1[ i ] = res1[ i ] + ( old - beta_a[ 0 ] ) * A[ i ];
     }

     /** update sigma_a */
     std::gamma_distribution<T> dist_a( 0.5 +  ha, 1.0 / ( beta_a[ 0 ] * beta_a[ 0 ] / 2.0 + la ) );
     sigma_a  = 1.0 / dist_a ( generator );

     /** update w_pg */
     PolyaGamma pg(1);
     int num_ = 1;
     for ( int j = 0; j < q; j ++ )
     {
       //rpg_devroye( & ( w_pg( j, 0 ) ), & num_, & ( b_multi( j, 0 ) ), & num_ );
      w_pg( j, 0 ) = pg.draw( 1, b_multi( j, 0 ) );
      if ( r_jk[ j ] == 3 )
      {
        w_pg( j, 1 ) = 0.0;
        w_pg( j, 2 ) = 0.0;
      }
      if ( r_jk[ j ] == 1 )
      {
        w_pg( j, 1 ) = pg.draw( 1, b_multi( j, 1 ) );
        w_pg( j, 2 ) = 0.0;
      }
      if ( r_jk[ j ] == 2 || r_jk[ j ] == 0 )
      {
        w_pg( j, 1 ) = pg.draw( 1, b_multi( j, 1 ) );
        w_pg( j, 2 ) = pg.draw( 1, b_multi( j, 2 ) );
      }


      if ( it % 100 == 0 ) {
        if ( j % 200 == 0 ) {
        printf( "r_jk0 %d w_pg0 %.2E w_pg1 %.2E w_pg2 %.2E \n", r_jk[ j ], w_pg( j, 0 ), w_pg( j, 1 ), w_pg( j, 2 ) ); fflush( stdout );
        }
      }
     }

     /** update b_multi */
     /** corrD * a_multi */
     for ( size_t j = 0; j < q; j ++ )
     {
       for ( size_t k = 0; k < 3; k ++ )
       {
         corrD_a( j, k ) = corrD_a_orig( j, k ) / sigma_multi[ k ];
       }
     }

     for ( int j = 0; j < q; j ++ )
     {
       corrD_a( j, 0 ) -= 0.5;
       if ( r_jk[ j ] == 3 )
       {
         corrD_a( j, 0 ) += 1;
       }
       if ( r_jk[ j ] == 1 )
       {
         corrD_a( j, 1 ) += 0.5;
       }
       if ( r_jk[ j ] == 2 || r_jk[ j ] == 0 )
       {
         corrD_a( j, 1 ) -= 0.5;
       }
       if ( r_jk[ j ] == 2 )
       {
         corrD_a( j, 2 ) += 0.5;
       }
       if ( r_jk[ j ] == 0 )
       {
         corrD_a( j, 2 ) -= 0.5;
       }
     }

     std::vector<size_t> I(q);
     std::vector<size_t> J(1);
     for ( size_t i = 0; i < q; i ++ ) I[ i ] = i;

     for ( int k = 0; k < 3; k ++ )
     {
       for ( size_t j = 0; j < q; j ++ )
       {
         for ( size_t i = 0; i < q; i ++ ) 
         {
           corrD( j, i ) = corrD_orig( j, i ) / sigma_multi[ k ];
         }
       }

       hmlp::Data<T> V_w = corrD;
       for ( int j = 0; j < q; j ++ )
       {
         V_w( j, j ) += w_pg( j, k );
       }
       J[ 0 ] = k;

       //MultiVariableNormal<T> b_mvn0( V_w( I, J ), V_w );
       //hmlp::Data<T> invV_w = b_mvn0.Inverse();
       //auto tmp = corrD_a( I, J );
       //hmlp::Data<T> mu_w = invV_w * tmp;

       //MultiVariableNormal<T> b_mvn1( mu_w, invV_w );
       //hmlp::Data<T> mvn_sample = b_mvn1.SampleFrom( 1 );

       //auto tmp = corrD_a(I, J);
       MultiVariableNormal<T> b_mvn(corrD_a(I, J), V_w);
       hmlp::Data<T> mvn_sample = b_mvn.SampleFromInverseCov(1);

       for ( int j = 0; j < q; j ++ )
       {
         b_multi( j, k ) = mvn_sample[ j ];
         if ( j % 200 == 0 && it % 100 == 0 ) {
           printf( "it %d b_multi0 %.2E b_multi1 %.2E b_multi2 %.2E \n", it, b_multi( j, 0 ), b_multi( j, 1 ), b_multi( j, 2 ) ); fflush( stdout );
         }
       }
     }

     /** update a_multi */
     /** corrD * b_multi */
     hmlp::Data<T> corrD_b_orig; corrD_b_orig.resize( q, 3, 0 );
     xgemm( "N", "N", q, 3, q, 1.0, corrD_orig.data(), q,
        b_multi.data(), q, 1.0, corrD_b_orig.data(), q );

     hmlp::Data<T> corrD_b; corrD_b.resize( q, 3, 0 );
     for ( size_t j = 0; j < q; j ++ )
     {
       for ( size_t k = 0; k < 3; k ++ )
       {
         corrD_b( j, k ) = corrD_b_orig( j, k ) / sigma_multi[ k ];
       }
     }

     if ( false ) {
     hmlp::Data<T> V_a = corrD;
     for ( size_t j = 0; j < q; j ++ )
     {
       V_a( j, j ) += 1.0 / sigma_a_multi;
     }

     MultiVariableNormal<T> a_mvn0( V_a( I, J ), V_a );
     hmlp::Data<T> invV_a = a_mvn0.Inverse();

     for ( size_t k = 0; k < 3; k ++ )
     {
       J[ 0 ] = k;
       auto tmp = corrD_b( I, J );
       hmlp::Data<T> mu_a = invV_a * tmp;
       MultiVariableNormal<T> a_mvn1( mu_a, invV_a );
       hmlp::Data<T> mvn_sample = a_mvn1.SampleFrom( 1 );
       for ( size_t j = 0; j < q; j ++ )
       {
         a_multi( j, k ) = mvn_sample[ j ];
       }
     }
     }

     /** update sigma_multi */
     //hmlp::Data<T> bcorrDb; bcorrDb.resize( 3, 1, 0 );
     for ( size_t k = 0; k < 3; k ++ )
     {
       T hu1 = hu + q / 2.0;
       T lu1 = 0.0;
       //xgemm( "T", "N", 3, 3, q, 1.0, b_multi.data(), q,
       //   corrD_b.data(), q, 1.0, bcorrDb.data(), 3 );
       for ( size_t j = 0; j < q; j ++ )
       {
         lu1 += ( b_multi( j, k ) - a_multi( j, k ) ) * ( corrD_b_orig( j, k ) - corrD_a_orig( j, k ) );
       }
       lu1 = 1.0 / ( lu1 / 2.0 + lu );
       std::gamma_distribution<T> dist_u( hu1, lu1 );
       if ( it % 100 == 0 ) 
       {  
         printf( "it %d lu1 %.2E \n", it, lu1 ); fflush( stdout );
       }
       sigma_multi[ k ]  = 1.0 / dist_u( generator );
     }

     /** update pi_mixtures */
     for ( int j = 0; j < q; j ++ )
     {
       pi_mixtures( j, 3 ) = sigmoid( b_multi( j, 0 ) );
       pi_mixtures( j, 1 ) = ( 1.0 - sigmoid( b_multi( j, 0 ) ) ) *  sigmoid( b_multi( j, 1 ) );
       pi_mixtures( j, 2 ) = ( 1.0 - sigmoid( b_multi( j, 0 ) ) ) * ( 1.0 - sigmoid( b_multi( j, 1 ) ) ) * sigmoid( b_multi ( j, 2 ) );
       pi_mixtures( j, 0 ) = 1.0 - pi_mixtures( j, 3 ) - pi_mixtures( j, 1 ) - pi_mixtures( j, 2 );

       if ( j % 200 == 0 && it % 100 == 0 ) {
         printf( "it %d pi0 %.2E pi1 %.2E pi2 %.2E \n", it, pi_mixtures( j, 0 ), pi_mixtures( j, 1 ), pi_mixtures( j, 2 ) ); fflush( stdout );
       }
     }

      if ( it > burnIn && it % 10 == 0 )
      {

        for ( int i = 0; i < q; i +=1 )
        {
          my_samples( count, 3*i   ) = beta_m[ i ];
          my_samples( count, 3*i+1 ) = alpha_a[ i ];
          my_samples( count, 3*i+2 ) = r_jk[ i ];
        }

        my_samples( count, 3* (int)q     ) = beta_a[ 0 ];
        my_samples( count, 3* (int)q + 1 ) = pi_mixtures[ 0 ];
        my_samples( count, 3* (int)q + 2 ) = pi_mixtures[ 1 ];
        my_samples( count, 3* (int)q + 3 ) = pi_mixtures[ 2 ];
        my_samples( count, 3* (int)q + 4 ) = pi_mixtures[ 3 ];

        count += 1;

      if ( count >= 499 )
      {
        string my_samples_filename = string( "results_" ) + to_string( (int)q1 ) + to_string( (int)q2 ) + string( ".txt" );
        my_samples.WriteFile( my_samples_filename.data() );

	      string my_probs_filename = string( "probs_" ) + to_string( (int)q1 ) + to_string( (int)q2 ) + string( ".txt" );
	      //hmlp::Data<T> output_mean = Mean( my_probs );
	      //output_mean.WriteFile( my_probs_filename.data() );
	      my_probs.WriteFile( my_probs_filename.data() );
      }

   }
 };


    size_t n;

    size_t w1;

  	size_t w2;

    size_t q;

    size_t q1;

    size_t q2;

    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;

    T ha  = 2.0;

    T la  = 1.0;

    T he  = 2.0;

    T le  = 1.0;

    T hg  = 2.0;

    T lg  = 1.0;

    T hu = 2.0;

    T lu = 1.0;

    T sigma_a;

    T sigma_g;

    T sigma_e;

    T sigma_a_multi = 10.0;

    int df = 2;

    hmlp::Data<T> &alpha_a;

    hmlp::Data<T> beta_a;

    hmlp::Data<T> &beta_m;

    hmlp::Data<T> &pi_mixtures;

    hmlp::Data<T> &Psi;

    hmlp::Data<T> alpha_c;

    hmlp::Data<T> beta_c;

    hmlp::Data<T> &A;

    hmlp::Data<T> A2norm;

    hmlp::Data<T> &M;

    hmlp::Data<T> M2norm;

    hmlp::Data<T> &Y;

    hmlp::Data<T> &C1;

    hmlp::Data<T> &C2;

    hmlp::Data<T> &corrD_orig;

    hmlp::Data<T> corrD;

    hmlp::Data<T> C1_2norm;

    hmlp::Data<T> C2_2norm;

    hmlp::Data<T> my_samples;
    hmlp::Data<T> my_labels;
    hmlp::Data<T> my_probs;

    hmlp::Data<T> res1;

    hmlp::Data<T> res2;

    hmlp::Data<T> var_a;

    size_t n_mixtures = 4;

    size_t d = 2;

    T old = 0.0;

    int count = 0;

    hmlp::Data<T> S_k;
    hmlp::Data<T> Vk_det;
    hmlp::Data<T> Psi_0;

    vector<Data<T>> Sigma_Mixture;
    vector<Data<T>> mu_mixture;
    vector<Data<T>> Vk_inv;
    hmlp::Data<T> Sigma_det;
    hmlp::Data<T> prop;
    hmlp::Data<T> w;

    hmlp::Data<T> w_pg;
    hmlp::Data<double> b_multi;
    hmlp::Data<T> a_multi;
    //T sigma_multi = 1.0;
    hmlp::Data<T> sigma_multi;

    hmlp::Data<T> corrD_a;
    hmlp::Data<T> corrD_a_orig;

    vector<size_t> r_jk;

  private:
};

template<typename T>
void mcmc( hmlp::Data<T> &Y,
	         hmlp::Data<T> &A,
	         hmlp::Data<T> &M,
           hmlp::Data<T> &C1,
	         hmlp::Data<T> &C2,
	         hmlp::Data<T> &beta_m,
	         hmlp::Data<T> &alpha_a,
           hmlp::Data<T> &pi_mixtures,
           hmlp::Data<T> &Psi,
           hmlp::Data<T> &corrD_orig,
	         size_t n, size_t w1, size_t w2, size_t q, size_t q1, size_t q2, size_t burnIn, size_t niter )
{
  Variables<T> variables( Y, A, M, C1, C2, beta_m, alpha_a, pi_mixtures, Psi, corrD_orig, n, w1, w2, q, q1, q2 );

  std::srand(std::time(nullptr));

  for ( size_t it = 0; it < niter; it ++ )
  {
    variables.Iteration( burnIn, it );
  }

};


};
};

#endif // ifndef MCMC_HPP
