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
#include <inverse_gaussian.hpp>

#include "truncated_normal.hpp"
#include <mvn.hpp>

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
		hmlp::Data<T> &uservar,
		size_t n, size_t w1, size_t w2, size_t q, size_t q1, size_t q2 )
	  : Y( userY ), A( userA ), M( userM ), C1( userC1 ), C2( userC2 ),
	    beta_m( userbeta_m ), alpha_a( useralpha_a ), Var( uservar )

	{
    this->n = n;
    this->w1 = w1;
    this->w2 = w2;
    this->q = q;
    this->q1 = q1;
    this->q2 = q2;

    /** Initialize my_samples here. */
    my_samples.resize( 499, 2 * q + 6, 0.0 );

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed( seed );

    /** gamma distribution initialization */
    std::gamma_distribution<T>  dist_a(  ha, 1.0 / la );
    std::gamma_distribution<T>  dist_g(  hg, 1.0 / lg );
    std::gamma_distribution<T>  dist_e(  he, 1.0 / le );
    sigma_a  = 1.0 / dist_a ( generator );
    sigma_g  = 1.0 / dist_g ( generator );
    sigma_e  = 1.0 / dist_e ( generator );

    r1.resize( 1, q, 1.0 );
    r3.resize( 1, q, 1.0 );

    /** resize beta_a */
    beta_a.resize( 0, 0 ); 
    beta_a.resize( 1, 1 ); 
    //beta_a.randn( 0, std::sqrt( sigma_a ) );
    beta_a[ 0 ] = 0.7;

    beta_c.resize( 1, w1, 0.0 );
    alpha_c.resize( w2, q, 0.0 );

    /** compute column 2-norm */
    A2norm.resize( 1, 1, 0.0 );
    for ( size_t i = 0; i < n; i ++ ) A2norm[ 0 ] += A[ i ] * A[ i ];

    //printf( "A2norm %.3E \n", A2norm[ 0 ] ); fflush( stdout );

    M2norm.resize( 1, q, 0.0 );
    for ( size_t j = 0; j < q; j ++ )
      for ( size_t i = 0; i < n; i ++ )
        M2norm[ j ] += M[ j * n + i ] * M[ j * n + i ];

    //printf( "M2norm %.3E \n", M2norm[ 10 ] ); fflush( stdout );

    C1_2norm.resize( 1, w1, 0.0);
    for ( size_t j = 0; j < w1; j ++ )
      for ( size_t i = 0; i < n; i ++ )
        C1_2norm[ j ] += C1[ j*n + i ] * C1[ j*n + i ];

    C2_2norm.resize( 1, w2, 0.0);
    for ( size_t j = 0; j < w2; j ++ )
      for ( size_t i = 0; i < n; i ++ )
        C2_2norm[ j ] += C2[ j*n + i ] * C2[ j*n + i ];

    lmin.resize(1, 3, 0.0);
    lmax.resize(1, 3, 0.0);
    lmin[ 0 ] = 0.01;  lmin[ 1 ] = 0.35; lmin[ 2 ] = 0.01;
    lmax[ 0 ] = 0.3; lmax[ 1 ] = 0.50; lmax[ 2 ] = 0.50;

  };

  void Residual( size_t it )
  {
    thd_beta_m.resize(0, 0);
    thd_alpha_a.resize(0, 0);
    thd_beta_m.resize(1, q, 0.0);
    thd_alpha_a.resize(1, q, 0.0);
    for ( size_t j = 0; j < q; j ++ )
    {
        if ( std::abs( beta_m[ j ] ) >= lambda1 || std::abs( beta_m[ j ] * alpha_a[ j ] ) >= lambda0 ) {
             thd_beta_m[ j ] = beta_m[ j ];
        }

        if ( std::abs( alpha_a[ j ] ) >= lambda2 || std::abs( beta_m[ j ] * alpha_a[ j ] ) >= lambda0 ) {
             thd_alpha_a[ j ] = alpha_a[ j ];
        }
    }

    /** res1 = Y - M * beta_m - beta_a * A */
    res1 = Y;
    for ( size_t i = 0; i < n; i ++ )
      res1[ i ] -= beta_a[ 0 ] * A[ i ];
    xgemm( "N", "N", n, 1, q, -1.0, M.data(), n,
        thd_beta_m.data(), q, 1.0, res1.data(), n );

    //xgemm( "N", "N", n, 1, w1, -1.0, C1.data(), n,
    //    beta_c.data(), w1, 1.0, res1.data(), n );

    /** res2 = M - A * alpha_a - C * alpha_c */
    res2 = M;
    xgemm( "N", "N", n, q, 1, -1.0, A.data(), n,
        thd_alpha_a.data(), 1, 1.0, res2.data(), n );

    //xgemm( "N", "N", n, q, w2, -1.0, C2.data(), n,
    //    alpha_c.data(), w2, 1.0, res2.data(), n );

    res2_c = M;
    //xgemm( "N", "N", n, q, w2, -1.0, C2.data(), n,
    //   alpha_c.data(), w2, 1.0, res2_c.data(), n );
  };

  void Calc_Mean( size_t it )
  {
    /** res1 = Y - M * beta_m - beta_a * A */
    thd_beta_m.resize(0, 0);
    thd_alpha_a.resize(0, 0);
    thd_beta_m.resize(1, q, 0.0);
    thd_alpha_a.resize(1, q, 0.0);
    for ( size_t j = 0; j < q; j ++ )
    {
        if ( std::abs( beta_m[ j ] ) >= lambda1 || std::abs( beta_m[ j ] * alpha_a[ j ] ) >= lambda0 ) {
             thd_beta_m[ j ] = beta_m[ j ];
        }

        if ( std::abs( alpha_a[ j ] ) >= lambda2 || std::abs( beta_m[ j ] * alpha_a[ j ] ) >= lambda0 ) {
             thd_alpha_a[ j ] = alpha_a[ j ];
        }
    }


    mean1.resize(0, 0);
    mean1.resize( n, 1, 0.0 );
    for ( size_t i = 0; i < n; i ++ )
        mean1[ i ] = beta_a[ 0 ] * A[ i ];
    xgemm( "N", "N", n, 1, q, 1.0, M.data(), n,
            thd_beta_m.data(), q, 1.0, mean1.data(), n );

    /** res2 = M - A * alpha_a - C * alpha_c */
    mean2.resize(0, 0);
    mean2.resize( n, q, 0.0);
    xgemm( "N", "N", n, q, 1, 1.0, A.data(), n,
            thd_alpha_a.data(), 1, 1.0, mean2.data(), n );
    for ( size_t i = 0; i < n; i++ ) {
      for ( size_t j = 0; j < q; j ++ ) {
        if ( mean2(i, j) != thd_alpha_a[ j ] * A[ i ] ) 
        {
          printf("not equal \n"); fflush( stdout );
        }
      }
    }
  };


   T normal_pdf(T mu, T sigma, T value)
   {
      return  ( 1 / ( sigma * std::sqrt(2*M_PI) ) ) * std::exp( -0.5 * std::pow( (value-mu)/sigma, 2.0 ) );
   }

   T PostDistribution( hmlp::Data<T> beta_m, hmlp::Data<T> alpha_a, T lambda0, T lambda1, T lambda2 )
   {
     T logc1 = 0.0;
     T logc2 = 0.0;
     hmlp::Data<T> beta_m_tmp; beta_m_tmp.resize(1, q, 0.0);
     hmlp::Data<T> alpha_a_tmp; alpha_a_tmp.resize(1, q, 0.0);
     for ( size_t j = 0; j < q; j ++ )
     {
       if ( std::abs( beta_m[ j ] ) >= lambda1 || std::abs( beta_m[ j ] * alpha_a[ j ] ) >= lambda0 ) {
	       beta_m_tmp[ j ] = beta_m[ j ];
      }
	    //logc1 += std::log ( normal_pdf ( 0.0, std::sqrt( sigma_m1 ), beta_m[ j ] ) );

      if ( std::abs( alpha_a[ j ] ) >= lambda2 || std::abs( beta_m[ j ] * alpha_a[ j ] ) >= lambda0 ) {
        alpha_a_tmp[ j ] = alpha_a[ j ];
      }
	    //logc2 += std::log ( normal_pdf ( 0.0, std::sqrt( sigma_ma1 ), alpha_a[ j ] ) );
     }

     //printf( "logc1 %.3E logc2 %.3E \n", logc1, logc2 ); fflush( stdout );
     for ( size_t i = 0; i < n; i++ ) {
	      T meanc1 = beta_a[ 0 ] * A[ i ];
	      for ( size_t j = 0; j < q; j++ ) {
	        meanc1 += M(i, j) * beta_m_tmp[ j ];
	        logc2 += std::log ( normal_pdf ( alpha_a_tmp[ j ] * A[ i ], std::sqrt( sigma_g ), M(i, j) ) );
          //if ( j % 500 == 0 && i % 500 == 0 ) {
          //printf( "logc2_pdf %.3E \n", normal_pdf ( alpha_a_tmp[ j ] * A[ i ], std::sqrt( sigma_g ), M(i, j) ) ); fflush( stdout );
          //}
	      }
	      logc1 += std::log ( normal_pdf ( meanc1, std::sqrt( sigma_e ), Y[ i ] ) );
        //if ( i % 500 == 0 ) {
        //printf( "meanc1 %.3E Y[ i ] %.3E logc1_pdf %.3E \n", meanc1, Y[ i ], normal_pdf ( meanc1, std::sqrt( sigma_e ), Y[ i ] ) ); fflush( stdout );
        //}
     }
     //printf( "logc1 %.3E logc2 %.3E \n", logc1, logc2 ); fflush( stdout );
     return logc1 + logc2;
   };


   T old = 0.0;

   int count = 0;

   int accept0 = 0; int accept1 = 0; int accept2 = 0; int accept3 = 0;

   int sed = std::chrono::system_clock::now().time_since_epoch().count();

   void Iteration( size_t burnIn, size_t it )
   {

     if ( it % 30000 == 0 )
     {
       printf( "Iter %4lu sigma_e %.3E sigma_g %.3E sigma_a %.3E \n",
                it, sigma_e, sigma_g, sigma_a ); fflush( stdout );
     }
     /** Update mean1, mean2, res1, res2 */
     if ( it == 0 ) {
       Residual( it );
       Calc_Mean( it );
     }

     //printf("Iter %4lu beta_m %.3E thd_beta_m %.3E alpha_a %.3E thd_beta_m %.3E \n",
		//it, beta_m[ 0 ], thd_beta_m[ 0 ], alpha_a[ 0 ], thd_alpha_a[ 0 ] ); fflush( stdout );
     /** var_m = 1.0 / ( 1 / sigma_m + M2norm / sigma_e ) */
     sigma_m1 = Var[ 0 ];
     sigma_ma1 = Var[ 1 ];

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

     /** var_alpha_a and var_a */
     var_a.resize( 1, 1, 0.0 );
     /** var_a[ 0 ] = sigma_e / ( sigma_e / sigma_a + A2norm[ 0 ] ); */
     var_a[ 0 ] = sigma_e / A2norm[ 0 ];

     hmlp::Data<T> Vj;
     Vj.resize( 2, 2, 0.0 );
     Vj( 0, 0 ) = 0.1; Vj( 1, 1 ) = 0.1;

     hmlp::Data<T> my_unif(1, 1);

     for ( size_t j = 0; j < q; j ++ )
     {
       hmlp::Data<T> muj;
       muj.resize( 2, 1, 0.0 );
       muj[ 0 ] = beta_m[ j ]; muj[ 1 ] = alpha_a[ j ];

       MultiVariableNormal<T> my_mvn( muj, Vj );
       Data<T> bvn_sample = my_mvn.SampleFrom( 1 );
       
       T beta_m_mh = bvn_sample[ 0 ];
       //T alpha_a_mh = bvn_sample[ 1 ];
       T alpha_a_mh = alpha_a[ j ];

       double hastings = std::log ( normal_pdf ( 0.0, std::sqrt( sigma_m1 ), beta_m_mh ) ) + std::log ( normal_pdf ( 0.0, std::sqrt( sigma_ma1 ), alpha_a_mh ) )
                      - std::log ( normal_pdf ( 0.0, std::sqrt( sigma_m1 ), beta_m[ j ] ) ) - std::log ( normal_pdf ( 0.0, std::sqrt( sigma_ma1 ), alpha_a[ j ] ) );

       T beta_m_thd = 0.0;
       if ( std::abs( beta_m_mh ) >= lambda1 || std::abs( beta_m_mh * alpha_a_mh ) >= lambda0 ) 
       {
         beta_m_thd = beta_m_mh;
       }

       T alpha_a_thd = 0.0;
       if ( std::abs( alpha_a_mh ) >= lambda2 || std::abs( beta_m_mh * alpha_a_mh ) >= lambda0 ) 
       {
         alpha_a_thd = alpha_a_mh;
       }

       for ( size_t i = 0; i < n; i ++ ) {
         hastings = hastings + std::log ( normal_pdf ( alpha_a_thd * A[ i ], std::sqrt( sigma_g ), M(i, j) ) )
                             - std::log ( normal_pdf ( thd_alpha_a[ j ] * A[ i ], std::sqrt( sigma_g ), M(i, j) ) )
                             + std::log ( normal_pdf ( mean1[ i ] - thd_beta_m[ j ] * M(i, j) + beta_m_thd * M(i, j), std::sqrt( sigma_e ), Y[ i ] ) )
                             - std::log ( normal_pdf ( mean1[ i ], std::sqrt( sigma_e ), Y[ i ] ) );

//	if ( j % 200 == 0 && i % 1000 == 0 )
//       	{
//        printf( "Iter %4lu j %2d i %2d mean1 %.3E M %.3E l2 %.3E thd_beta_m %.3E beta_m_thd %.3E \n", it, j, i, mean1[ i ], M(i, j), std::log ( normal_pdf ( mean1[ i ], std::sqrt( sigma_e ), Y[ i ] ) ), thd_beta_m[ j ], beta_m_thd ); fflush( stdout );
//       	}


       }
         
       if ( it % 1000 == 0 && j % 1000 == 0 )
       {
          printf( "Iter %4lu hastings %.3E my_unif %.3E \n", it, hastings, my_unif[ 0 ] ); fflush( stdout );
       }

       my_unif.rand( 0.0, 1.0 );
       if ( hastings > std::log( my_unif[ 0 ] ) ) {
              beta_m[ j ] = beta_m_mh;
              //alpha_a[ j ] = alpha_a_mh;
              if ( it > burnIn ) accept3++;

	      if ( it % 1000 == 0 && j % 100 == 0 )
              {
               printf( "Iter %4lu beta_m %.3E \n",
                  it, beta_m[ j ] ); fflush( stdout );
              }

	      if ( it % 1000 == 0 && j % 100 == 0 )
              {
               printf( "Iter %4lu alpha_a %.3E \n",
                 it, alpha_a[ j ] ); fflush( stdout );
              }
	    //printf( "Iter %4lu res1 %.3E res2 %.3E mean1 %.3E mean2 %.3E \n", it, res1[ 0 ], res2[ 0 ], mean1[ 0 ], mean2[ 0 ] ); fflush( stdout );

     //printf("Iter %4lu beta_m %.3E thd_beta_m %.3E alpha_a %.3E thd_beta_m %.3E \n",
     //           it, beta_m[ 0 ], thd_beta_m[ 0 ], alpha_a[ 0 ], thd_alpha_a[ 0 ] ); fflush( stdout );


       for ( size_t i = 0; i < n; i ++ )
       {
         res1[ i ] = res1[ i ] + ( thd_beta_m[ j ] - beta_m_thd ) * M( i, j );
         mean1[ i ] = mean1[ i ] + ( beta_m_thd - thd_beta_m[ j ] ) * M( i, j );
       }


       for ( size_t i = 0; i < n; i ++ )
       {
         res2[ j*n + i ] = res2[ j*n + i ] + ( thd_alpha_a[ j ] - alpha_a_thd ) * A[ i ];
         mean2[ j*n + i ] = alpha_a_thd * A[ i ];
       }

       thd_beta_m[ j ] = beta_m_thd;
       thd_alpha_a[ j ] = alpha_a_thd;

       //Calc_Mean( it );

    //printf( "Iter %4lu res1 %.3E res2 %.3E mean1 %.3E mean2 %.3E \n", it, res1[ 0 ], res2[ 0 ], mean1[ 0 ], mean2[ 0 ] ); fflush( stdout );

     //printf("Iter %4lu beta_m %.3E thd_beta_m %.3E alpha_a %.3E thd_beta_m %.3E \n",
     //           it, beta_m[ 0 ], thd_beta_m[ 0 ], alpha_a[ 0 ], thd_alpha_a[ 0 ] ); fflush( stdout );


       }

     } /** end for each j < q */


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
       mean1[ i ] = mean1[ i ] + ( beta_a[ 0 ] - old ) * A[ i ];
     }

     /** update sigma_m, sigma_a and sigma_ma */
     std::gamma_distribution<T>  dist_a( 0.5 +  ha, 1.0 / ( beta_a[ 0 ] * beta_a[ 0 ] / 2.0 + la ) );
     sigma_a  = 1.0 / dist_a ( generator );

     T const6 = 0.0;
     T const7 = 0.0;
     T const8 = 0.0;
     T const9 = 0.0;

     for ( size_t j = 0; j < q; j ++ )
      {
        const6 += 1.0;
        const7 += beta_m[ j ] * beta_m[ j ];
        const8 += 1.0;
        const9 += alpha_a[ j ] * alpha_a[ j ];
      }
      const6 /= 2.0;
      const7 /= 2.0;
      const8 /= 2.0;
      const9 /= 2.0;

      std::gamma_distribution<T>  dist_m1( const6 +  km1, 1.0 / ( const7 +  lm1 ) );
      std::gamma_distribution<T> dist_ma1( const8 + kma1, 1.0 / ( const9 + lma1 ) );
      sigma_m1  = 1.0 / dist_m1 ( generator );
      sigma_ma1 = 1.0 / dist_ma1 ( generator );

    if ( it % 1000 == 0 )
     {
       printf( "Iter %4lu sigma_m1 %.3E sigma_ma1 %.3E sigma_e %.3E sigma_g %.3E const7 %.3E const9 %.3E \n",
                it,  sigma_m1, sigma_ma1, sigma_e, sigma_g, const7, const9 ); fflush( stdout );
     }


      /** update lambda */
      if (false) {
        hmlp::Data<T> beta_m_tmp;
        hmlp::Data<T> alpha_a_tmp;
        hmlp::Data<T> prod_tmp;
        T qn1 = 0.85;
        T qn2 = 0.90;
        prod_tmp.resize(1, q, 0.0); beta_m_tmp.resize(1, q, 0.0); alpha_a_tmp.resize(1, q, 0.0);
        for ( size_t j = 0; j < q; j++ ) {
          prod_tmp[ j ] = std::abs( beta_m[ j ] * alpha_a[ j ] );
          alpha_a_tmp[ j ] = std::abs( alpha_a[ j ] );
          beta_m_tmp[ j ] = std::abs( beta_m[ j ] );
         }

       std::sort( beta_m_tmp.begin(), beta_m_tmp.end() );
       std::sort( alpha_a_tmp.begin(), alpha_a_tmp.end() );
       std::sort( prod_tmp.begin(), prod_tmp.end() );

       lmin[ 0 ] = prod_tmp[ (int)(0.90 * q) ];
       //lmin[ 0 ] = 0.0;
       lmin[ 1 ] = beta_m_tmp[ (int)(qn1 * q) ];
       lmin[ 2 ] = alpha_a_tmp[ (int)(qn1 * q) ];

       lmax[ 0 ] = prod_tmp[ (int)(0.96 * q) ];
       //lmax[ 0 ] = 0.10;
       lmax[ 1 ] = beta_m_tmp[ (int)(qn2 * q) ];
       lmax[ 2 ] = alpha_a_tmp[ (int)(qn2 * q) ];

	      hmlp::Data<T> lambda_tmp; lambda_tmp.resize(1, 3, 0.0);
	      for ( int i = 0; i < 3; i++ ) {
	        //std::uniform_real_distribution<double> dist1( lmin[ i ], lmax[ i ] );
      		//lambda_tmp[ i ] = dist1( generator );
		     }
	      lambda_tmp[ 0 ] = truncated_normal_ab_sample ( lambda0, 1.0, lmin[ 0 ], lmax[ 0 ], sed );
	      lambda_tmp[ 1 ] = truncated_normal_ab_sample ( lambda1, 1.0, lmin[ 1 ], lmax[ 1 ], sed );
	      lambda_tmp[ 2 ] = truncated_normal_ab_sample ( lambda2, 1.0, lmin[ 2 ], lmax[ 2 ], sed );

        T probab = 0.0;

        probab = PostDistribution( beta_m, alpha_a, lambda_tmp[ 0 ], lambda1, lambda2 )
               - PostDistribution( beta_m, alpha_a, lambda0, lambda1, lambda2 )
               - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 0 ], lambda0, 1.0, lmin[ 0 ], lmax[ 0 ] ) )
	             + std::log( truncated_normal_ab_pdf ( lambda0, lambda_tmp[ 0 ], 1.0, lmin[ 0 ], lmax[ 0 ] ) );

       my_unif.rand( 0.0, 1.0 );
       if ( probab >= std::log (my_unif[ 0 ]) )
       {
          lambda0 = lambda_tmp[ 0 ];
          Calc_Mean( it );

	        if ( it > burnIn ) accept0++;
          if ( it % 10 == 0 )
          {
           printf( "Iter %4lu updated_lambda0 %.3E \n", it, lambda0 ); fflush( stdout );
          }
      }

        probab = PostDistribution( beta_m, alpha_a, lambda0, lambda_tmp[ 1 ], lambda2 )
               - PostDistribution( beta_m, alpha_a, lambda0, lambda1, lambda2 )
               - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 1 ], lambda1, 1.0, lmin[ 1 ], lmax[ 1 ] ) )
               + std::log( truncated_normal_ab_pdf ( lambda1, lambda_tmp[ 1 ], 1.0, lmin[ 1 ], lmax[ 1 ] ) );

       my_unif.rand( 0.0, 1.0 );
       if ( probab >= std::log (my_unif[ 0 ]) )
       {
          lambda1 = lambda_tmp[ 1 ];
          Calc_Mean( it );

          if ( it > burnIn ) accept1++;
          if ( it % 10 == 0 )
          {
           printf( "Iter %4lu updated_lambda1 %.3E \n", it, lambda1 ); fflush( stdout );
          }
      }

        probab = PostDistribution( beta_m, alpha_a, lambda0, lambda1, lambda_tmp[ 2 ] )
               - PostDistribution( beta_m, alpha_a, lambda0, lambda1, lambda2 )
               - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 2 ], lambda2, 1.0, lmin[ 2 ], lmax[ 2 ] ) )
               + std::log( truncated_normal_ab_pdf ( lambda2, lambda_tmp[ 2 ], 1.0, lmin[ 2 ], lmax[ 2 ] ) );

       my_unif.rand( 0.0, 1.0 );
       if ( probab >= std::log (my_unif[ 0 ]) )
       {
          lambda2 = lambda_tmp[ 2 ];
          Calc_Mean( it );

          if ( it > burnIn ) accept2++;
          if ( it % 10 == 0 )
          {
           printf( "Iter %4lu updated_lambda2 %.3E \n", it, lambda2 ); fflush( stdout );
          }
      }

        //probab = PostDistribution( lambda_tmp[ 0 ], lambda_tmp[ 1 ], lambda_tmp[ 2 ] )
		   //          - PostDistribution( lambda0, lambda1, lambda2 )
   //- std::log( truncated_normal_ab_pdf ( lambda_tmp[ 0 ], lambda0, 1.0, lmin[ 0 ], lmax[ 0 ] ) ) - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 1 ], lambda1,  1.0, lmin[ 1 ], lmax[ 1 ] ) ) - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 2 ], lambda2, 1.0, lmin[ 2 ], lmax[ 2 ] ) )
  // + std::log( truncated_normal_ab_pdf ( lambda0, lambda_tmp[ 0 ], 1.0, lmin[ 0 ], lmax[ 0 ] ) ) + std::log( truncated_normal_ab_pdf ( lambda1, lambda_tmp[ 1 ], 1.0, lmin[ 1 ], lmax[ 1 ] ) ) + std::log( truncated_normal_ab_pdf ( lambda2, lambda_tmp[ 2 ], 1.0, lmin[ 2 ], lmax[ 2 ] ) );

     if ( it % 1000 == 0 ) {
         printf( "Iter %4lu lambda0 %.3E lambda1 %.3E lambda2 %.3E \n",
                it, lambda0, lambda1, lambda2 ); fflush( stdout );
	       printf( "Iter %4lu lambda_tmp0 %.3E lambda_tmp1 %.3E lambda_tmp2 %.3E \n",
                it, lambda_tmp[ 0 ], lambda_tmp[ 1 ], lambda_tmp[ 2 ] ); fflush( stdout );
         printf( "Iter %4lu post1 %.3E post2 %.3E post3 %.3E \n",
                  it, PostDistribution( beta_m, alpha_a, lambda_tmp[ 0 ], lambda_tmp[ 1 ], lambda_tmp[ 2 ] ), PostDistribution( beta_m, alpha_a, lambda0, lambda1, lambda2 ) , probab ); fflush( stdout );
         printf( "Iter %4lu lmin0 %.3E lmin1 %.3E lmin2 %.3E \n",
                it, lmin[ 0 ], lmin[ 1 ], lmin [ 2 ] ); fflush( stdout );
	printf( "Iter %4lu lmax0 %.3E lmax1 %.3E lmax2 %.3E \n",
                it, lmax[ 0 ], lmax[ 1 ], lmax [ 2 ] ); fflush( stdout );
	printf( "Iter %4lu l_tmp_p %.3E l_tmp_p %.3E lmin2 %.3E \n",
               it, std::log( truncated_normal_ab_pdf ( lambda_tmp[ 1 ], lambda1, 1.0, lmin[ 1 ], lmax[ 1 ] ) ), std::log( truncated_normal_ab_pdf ( lambda2, lambda_tmp[ 2 ], 1.0, lmin[ 2 ], lmax[ 2 ] ) ), lmin [ 2 ] ); fflush( stdout );
}
   //   hmlp::Data<T> my_unif(1, 1); my_unif.rand( 0.0, 1.0 );
   //   if ( probab >= std::log (my_unif[ 0 ]) )
   //   {
   //     lambda0 = lambda_tmp[ 0 ];
	 //     lambda1 = lambda_tmp[ 1 ];
	 //     lambda2 = lambda_tmp[ 2 ];

   //     if ( it > burnIn ) accept++;

   //     if ( it % 100 == 0 )
   //     {
   //      printf( "Iter %4lu updated_lambda0 %.3E lambda1 %.3E lambda2 %.3E \n",
   //             it, lambda0, lambda1, lambda2 ); fflush( stdout );
   //     }
   //   }

   }


      if ( it > burnIn && it % 10 == 0 )
      {

        for ( int i = 0; i < q; i +=1 )
        {
          my_samples( count, 2*i   ) = beta_m[ i ];
          my_samples( count, 2*i+1 ) = alpha_a[ i ];
        }

        my_samples( count, 2* (int)q ) = beta_a[ 0 ];
        my_samples( count, 2* (int)q + 1 ) = lambda0;
        my_samples( count, 2* (int)q + 2 ) = lambda1;
        my_samples( count, 2* (int)q + 3 ) = lambda2;
        my_samples( count, 2* (int)q + 4 ) = sigma_e;
      	my_samples( count, 2* (int)q + 5 ) = sigma_g;
        count += 1;

      if ( count >= 499 )
      {
     	printf( "Iter %4lu \n", count ); fflush( stdout );
	printf( "Acceptance_lambda0 %.3E \n", accept0 * 1.0 / 5000 ); fflush( stdout );
	printf( "Acceptance_lambda1 %.3E \n", accept1 * 1.0 / 5000 ); fflush( stdout );
	printf( "Acceptance_lambda2 %.3E \n", accept2 * 1.0 / 5000 ); fflush( stdout );
  printf( "Acceptance_effects %.3E \n", accept3 * 1.0 / 5000 ); fflush( stdout );
        string my_samples_filename = string( "results_" ) + to_string( (int)q1 ) + to_string( (int)q2 ) + string( ".txt" );
        my_samples.WriteFile( my_samples_filename.data() );
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

    T lg = 1.0;

    T km1 = 2.0;

    T lm1 = 1.0;

    T kma1 = 2.0;

    T lma1 = 1.0;

    T sigma_a;

    T sigma_g;

    T sigma_e;

    T lambda0 = 0.25;

    T lambda1 = 0.5;

    T lambda2 = 0.5;

    hmlp::Data<T> lmin;

    hmlp::Data<T> lmax;


    hmlp::Data<T> r1;

    hmlp::Data<T> r3;

    hmlp::Data<T> &alpha_a;

    hmlp::Data<T> beta_a;

    hmlp::Data<T> &beta_m;

	  hmlp::Data<T> alpha_c;

	  hmlp::Data<T> beta_c;

    hmlp::Data<T> &A;

    hmlp::Data<T> A2norm;

    hmlp::Data<T> &M;

    hmlp::Data<T> M2norm;

    hmlp::Data<T> &Y;

	  hmlp::Data<T> &C1;

	  hmlp::Data<T> &C2;

    hmlp::Data<T> &Var;

	  hmlp::Data<T> C1_2norm;

	  hmlp::Data<T> C2_2norm;

    hmlp::Data<T> my_samples;
    /** in Iteration() */

    hmlp::Data<T> res1;
    hmlp::Data<T> res2;
    hmlp::Data<T> res2_c;

    hmlp::Data<T> mean1;
    hmlp::Data<T> mean2;

    hmlp::Data<T> thd_beta_m;
    hmlp::Data<T> thd_alpha_a;

    hmlp::Data<T> var_m1;

    T sigma_m1;

    hmlp::Data<T> var_ma1;

    T sigma_ma1;

    hmlp::Data<T> var_a;

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
		       hmlp::Data<T> &Var,
	         size_t n, size_t w1, size_t w2, size_t q, size_t q1, size_t q2, size_t burnIn, size_t niter )
{
  Variables<T> variables( Y, A, M, C1, C2, beta_m, alpha_a, Var, n, w1, w2, q, q1, q2 );

  std::srand(std::time(nullptr));

  for ( size_t it = 0; it < niter; it ++ )
  {
    variables.Iteration( burnIn, it );
  }

};


};
};

#endif // ifndef MCMC_HPP
