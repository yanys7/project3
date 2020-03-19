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

#include "truncated_normal.hpp"

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
		size_t n, size_t w1, size_t w2, size_t q, size_t q1, size_t q2, size_t permute, T l01, T l02, T l11, T l12, T l21, T l22 )
	  : Y( userY ), A( userA ), M( userM ), C1( userC1 ), C2( userC2 ),
	    beta_m( userbeta_m ), alpha_a( useralpha_a ), Var( uservar )

	{
    this->n = n;
    this->w1 = w1;
    this->w2 = w2;
    this->q = q;
    this->q1 = q1;
    this->q2 = q2;
    this->permute = permute;
    this->l01 = l01;
    this->l02 = l02;
    this->l11 = l11;
    this->l12 = l12;
    this->l21 = l21;
    this->l22 = l22;

    /** Initialize my_samples here. */
    my_samples.resize( 499, 4 * q + 7, 0.0 );

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

    /** Create a random row-permutation of M. */
    vector<size_t> I( n ), J( q );
    for ( size_t i = 0; i < n; i ++ ) I[ i ] = i;
    for ( size_t j = 0; j < q; j ++ ) J[ j ] = j;
    /** Now shuffle row indices I. */
    shuffle( I.begin(), I.end(), generator );
    /** Apply row permutation. */
    M_perm = M( I, J );

    lmin.resize(1, 3, 0.0);
    lmax.resize(1, 3, 0.0);
    lmin[ 0 ] = 0.05;  lmin[ 1 ] = 0.20; lmin[ 2 ] = 0.05;
    lmax[ 0 ] = 0.50; lmax[ 1 ] = 0.90; lmax[ 2 ] = 1.00;

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

    /** res2 = M_perm - A * alpha_a - C * alpha_c */
    res2 = M_perm;
    xgemm( "N", "N", n, q, 1, -1.0, A.data(), n,
        thd_alpha_a.data(), 1, 1.0, res2.data(), n );

    //xgemm( "N", "N", n, q, w2, -1.0, C2.data(), n,
    //    alpha_c.data(), w2, 1.0, res2.data(), n );

    res2_c = M_perm;
    //xgemm( "N", "N", n, q, w2, -1.0, C2.data(), n,
    //   alpha_c.data(), w2, 1.0, res2_c.data(), n );

  };

   T normal_pdf(T mu, T sigma, T value)
   {
      return  ( 1 / ( sigma * std::sqrt(2*M_PI) ) ) * std::exp( -0.5 * std::pow( (value-mu)/sigma, 2.0 ) );
   }

   T PostDistribution( T lambda0, T lambda1, T lambda2 )
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
	        logc2 += std::log ( normal_pdf ( alpha_a_tmp[ j ] * A[ i ], std::sqrt( sigma_g ), M_perm(i, j) ) );
          if ( j % 500 == 0 && i % 500 == 0 && isinf(normal_pdf ( alpha_a_tmp[ j ] * A[ i ], std::sqrt( sigma_g ), M(i, j) ) ) ) {
          printf( "logc2_pdf %.3E \n", normal_pdf ( alpha_a_tmp[ j ] * A[ i ], std::sqrt( sigma_g ), M_perm(i, j) ) ); fflush( stdout );
          }
	      }
	      logc1 += std::log ( normal_pdf ( meanc1, std::sqrt( sigma_e ), Y[ i ] ) );
        if ( i % 500 == 0 && isinf(normal_pdf ( meanc1, std::sqrt( sigma_e ), Y[ i ] )) ) {
        printf( "logc1_pdf %.3E \n", normal_pdf ( meanc1, std::sqrt( sigma_e ), Y[ i ] ) ); fflush( stdout );
        }
     }
     //printf( "logc1 %.3E logc2 %.3E \n", logc1, logc2 ); fflush( stdout );
     return logc1 + logc2;
   };


   T old = 0.0;

   int count = 0;

   int accept0 = 0; int accept1 = 0; int accept2 = 0;

   int sed = std::chrono::system_clock::now().time_since_epoch().count();

   void Iteration( size_t burnIn, size_t it )
   {

     if ( it % 50000 == 0 )
     {
       printf( "Iter %4lu sigma_e %.3E sigma_g %.3E sigma_a %.3E \n",
                it, sigma_e, sigma_g, sigma_a ); fflush( stdout );
     }
     /** Update res1, res2, res2_c */
     if ( it == 0 ) Residual( it );
     if ( it == 0 )
     {
        printf( "Iter %4lu l01 %.2E l02 %.2E l11 %.2E l12 %.2E l21 %.2E l22 %.2E \n",
                it, l01, l02, l11, l12, l21, l22 ); fflush( stdout );
     }

     /** var_m = 1.0 / ( 1 / sigma_m + M2norm / sigma_e ) */
     sigma_m1 = Var[ 0 ];
     sigma_ma1 = Var[ 1 ];

     var_m1.resize( 1, q, 0.0 );
     var_ma1.resize( 1, q, 0.0 );
     for ( size_t j = 0; j < q; j ++ )
     {
       var_m1[ j ] = 1.0 / ( 1.0 / sigma_m1 + M2norm[ j ] / sigma_e );
       var_ma1[ j ] = 1.0 / ( 1.0 / sigma_ma1 + A2norm[ 0 ] / sigma_g );
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

    //if ( it % 100 == 0 )
    // {
    //   printf( "Iter %4lu he1 %.3E le1 %.3E hg1 %.3E lg1 %.3E \n",
    //            it, he1, le1, hg1, lg1 ); fflush( stdout );
    // }

     /** var_alpha_a and var_a */
     var_a.resize( 1, 1, 0.0 );
     /** var_a[ 0 ] = sigma_e / ( sigma_e / sigma_a + A2norm[ 0 ] ); */
     var_a[ 0 ] = sigma_e / A2norm[ 0 ];

     if (true) {

     for ( size_t j = 0; j < q; j ++ )
     {
       /** mu_mj, mu_alpha_aj */
       T mu_mj = 0.0;
       T mu_alpha_aj = 0.0;

       for ( size_t i = 0; i < n; i ++ )
       {
         mu_mj += M( i, j ) * ( res1[ i ] + M( i, j ) * thd_beta_m[ j ] );
         mu_alpha_aj += A[ i ] * ( res2_c[ j * n + i ] );
       }
       T mu_mj1 = mu_mj / ( ( sigma_e / sigma_m1 ) + M2norm[ j ] );
       T mu_alpha_aj1 = mu_alpha_aj / ( ( sigma_g / sigma_ma1 ) + A2norm[ 0 ] );

       /** beta_m[ j ] = randn( mu_mj, var_m[ j ] ) */
       //old = beta_m[ j ];
       //std::normal_distribution<T> dist_norm_m1( mu_mj1, std::sqrt( var_m1[ j ] ) );
       //beta_m[ j ] = dist_norm_m1( generator );

      T thd_beta;
      T beta_m_thd;

     if ( alpha_a[ j ] != 0.0 ) 
     { 
       thd_beta = std::min( lambda1, lambda0 * 1.0 / std::abs(alpha_a[ j ]) ); 
     }
     else 
     { 
       thd_beta = lambda1; 
       //thd_beta = std::min( lambda1, lambda0 * 1.0 / std::abs(alpha_a[ j ]) ); 
     }

      T c1 = 1 - normal_ms_cdf( thd_beta, mu_mj1, std::sqrt( var_m1[ j ] ) );
      T c2 = normal_ms_cdf( -thd_beta, mu_mj1, std::sqrt( var_m1[ j ] ) );
      T c3 = 1 - 2.0 * normal_ms_cdf( -thd_beta, 0.0, std::sqrt( sigma_m1 ) );

      T const0 = std::log( c3 );

      T const1 = mu_mj1 * mu_mj1 / ( 2 * var_m1[ j ] ) +
                  0.5 * std::log( var_m1[ j ] / sigma_m1 ) +
                  std::log( c1 );

      T const2 = mu_mj1 * mu_mj1 / ( 2 * var_m1[ j ] ) +
                  0.5 * std::log( var_m1[ j ] / sigma_m1 ) +
                  std::log( c2 );

      //T max_const = std::max( const0, std::max( const1, const2 ) );
      T max_const = 0.0;
      const0 = std::exp( const0 - max_const );
      const1 = std::exp( const1 - max_const );
      const2 = std::exp( const2 - max_const );

      //T sum_prop = const0 + const1 + const2;

      //const0 /= sum_prop;
      //const1 /= sum_prop;
      //const2 /= sum_prop;
      std::discrete_distribution<int> dist_r1 ( { const0, const1, const2 } );
      r1[ j ] = dist_r1 ( generator );

//    if ( it % 100 == 0 && j % 100 == 0 )
//    {
//       printf( "Iter %4lu const0 %.2E, const1 %.2E, const2 %.2E \n",
//                it, const0, const1, const2 ); fflush( stdout );
//     }


    // if ( it % 100 == 0 && j % 100 == 0 )
    // {
    //  printf( "Iter %4lu c1 %.3E c2 %.3E c3 %.3E thd_beta %.3E mu_mj1 %.3E var_m1 %.3E sigma_m1, %.3E r1 %.2E \n",
    //            it, c1, c2, c3, thd_beta,  mu_mj1, std::sqrt( var_m1[ j ] ), std::sqrt( sigma_m1 ), r1[ j ] ); fflush( stdout );
    // }

      if ( r1[ j ] == 1 ) {
          beta_m[ j ] = truncated_normal_a_sample( mu_mj1, std::sqrt( var_m1[ j ] ), thd_beta, sed );
 	        if ( isinf(beta_m[ j ]) ) { beta_m[ j ] = thd_beta; }
          beta_m_thd = beta_m[ j ];
      }
      else if ( r1[ j ] == 2 ) {
          beta_m[ j ] = truncated_normal_b_sample( mu_mj1, std::sqrt( var_m1[ j ] ), -thd_beta, sed );
	        if ( isinf(beta_m[ j ]) ) { beta_m[ j ] = -thd_beta; }
          beta_m_thd = beta_m[ j ];
       }
      else if ( r1[ j ] == 0 ) 
      { 
        beta_m[ j ] = truncated_normal_ab_sample( 0.0, std::sqrt( sigma_m1 ), -thd_beta, thd_beta, sed);
        beta_m_thd = 0.0;
      }

     //if ( it % 100 == 0 && j % 100 == 0 )
     //{
     //  printf( "Iter %4lu beta_m %.3E \n",
     //           it, beta_m[ j ] ); fflush( stdout );
     //}

       for ( size_t i = 0; i < n; i ++ )
       {
         res1[ i ] = res1[ i ] + ( thd_beta_m[ j ] - beta_m_thd ) * M( i, j );
       }

       thd_beta_m[ j ] = beta_m_thd;

      //Residual( it );

       /** alpha_a[ j ] = randn( mu_alpha_aj, var_alpha_a ) */
       //old = alpha_a[ j ];
       //std::normal_distribution<T> dist_alpha_a1( mu_alpha_aj1, std::sqrt( var_ma1[ j ] ) );
       //alpha_a[ j ] = dist_alpha_a1( generator );

       T thd_alpha;
       T alpha_a_thd;

       if ( beta_m[ j ] != 0.0 )
       { 
         thd_alpha = std::min( lambda2, lambda0 * 1.0 / std::abs(beta_m[ j ]) ); 
       }
       else 
       {
         thd_alpha = lambda2;
         //thd_alpha = std::max( lambda2, lambda0 * 1.0 / std::abs(beta_m[ j ]) ); 
       }

       c1 = 1 - normal_ms_cdf( thd_alpha, mu_alpha_aj1, std::sqrt( var_ma1[ j ] ) );
       c2 = normal_ms_cdf( -thd_alpha, mu_alpha_aj1, std::sqrt( var_ma1[ j ] ) );
       c3 = 1 - 2.0 * normal_ms_cdf( -thd_alpha, 0.0, std::sqrt( sigma_ma1) );

       T const3 = std::log( c3 );

       T const4 = mu_alpha_aj1 * mu_alpha_aj1 / ( 2 * var_ma1[ j ] ) +
                  0.5 * std::log( var_ma1[ j ] / sigma_ma1 )
                  + std::log( c1 );

       T const5 = mu_alpha_aj1 * mu_alpha_aj1 / ( 2 * var_ma1[ j ] ) +
                  0.5 * std::log( var_ma1[ j ] / sigma_ma1 )
                  + std::log( c2 );


      //max_const = std::max( const3, std::max( const4, const5 ) );
      max_const = 0.0;

      const3 = std::exp( const3 - max_const );
      const4 = std::exp( const4 - max_const );
      const5 = std::exp( const5 - max_const );

      //sum_prop = const3 + const4 + const5;

      //const3 /= sum_prop;
      //const4 /= sum_prop;
      //const5 /= sum_prop;
      
      std::discrete_distribution<int> dist_r3 ( { const3, const4, const5 } );
      r3[ j ] = dist_r3 ( generator );


    //if ( it % 100 == 0 && j % 100 == 0 )
    //{
    //   printf( "Iter %4lu alpha_c1 %.3E alpha_c2 %.3E thd_alpha %.3E mu_alpha_aj1 %.3E var_ma1[ j ] %.3E \n",
    //            it, c1, c2, thd_alpha, mu_alpha_aj1, std::sqrt( var_ma1[ j ] ) ); fflush( stdout );
    // }

      if ( r3[ j ] == 1 ) 
      {
          alpha_a[ j ] = truncated_normal_a_sample( mu_alpha_aj1, std::sqrt( var_ma1[ j ] ), thd_alpha, sed );
	        if ( isinf(alpha_a[ j ]) ) { alpha_a[ j ] = thd_alpha; }
          alpha_a_thd = alpha_a[ j ];
      }
      else if ( r3[ j ] == 2 )
      {
           alpha_a[ j ] = truncated_normal_b_sample( mu_alpha_aj1, std::sqrt( var_ma1[ j ] ), -thd_alpha, sed );
	         if ( isinf(alpha_a[ j ]) ) { alpha_a[ j ] = -thd_alpha; }
           alpha_a_thd = alpha_a[ j ];
      }

      else if ( r3[ j ] == 0 ) 
      { 
         alpha_a[ j ] = truncated_normal_ab_sample( 0.0, std::sqrt( sigma_ma1 ), -thd_alpha, thd_alpha, sed );
         alpha_a_thd = 0.0;
      }

     //if ( it % 100 == 0 && j % 100 == 0 )
     //{
     //  printf( "Iter %4lu alpha_a %.3E \n",
     //           it, alpha_a[ j ] ); fflush( stdout );
     //}

      //Residual( it );

      for ( size_t i = 0; i < n; i ++ )
      {
         res2[ j*n + i ] = res2[ j*n + i ] + ( thd_alpha_a[ j ] - alpha_a_thd ) * A[ i ];
      }

       thd_alpha_a[ j ] = alpha_a_thd;

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
     }

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

    //if ( it % 1000 == 0 )
    // {
    //   printf( "Iter %4lu sigma_m1 %.3E sigma_ma1 %.3E const6 %.3E const7 %.3E const8 %.3E const9 %.3E \n",
    //            it,  sigma_m1, sigma_ma1, const6, const7, const8, const9 ); fflush( stdout );
    // }


      /** update lambda */
      if (false) {
      hmlp::Data<T> beta_m_tmp;
      hmlp::Data<T> alpha_a_tmp;
      hmlp::Data<T> prod_tmp;
      T qn1 = 0.80;
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

      lambda0 = prod_tmp[ (int)(qn2 * q) ];
      lambda1 = beta_m_tmp[ (int)(qn2 * q) ];
      lambda2 = alpha_a_tmp[ (int)(qn2 * q) ];

      printf( "Iter %4lu lambda0 %.3E lambda1 %.3E lambda2 %.3E \n",
                it, lambda0, lambda1, lambda2 ); fflush( stdout );
      }

      if (true) {
        hmlp::Data<T> beta_m_tmp;
        hmlp::Data<T> alpha_a_tmp;
        hmlp::Data<T> prod_tmp;
        T qn1 = 0.80;
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

       lmin[ 0 ] = prod_tmp[ (int)(l01 * q) ];
       //lmin[ 0 ] = 0.0;
       lmin[ 1 ] = beta_m_tmp[ (int)(l11 * q) ];
       lmin[ 2 ] = alpha_a_tmp[ (int)(l21 * q) ];

       lmax[ 0 ] = prod_tmp[ (int)(l02 * q) ];
       //lmax[ 0 ] = 0.10;
       lmax[ 1 ] = beta_m_tmp[ (int)(l12 * q) ];
       lmax[ 2 ] = alpha_a_tmp[ (int)(l22 * q) ];

	      hmlp::Data<T> lambda_tmp; lambda_tmp.resize(1, 3, 0.0);
	      for ( int i = 0; i < 3; i++ ) {
	        //std::uniform_real_distribution<double> dist1( lmin[ i ], lmax[ i ] );
      		//lambda_tmp[ i ] = dist1( generator );
		     }
	      lambda_tmp[ 0 ] = truncated_normal_ab_sample ( lambda0, 1.0, lmin[ 0 ], lmax[ 0 ], sed );
	      lambda_tmp[ 1 ] = truncated_normal_ab_sample ( lambda1, 1.0, lmin[ 1 ], lmax[ 1 ], sed );
	      lambda_tmp[ 2 ] = truncated_normal_ab_sample ( lambda2, 1.0, lmin[ 2 ], lmax[ 2 ], sed );

        T probab = 0.0;

        if (false) {

        hmlp::Data<T> my_unif(1, 1);
       
        if (false) {
        probab = PostDistribution( lambda_tmp[ 0 ], lambda1, lambda2 )
               - PostDistribution( lambda0, lambda1, lambda2 )
               - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 0 ], lambda0, 1.0, lmin[ 0 ], lmax[ 0 ] ) )
	             + std::log( truncated_normal_ab_pdf ( lambda0, lambda_tmp[ 0 ], 1.0, lmin[ 0 ], lmax[ 0 ] ) );

       my_unif.rand( 0.0, 1.0 );
       if ( probab >= std::log (my_unif[ 0 ]) || lambda0 < lmin[ 0 ] || lambda0 > lmax[ 0 ] )
       {
          lambda0 = lambda_tmp[ 0 ];
          Residual( it );

	        if ( it > burnIn ) accept0++;
          if ( it % 20000 == 0 )
          {
           printf( "Iter %4lu updated_lambda0 %.3E \n", it, lambda0 ); fflush( stdout );
          }
      }
        }

        probab = PostDistribution( lambda0, lambda_tmp[ 1 ], lambda2 )
               - PostDistribution( lambda0, lambda1, lambda2 )
               - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 1 ], lambda1, 1.0, lmin[ 1 ], lmax[ 1 ] ) )
               + std::log( truncated_normal_ab_pdf ( lambda1, lambda_tmp[ 1 ], 1.0, lmin[ 1 ], lmax[ 1 ] ) );

       my_unif.rand( 0.0, 1.0 );
       if ( probab >= std::log (my_unif[ 0 ]) || lambda1 < lmin[ 1 ] || lambda1 > lmax[ 1 ] )
       {
          lambda1 = lambda_tmp[ 1 ];
          Residual( it );

          if ( it > burnIn ) accept1++;
          if ( it % 20000 == 0 )
          {
           printf( "Iter %4lu updated_lambda1 %.3E \n", it, lambda1 ); fflush( stdout );
          }
      }

        probab = PostDistribution( lambda0, lambda1, lambda_tmp[ 2 ] )
               - PostDistribution( lambda0, lambda1, lambda2 )
               - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 2 ], lambda2, 1.0, lmin[ 2 ], lmax[ 2 ] ) )
               + std::log( truncated_normal_ab_pdf ( lambda2, lambda_tmp[ 2 ], 1.0, lmin[ 2 ], lmax[ 2 ] ) );

       my_unif.rand( 0.0, 1.0 );
       if ( probab >= std::log (my_unif[ 0 ]) || lambda2 < lmin[ 2 ] || lambda2 > lmax[ 2 ] )
       {
          lambda2 = lambda_tmp[ 2 ];
          Residual( it );

          if ( it > burnIn ) accept2++;
          if ( it % 20000 == 0 )
          {
           printf( "Iter %4lu updated_lambda2 %.3E \n", it, lambda2 ); fflush( stdout );
          }
      }

    }

        probab = PostDistribution( lambda_tmp[ 0 ], lambda_tmp[ 1 ], lambda_tmp[ 2 ] )
		             - PostDistribution( lambda0, lambda1, lambda2 )
   - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 0 ], lambda0, 1.0, lmin[ 0 ], lmax[ 0 ] ) ) - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 1 ], lambda1, 1.0, lmin[ 1 ], lmax[ 1 ] ) ) - std::log( truncated_normal_ab_pdf ( lambda_tmp[ 2 ], lambda2, 1.0, lmin[ 2 ], lmax[ 2 ] ) )
   + std::log( truncated_normal_ab_pdf ( lambda0, lambda_tmp[ 0 ], 1.0, lmin[ 0 ], lmax[ 0 ] ) ) + std::log( truncated_normal_ab_pdf ( lambda1, lambda_tmp[ 1 ], 1.0, lmin[ 1 ], lmax[ 1 ] ) ) + std::log( truncated_normal_ab_pdf ( lambda2, lambda_tmp[ 2 ], 1.0, lmin[ 2 ], lmax[ 2 ] ) );

     if ( it % 50000 == 0 ) {
         printf( "Iter %4lu lambda0 %.3E lambda1 %.3E lambda2 %.3E \n",
                it, lambda0, lambda1, lambda2 ); fflush( stdout );
	       printf( "Iter %4lu lambda_tmp0 %.3E lambda_tmp1 %.3E lambda_tmp2 %.3E \n",
                it, lambda_tmp[ 0 ], lambda_tmp[ 1 ], lambda_tmp[ 2 ] ); fflush( stdout );
         printf( "Iter %4lu post1 %.3E post2 %.3E post3 %.3E \n",
                  it, PostDistribution( lambda_tmp[ 0 ], lambda_tmp[ 1 ], lambda_tmp[ 2 ] ), PostDistribution( lambda0, lambda1, lambda2 ) , probab ); fflush( stdout );
         printf( "Iter %4lu lmin0 %.3E lmin1 %.3E lmin2 %.3E \n",
                it, lmin[ 0 ], lmin[ 1 ], lmin [ 2 ] ); fflush( stdout );
	printf( "Iter %4lu lmax0 %.3E lmax1 %.3E lmax2 %.3E \n",
                it, lmax[ 0 ], lmax[ 1 ], lmax [ 2 ] ); fflush( stdout );

     }

      hmlp::Data<T> my_unif(1, 1); my_unif.rand( 0.0, 1.0 );
      if ( probab >= std::log (my_unif[ 0 ]) || lambda0 < lmin[ 0 ] || lambda0 > lmax[ 0 ] || lambda1 < lmin[ 1 ] || lambda1 > lmax[ 1 ] || lambda2 < lmin[ 2 ] || lambda2 > lmax[ 2 ] )
      {
        lambda0 = lambda_tmp[ 0 ];
	      lambda1 = lambda_tmp[ 1 ];
	      lambda2 = lambda_tmp[ 2 ];
        Residual( it );

        if ( it > burnIn ) accept0++;

        if ( it % 20000 == 0 )
        {
          printf( "Iter %4lu updated_lambda0 %.3E lambda1 %.3E lambda2 %.3E \n",
                it, lambda0, lambda1, lambda2 ); fflush( stdout );
        }
      }

   }


      if ( it > burnIn && it % 10 == 0 )
      {

        for ( int i = 0; i < q; i +=1 )
        {
          my_samples( count, 4*i   ) = beta_m[ i ];
          my_samples( count, 4*i+1 ) = alpha_a[ i ];
          my_samples( count, 4*i+2 ) = r1[ i ];
          my_samples( count, 4*i+3 ) = r3[ i ];
        }

        my_samples( count, 4* (int)q ) = beta_a[ 0 ];
        my_samples( count, 4* (int)q + 1 ) = lambda0;
        my_samples( count, 4* (int)q + 2 ) = lambda1;
        my_samples( count, 4* (int)q + 3 ) = lambda2;
        my_samples( count, 4* (int)q + 4 ) = sigma_e;
      	my_samples( count, 4* (int)q + 5 ) = sigma_g;
        my_samples( count, 4* (int)q + 6 ) = PostDistribution( lambda0, lambda1, lambda2 );
        count += 1;

      if ( count >= 499 )
      {
     	printf( "Iter %4lu \n", count ); fflush( stdout );
	printf( "Acceptance_lambda0 %.3E \n", accept0 * 1.0 / 5000 ); fflush( stdout );
	printf( "Acceptance_lambda1 %.3E \n", accept1 * 1.0 / 5000 ); fflush( stdout );
	printf( "Acceptance_lambda2 %.3E \n", accept2 * 1.0 / 5000 ); fflush( stdout );
        string my_samples_filename = string( "results_" ) + to_string( (int)q1 ) + to_string( (int)q2 ) + string( "_") + to_string( (int)permute ) + string( ".txt" );
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

    size_t permute;

    T l01;

    T l02;

    T l11;

    T l12;

    T l21;

    T l22;

    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;

    T ha  = 2.0;

    T la  = 1.0;

    T he  = 2.0;

    T le  = 1.0;

    T hg  = 2.0;

    T lg = 1.0;

    T km1 = 2.0;

    T lm1 = 0.09;

    T kma1 = 2.0;

    T lma1 = 0.09;

    T sigma_a;

    T sigma_g;

    T sigma_e;

    T lambda0 = 0.04;

    T lambda1 = 0.2;

    T lambda2 = 0.2;

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

    hmlp::Data<T> M_perm;

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
	         size_t n, size_t w1, size_t w2, size_t q, size_t q1, size_t q2, size_t permute, T l01, T l02, T l11, T l12, T l21, T l22, size_t burnIn, size_t niter )
{
  Variables<T> variables( Y, A, M, C1, C2, beta_m, alpha_a, Var, n, w1, w2, q, q1, q2, permute, l01, l02, l11, l12, l21, l22 );

  std::srand(std::time(nullptr));

  for ( size_t it = 0; it < niter; it ++ )
  {
    variables.Iteration( burnIn, it );
  }

};


};
};

#endif // ifndef MCMC_HPP
