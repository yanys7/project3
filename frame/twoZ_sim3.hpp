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

#include "wishart.hpp"
#include "pdflib.hpp"
#include "rnglib.hpp"
#include <mvn.hpp>
#include <gsl_cdf.h>

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
		size_t n, size_t w1, size_t w2, size_t q, size_t q1, size_t q2, string Z_type )
	  : Y( userY ), A( userA ), M( userM ), C1( userC1 ), C2( userC2 ),
	    beta_m( userbeta_m ), alpha_a( useralpha_a ), Var( uservar )

	{
    this->n = n;
    this->w1 = w1;
    this->w2 = w2;
    this->q = q;
    this->q1 = q1;
    this->q2 = q2;
    this->Z_type = Z_type;

    /** Initialize my_samples here. */
    my_samples.resize( 499, 4 * q + 5, 0.0 );

    /** generate synthetic data Y, A, M and E */
    //beta_a.resize( 1, 1, 0.5 );
    //beta_m.resize( 1, q, 0.0 ); beta_m.randn( 0.0, 2.0 );
    //for ( size_t i = q1; i < beta_m.size(); i ++ ) beta_m[ i ] = 0.0;
    //A.resize( n, 1 ); A.randn( 0.0, 1.0 );
    //M.resize( n, q ); M.randn( 0.0, 2.0 );
    //E.resize( n, 1 ); E.randn( 0.0, 1.0 );

    /** Y <- beta_a*A + M %*% beta_m + E */
    //Y = E;
    //for ( size_t i = 0; i < n; i ++ ) Y[ i ] += beta_a[ 0 ] * A[ i ];
    //xgemm( "N", "N", n, 1, q, 1.0, M.data(), n, beta_m.data(), q, 1.0, Y.data(), n );

    /** normalize all synthetic data */
    //Normalize( Y );
    //Normalize( A );
    //Normalize( M );

    /** beta distribution */
    //beta_distribution<T> dist_pi_m( um, vm );
    //beta_distribution<T> dist_pi_a( ua, va );
    //pi_m.resize( 1, q, 0.0);
    //pi_a.resize( 1, q, 0.0);
    //for ( size_t i = 0; i < q; i ++) 
    //{
    // pi_m[ i ]  = dist_pi_m( generator );
    // pi_a[ i ]  = dist_pi_a( generator );
    //};

    //printf( "here\n" ); fflush( stdout );

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
    //beta_a.randn( 0, std::sqrt( sigma_a ) );
    beta_a[ 0 ] = 0.0135567741;

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

    res2_c = M;
    //xgemm( "N", "N", n, q, w2, -1.0, C2.data(), n,
    //   alpha_c.data(), w2, 1.0, res2_c.data(), n );
  };

   T old = 0.0;

   int count = 0;

   void Iteration( size_t burnIn, size_t it )
   {

     if ( it % 30000 == 0 ) 
     { 
       printf( "Iter %4lu sigma_e %.3E sigma_g %.3E sigma_a %.3E \n", 
           it, sigma_e, sigma_g, sigma_a ); fflush( stdout ); 
     }
     /** Update res1, res2, res2_c */
     if ( it == 0 ) Residual( it );
     //printf( "Iter %4lu res1 %.3E res2 %.3E \n", it, res1[ 0 ], res2[ 0 ] ); fflush( stdout ); 

     /** var_m = 1.0 / ( 1 / sigma_m + M2norm / sigma_e ) */
     var_m1.resize( 1, q, 0.0 );
     var_ma1.resize( 1, q, 0.0 );
     Z1.resize( 1, q, 0.5 );
     Z2.resize( 1, q, 0.5 );
     sigma_m1.resize( 1, q, 0.0 );
     sigma_ma1.resize( 1, q, 0.0 );

     for ( size_t j = 0; j < q; j ++ )
     {
       sigma_m1[ j ] = Z1[ j ] * sigma_e;
       sigma_ma1[ j ] = Z2[ j ] * sigma_g;
     }

     for ( size_t j = 0; j < q; j ++ )
     {
       var_m1[ j ] = 1.0 / ( 1.0 / sigma_m1[ j ] + M2norm[ j ] / sigma_e );
       var_ma1[ j ] = 1.0 / ( 1.0 / sigma_ma1[ j ] + A2norm[ 0 ] / sigma_g );
     }

     /** sigma_e and sigma_g */
     T he1 = he + n / 2.0 + q / 2.0;
     T hg1 = q * ( n / 2.0 ) + hg + q / 2.0;
     T le1 = 0.0;
     T lg1 = 0.0;
     for ( size_t i = 0; i < n; i ++ )
       le1 += res1[ i ] * res1[ i ];
     for ( size_t j = 0; j < q; j ++ )
       le1 += beta_m[ j ] * beta_m[ j ] / Z1[ j ];
     for ( size_t i = 0; i < n * q; i ++ )
       lg1 += res2[ i ] * res2[ i ];
     for ( size_t j = 0; j < q; j ++ )
       lg1 += alpha_a[ j ] * alpha_a[ j ] / Z2[ j ];

     le1 = 1.0 / ( le1 / 2.0 + le );
     lg1 = 1.0 / ( lg1 / 2.0 + lg );
     std::gamma_distribution<T> dist_e( he1, le1 );
     std::gamma_distribution<T> dist_g( hg1, lg1 );
     sigma_e  = 1.0 / dist_e( generator );
     sigma_g  = 1.0 / dist_g( generator );

     //sigma_e = 0.49;

     /** var_alpha_a and var_a */
     var_a.resize( 1, 1, 0.0 );
     /** var_a[ 0 ] = sigma_e / ( sigma_e / sigma_a + A2norm[ 0 ] ); */
     var_a[ 0 ] = sigma_e / A2norm[ 0 ];

     for ( size_t j = 0; j < q; j ++ )
     {
       /** mu_mj, mu_alpha_aj */
       T mu_mj = 0.0;
       T mu_alpha_aj = 0.0;

       for ( size_t i = 0; i < n; i ++ )
       {
         mu_mj += M( i, j ) * ( res1[ i ] + M( i, j ) * beta_m[ j ] );
         mu_alpha_aj += A[ i ] * ( res2_c[ j * n + i ] );
       }
       T mu_mj1 = mu_mj / ( ( sigma_e / sigma_m1[ j ] ) + M2norm[ j ] );
       T mu_alpha_aj1 = mu_alpha_aj / ( ( sigma_g / sigma_ma1[ j ] ) + A2norm[ 0 ] );

       /** beta_m[ j ] = randn( mu_mj, var_m[ j ] ) */
       old = beta_m[ j ];
       std::normal_distribution<T> dist_norm_m1( mu_mj1, std::sqrt( var_m1[ j ] ) );
       beta_m[ j ] = dist_norm_m1( generator );

       for ( size_t i = 0; i < n; i ++ )
       {
         res1[ i ] = res1[ i ] + ( old - beta_m[ j ] ) * M( i, j );
       }

       /** alpha_a[ j ] = randn( mu_alpha_aj, var_alpha_a ) */
       old = alpha_a[ j ];
       std::normal_distribution<T> dist_alpha_a1( mu_alpha_aj1, std::sqrt( var_ma1[ j ] ) );
       alpha_a[ j ] = dist_alpha_a1( generator );

       for ( size_t i = 0; i < n; i ++ )
       {
         res2[ j*n + i ] = res2[ j*n + i ] + ( old - alpha_a[ j ] ) * A[ i ];
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
     }


     /** update sigma_m, sigma_a and sigma_ma */
      std::gamma_distribution<T>  dist_a( 0.5 +  ha, 1.0 / ( beta_a[ 0 ] * beta_a[ 0 ] / 2.0 + la ) );
      sigma_a  = 1.0 / dist_a ( generator );

     /** update Z1, Z2 */
     hmlp::Data<T> upsi( 1, 1 );
     hmlp::Data<T> up( 1, 1 );

     for ( size_t j = 0; j < q; j ++ )
     {
       T bj = beta_m[ j ] * beta_m[ j ] / ( 2 * sigma_e );
       //Z1[ j ] = -0.5 + std::sqrt( 0.25 + bj );
       //bj = alpha_a[ j ] * alpha_a[ j ] / ( 2 * Var[ 1 ] );
       //Z2[ j ] = -0.5 + std::sqrt( 0.25 + bj );

       upsi.rand( 0.0, std::exp( - bj / Z1[ j ] ) );
       T ub = - bj / std::log( upsi[ 0 ] );
       T Fub = gsl_cdf_gamma_P( ub, 0.5, 2.0 / Var[ 0 ] );
       //printf( "Iter %4lu ub %.3E Fub %.3E \n", it, ub, Fub ); fflush( stdout );
       if ( Fub < 1E-04 ) Fub = 1E-04;

       up.rand( Fub, 1.0 );
       T eta = gsl_cdf_gamma_Pinv( up[ 0 ], 0.5, 2.0 / Var[ 0 ] );
       if ( eta > 1E08) eta = 1E08;
       Z1[ j ] = eta;
       //printf( "Iter %4lu up %.3E inv_bj %.3E eta %.3E\n", it, up[ 0 ], 1.0 / bj, eta ); fflush( stdout );

       bj = alpha_a[ j ] * alpha_a[ j ] / ( 2 * sigma_g );

       upsi.rand( 0.0, std::exp( - bj / Z2[ j ] ) );
       ub = - bj / std::log( upsi[ 0 ] );
       Fub = gsl_cdf_gamma_P( ub, 0.5, 2.0 / Var[ 1 ] );
       //printf( "Iter %4lu ub %.3E Fub %.3E \n", it, ub, Fub ); fflush( stdout );
       if ( Fub < 1E-04 ) Fub = 1E-04;

       up.rand( Fub, 1.0 );
       eta = gsl_cdf_gamma_Pinv( up[ 0 ], 0.5, 2.0 / Var[ 1 ] );
       if ( eta > 1E08) eta = 1E08;
       Z2[ j ] = eta;
       //printf( "Iter %4lu up %.3E inv_bj %.3E eta %.3E\n", it, up[ 0 ], 1.0 / bj, eta ); fflush( stdout );

     }

     /** update Var[ 0 ], Var[ 1 ]  */
     if (false) {
     T mean_Zj = 0.0;
     for ( size_t j = 0; j < q; j ++ ) {
       mean_Zj += std::abs( beta_m[ j ] / std::sqrt( sigma_e * Var[ 0 ] ) ) + 1.0 / Var[ 0 ];
     }
     Var[ 0 ] = 2 * q / mean_Zj;

     mean_Zj = 0.0;
     for ( size_t j = 0; j < q; j ++ ) {
       mean_Zj += std::abs( alpha_a[ j ] / std::sqrt( sigma_g * Var[ 1 ] ) ) + 1.0 / Var[ 1 ];
     }
     Var[ 1 ] = 2 * q / mean_Zj;

     //printf( "Iter %4lu Var_0 %.3E Var_1 %.3E \n", it, Var[ 0 ], Var[ 1 ] ); fflush( stdout );
     }

     if (true) {
     T sum_Zj = 0.0;
     for ( size_t j = 0; j < q; j ++ )
       sum_Zj += Z1[ j ];
     sum_Zj /= 2.0;
     std::gamma_distribution<T>  dist_Z1(  5000.0 + q, 1.0 / ( 1.0 + sum_Zj ) );
     Var[ 0 ]  = dist_Z1( generator );

     sum_Zj = 0.0;
     for ( size_t j = 0; j < q; j ++ )
       sum_Zj += Z2[ j ];
     sum_Zj /= 2.0;
     std::gamma_distribution<T>  dist_Z2(  2.0 + q, 1.0 / ( 1.0 + sum_Zj ) );
     Var[ 1 ]  = dist_Z2( generator );

     }

     //printf( "Iter %4lu Var_0 %.3E Var_1 %.3E \n", it, Var[ 0 ], Var[ 1 ] ); fflush( stdout );
     //printf( "Iter %4lu sigma_e %.3E sigma_g %.3E \n", it, sigma_e, sigma_g ); fflush( stdout );
     //printf( "Iter %4lu beta_m0 %.3E beta_m1 %.3E \n", it, beta_m[ 0 ], beta_m[ 1 ] ); fflush( stdout );
     //printf( "Iter %4lu alpha_a0 %.3E alpha_a1 %.3E \n", it, alpha_a[ 0 ], alpha_a[ 1 ] ); fflush( stdout );

      if ( it > burnIn && it % 10 == 0 )
      {

        for ( int i = 0; i < q; i +=1 )
        {
          my_samples( count, 4*i   ) = beta_m[ i ];
          my_samples( count, 4*i+1 ) = alpha_a[ i ];
          my_samples( count, 4*i+2 ) = Z1[ i ];
          my_samples( count, 4*i+3 ) = Z2[ i ];
        }

        my_samples( count, 4* (int)q ) = beta_a[ 0 ];
        my_samples( count, 4* (int)q + 1 ) = Var[ 0 ];
        my_samples( count, 4* (int)q + 2 ) = Var[ 1 ];
        my_samples( count, 4* (int)q + 3 ) = sigma_e;
        my_samples( count, 4* (int)q + 4 ) = sigma_g;
        count += 1;

      if ( count >= 499 )
      {
     	printf( "Iter %4lu \n", count ); fflush( stdout );
        string my_samples_filename = string( "twoZ_" ) + Z_type + to_string( (int)q1 ) + to_string( (int)q2 ) + string( ".txt" );
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

    string Z_type;

    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;

    T ha  = 2.0;

    T la  = 1.0;

    T he  = 2.0;

    T le  = 1.0;

    T hg  = 2.0;

    T lg = 1.0;

    T sigma_a;

    T sigma_g;

    T sigma_e;

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

    hmlp::Data<T> Z1;
    hmlp::Data<T> Z2;

    hmlp::Data<T> var_m1;

    hmlp::Data<T> sigma_m1;

    hmlp::Data<T> var_ma1;

    hmlp::Data<T> sigma_ma1;

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
	         size_t n, size_t w1, size_t w2, size_t q, size_t q1, size_t q2, size_t burnIn, size_t niter, string Z_type )
{
  Variables<T> variables( Y, A, M, C1, C2, beta_m, alpha_a, Var, n, w1, w2, q, q1, q2, Z_type );

  std::srand(std::time(nullptr));

  for ( size_t it = 0; it < niter; it ++ )
  {
    variables.Iteration( burnIn, it );
  }

};


};
};

#endif // ifndef MCMC_HPP
