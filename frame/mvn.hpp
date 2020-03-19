#include <hmlp_blas_lapack.h>
#include <data.hpp>

using namespace std;

namespace hmlp
{

template<typename T>
class MultiVariableNormal
{
  public:

    MultiVariableNormal( Data<T> mu, Data<T> &user_Sigma )
    {
      this->d = mu.row();
      this->mu = mu;
      this->Sigma = user_Sigma;
      /** Cholesky factorization (POTRF): Sigma = LL' */
      xpotrf( "Lower", d, Sigma.data(), d );
      /** Compute the determinant from the Cholesky factorization */
      for ( uint64_t i = 0; i < d; i ++ ) det *= Sigma( i, i ) * Sigma( i, i );
    };

    T Determinant() { return det; };

    T LogDeterminant() { return std::log( det ); }

    Data<T> Inverse()
    {
      Data<T> inv( d, d, 0 );
      for ( uint64_t i = 0; i < d; i ++ ) inv( i, i ) = 1;
      xtrsm( "Left", "Lower", "No Transpose", "Not Unit", d, d, 
          1.0, Sigma.data(), d, inv.data(), d );
      xtrsm( "Left", "Lower", "Transpose", "Not Unit", d, d, 
          1.0, Sigma.data(), d, inv.data(), d );
      return inv;
    };

    Data<T> SampleFrom( uint64_t num_of_samples )
    {
      Data<T> X( d, num_of_samples );
      /** TODO: Normal( 0, 1 ) */
      X.randn();
      /** Compute L * X using TRMM. */
      xtrmm( "Left", "Lower", "No Transpose", "Not Unit", d, num_of_samples,
          1.0, Sigma.data(), d, X.data(), d );
      /** X = mu + X; */
      for ( uint64_t j = 0; j < num_of_samples; j ++)
      {
        for ( uint64_t i = 0; i < d; i ++ ) X( i, j ) += mu[ i ];
      }

      return X;
    };

  private:

    /** Dimension (number of variables) */
    uint64_t d = 0;

    /** d-by-1 expectation */
    Data<T> mu;

    /** d-by-d variance-covariance matrix */
    Data<T> Sigma;

    T det = 1;

};


}; /** end namespace hmlp */
