#ifndef DATA_HPP
#define DATA_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <vector>
#include <deque>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>

#include <hmlp_blas_lapack.h>

#define DEBUG_DATA 1


using namespace std;


namespace hmlp
{

template<class T, class Allocator = allocator<T> >
class Data : public vector<T, Allocator>
{
  public:

    Data() : d( 0 ), n( 0 ) {};

    Data( size_t d, size_t n ) : vector<T, Allocator>( d * n )
    { 
      this->d = d;
      this->n = n;
    };

    Data( std::size_t d, std::size_t n, T initT ) : std::vector<T, Allocator>( d * n, initT )
    { 
      this->d = d;
      this->n = n;
    };

    Data( std::size_t d, std::size_t n, std::string &filename ) : std::vector<T, Allocator>( d * n )
    {
      this->d = d;
      this->n = n;

      std::cout << filename << std::endl;

      std::ifstream file( filename.data(), std::ios::in|std::ios::binary|std::ios::ate );
      if ( file.is_open() )
      {
        auto size = file.tellg();
        assert( size == d * n * sizeof(T) );
        file.seekg( 0, std::ios::beg );
        file.read( (char*)this->data(), size );
        file.close();
      }
    };

    enum Pattern : int { STAR = -1 };

    void resize( std::size_t d, std::size_t n )
    { 
      this->d = d;
      this->n = n;
      std::vector<T, Allocator>::resize( d * n );
    };

    void resize( std::size_t d, std::size_t n, T initT )
    {
      this->d = d;
      this->n = n;
      std::vector<T, Allocator>::resize( d * n, initT );
    };

    void reserve( std::size_t d, std::size_t n ) 
    {
      this->reserve( d * n );
    };

    void read( std::size_t d, std::size_t n, std::string &filename )
    {
      assert( this->d == d );
      assert( this->n == n );
      assert( this->size() == d * n );

      std::cout << filename << std::endl;

      std::ifstream file( filename.data(), std::ios::in|std::ios::binary|std::ios::ate );
      if ( file.is_open() )
      {
        auto size = file.tellg();
        assert( size == d * n * sizeof(T) );
        file.seekg( 0, std::ios::beg );
        file.read( (char*)this->data(), size );
        file.close();
      }
    };


    void readSubColumns(std::vector<int32_t> J, std::string infilename, std::string outfilename)
    {
      if (J.size() == 0) 
      {
        return;
      }
      std::cout << "input  filename: " << infilename << std::endl;
      std::cout << "output filename: " << outfilename << std::endl;
      std::ifstream infile(infilename.data()); 
      std::ofstream outfile(outfilename.data());
      std::string line;

      if (infile.is_open())
      {
        int32_t i = 0;
        while (std::getline(infile, line)) 
        {
          std::vector<std::string> tokens;
          std::string token;
          std::istringstream tokenStream(line);
          while (std::getline(tokenStream, token, ',')) 
          {
            tokens.push_back(token);
          }
          if (i % 100000 == 0) 
          {
            printf("row: %8lu ", i); fflush(stdout);
            std::cout << tokens[0] << std::endl;
          }
          outfile << tokens[ 0 ] << "," << tokens[ 1 ] << "," << tokens[ 2 ] << ",";
          for (int32_t j = 0; j < J.size(); j ++)
          {
            outfile << tokens[ J[j] ];
            if (j != J.size() - 1) 
            {
              outfile << ",";
            }
          }
          outfile << "\n";
          /* Increase the number of rows. */
          i ++;
        }
        /* Close the file. */
        infile.close();
        outfile.close();
      }
    }



    template<bool TRANS = false>
    void readmatrix( std::size_t d, std::size_t n, std::string &filename )
    {
      assert( this->d == d );
      assert( this->n == n );
      assert( this->size() == d * n );

      std::cout << filename << std::endl;
      std::ifstream file( filename.data() ); 
      std::string line;
      if ( file.is_open() )
      {
        size_t i = 0;
        while ( std::getline( file, line ) ) 
        {
          if ( i % 1000 == 0 ) printf( "%4lu ", i ); fflush( stdout );
          if ( i >= d ) 
          {
            printf( "more data then execpted %lu\n", d );
          }
          std::istringstream iss( line );
          for ( size_t j = 0; j < n; j ++ )
          {
            T tmp;
            if ( !( iss >> tmp ) )
            {
              printf( "line %lu does not have enough elements (only %lu)\n", i, j );
              break;
            }
            else
            {
              (*this)[ j * d + i ] = tmp;
            } 
          }
          i ++;
        }
        printf( "\nfinish readmatrix %s\n", filename.data() );
      }
    };

    void readdiag( std::size_t d, std::string &filename )
    {
      std::cout << filename << std::endl;
      std::ifstream file( filename.data() ); 
      std::string line;
      if ( file.is_open() )
      {
        size_t j = 0;
        while ( std::getline( file, line ) ) 
        {
          if ( j % 1000 == 0 ) printf( "%4lu ", j );
          std::istringstream iss( line );
          for ( size_t i = 0; i < d; i ++ )
          {
            T tmp;
            if ( !( iss >> tmp ) )
            {
              printf( "line %lu does not have enough elements (only %lu)\n", j, i );
              break;
            }
            else
            {
              if ( i == j ) (*this)[ i ] = tmp;
            }
          }
          j ++;
        }
        printf( "\nfinish readmatrix %s\n", filename.data() );
      }
    };


    void WriteFile( const char *name )
    {
      FILE * pFile;
      pFile = fopen ( name, "w" );
      //fprintf( pFile, "%s=[\n", name );
      for ( size_t i = 0; i < d; i ++ )
      {
        for ( size_t j = 0; j < n; j ++ )
        {
          //fprintf( pFile, "%lf,", (*this)( i, j ) );
          fprintf( pFile, "%lf ", (*this)( i, j ) );
        }
        //fprintf( pFile, ";\n" );
        fprintf( pFile, "\n" );
      }
      //fprintf( pFile, "];\n" );
      fclose( pFile );
    };


    std::tuple<size_t, size_t> shape()
    {
      return std::make_tuple( d, n );
    };

    template<typename TINDEX>
    inline T & operator()( TINDEX i, TINDEX j )
    {
      return (*this)[ d * j + i ];
    };


    template<typename TINDEX>
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &imap, std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( imap.size(), jmap.size() );

      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < imap.size(); i ++ )
        {
          submatrix[ j * imap.size() + i ] = (*this)[ d * jmap[ j ] + imap[ i ] ];
        }
      }

      return submatrix;
    }; 

    template<typename TINDEX>
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( d, jmap.size() );

      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < d; i ++ )
        {
          submatrix[ j * d + i ] = (*this)[ d * jmap[ j ] + i ];
        }
      }

      return submatrix;
    }; 

    //std::vector<T> operator()( std::vector<std::size_t> &jmap )
    //{
    //  std::vector<T> submatrix( d * jmap.size() );

    //  for ( int j = 0; j < jmap.size(); j ++ )
    //  {
    //    for ( int i = 0; i < d; i ++ )
    //    {
    //      submatrix[ j * d + i ] = (*this)[ d * jmap[ j ] + i ];
    //    }
    //  }

    //  return submatrix;
    //}; 



    template<bool SYMMETRIC = false>
    void rand( T a, T b )
    {
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator( seed );
      std::uniform_real_distribution<T> distribution( a, b );

      if ( SYMMETRIC ) assert( n == d );

      for ( std::size_t j = 0; j < n; j ++ )
      {
        for ( std::size_t i = 0; i < d; i ++ )
        {
          if ( SYMMETRIC )
          {
            if ( i > j )
              (*this)[ j * d + i ] = distribution( generator );
            else
              (*this)[ j * d + i ] = (*this)[ i * d + j ];
          }
          else
          {
            (*this)[ j * d + i ] = distribution( generator );
          }
        }
      }
    };

    template<bool SYMMETRIC = false>
    void rand()
    {
      rand<SYMMETRIC>( 0.0, 1.0 );
    };

    void randn( T mu, T sd )
    {
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator( seed );
      std::normal_distribution<T> distribution( mu, sd );
      for ( std::size_t i = 0; i < d * n; i ++ )
      {
        (*this)[ i ] =  distribution( generator );
      }
    };

    void randn()
    {
      randn( 0.0, 1.0 );
    };

    template<bool USE_LOWRANK>
    void randspd( T a, T b )
    {
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator( seed );
      std::uniform_real_distribution<T> distribution( a, b );

      assert( n == d );

      if ( USE_LOWRANK )
      {
        hmlp::Data<T> X( ( std::rand() % n ) / 2 + 1, n );
        X.rand();
        xgemm
        (
          "T", "N",
          n, n, X.dim(),
          1.0, X.data(), X.dim(),
               X.data(), X.dim(),
          0.0, this->data(), this->dim()
        );
      }
      else // diagonal dominating
      {
        for ( std::size_t j = 0; j < n; j ++ )
        {
          for ( std::size_t i = 0; i < d; i ++ )
          {
            if ( i > j )
              (*this)[ j * d + i ] = distribution( generator );
            else
              (*this)[ j * d + i ] = (*this)[ i * d + j ];

            // Make sure diagonal dominated
            (*this)[ j * d + j ] += std::abs( (*this)[ j * d + i ] );
          }
        }
      }
    };

    template<bool USE_LOWRANK>
    void randspd()
    {
      randspd<USE_LOWRANK>( 0.0, 1.0 );
    };


    size_t dim() { return d; };
    size_t num() { return n; };

    size_t row() { return d; };
    size_t col() { return n; };

    void Print()
    {
      printf( "Data in %lu * %lu\n", d, n );
      for ( int j = 0; j < n; j ++ )
      {
        if ( j % 5 == 0 || j == 0 || j == n - 1 ) printf( "col[%4d] ", j );
        else printf( "          " );
      }
      printf( "\n===========================================================\n" );

      printf( "A = [\n" );
      for ( int i = 0; i < d; i ++ )
      {
        for ( int j = 0; j < n; j ++ ) printf( "% .2E ", (double) (*this)[ j * d + i ] );
        printf(";\n");
      }
      printf("];\n");
    };

    Data<T> operator * ( Data<T> &B )
    {
      //assert( this->col() == B.row() );
      size_t m = this->row();
      size_t k = this->col();
      size_t n = B.col();
      Data<T> C( m, n );
      xgemm( "No Transpose", "No Transpose", m, n, k, 
          1.0, this->data(), m,
               B.data(), k, 
          0.0, C.data(), m );

      return C;
    };

  private:

    size_t d = 0;

    size_t n = 0;

};


}; /** end namespace hmlp */

#endif //define DATA_HPP
