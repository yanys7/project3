#include <stdio.h>
#include <gsl_cdf.h>
#include <inverse_gaussian.hpp>

int main (void)
{
    double x = 0.01;
    double y = gsl_cdf_gamma_P (x, 0.5, 1.0);
    printf ("J0(%g) = %.3e/n", y, x);

    using boost::math::inverse_gaussian;
    inverse_gaussian my_ig( 1.0, 5.642E-04);
    printf ("J0(%g) = %.3e/n", cdf( my_ig, 5.0 ), quantile( my_ig, 9.987E-01));
    return 0;
}
