#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <complex>


using namespace dealii;
typedef std::complex<double> value_type;
static constexpr value_type  imag{0., 1.};

int main ()
{
static constexpr int dim = 2;
/*int point=1;
std::function<value_type(const Point<dim> &point)> return_foo =
    [](const Point<dim> &point) {
      return 42.;

    };

  std::function<value_type(const Point<dim> &point)> return_bar =
    [&](const Point<dim> &point) {
      //
      return return_foo(point)+42.;
    };

std::cout << "return_foo" <<std::endl;
*/





// PARAMETER
  std::function<value_type(const Point<dim> &point)> f_func=
      [](const Point<dim> & point) {
     double M = 1.0; //need to change before running
     double Q = 1.0;

      return 1.0-(2*M/point[0])+(Q*Q)/(point[0]*point[0]) + 0.0 * imag ;
  };
     std::function<value_type(const Point<dim> &point)> fprime_func=    
       [](const Point<dim> & point) {
       double M = 1.0; //need to change before running
       double Q = 1.0;

      return  (2/(point[0]*point[0]))*(M-((Q*Q)/point[0])) + 0.0 * imag ;
      };
      

     std::function<value_type(const Point<dim> &point)> g_func=

        [&](const Point<dim> & point) {
        //
        double Q = 1.0;
        double q = 1.0;

       return (2.0*f_func(point))/point[0] + 2.0 * imag*(q*Q)/point[0] +fprime_func(point);      
};


return 0;

}
