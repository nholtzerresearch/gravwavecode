/* ---------------------------------------------------------------------
 *
 * This file aims to solve a GR problem utilizing a Reissner-Nordstrom metric
 *
 * --------------------------------------------------------------------- */

#include <deal.II/base/config.h>
#include <deal.II/base/parameter_acceptor.h>

using namespace dealii;


class Coefficients : public ParameterAcceptor
{
public:
  typedef std::complex<double> value_type;
  static constexpr value_type imag{0., 1.};

  Coefficients()
    : ParameterAcceptor("A - Coefficients")
  {
    M = 0.2;
    add_parameter("M", M, "mass of the black hole");

    Q = 0.08;
    add_parameter("Q", Q, "total charge of the black hole");

    q = 1.;
    add_parameter("q", q, "charge density");


    f = [&](double r) {
      return 1. - (2. * M / r) + (Q * Q) / (r * r) + 0. * imag;
    };

    f_prime = [&](double r) {
      return (2. / (r * r)) * (M - ((Q * Q) / r)) + 0. * imag;
    };

    g = [&](double r) {
      return 2. * f_prime(r) + 2. * (f(r) + imag * q * Q) / r;
    };

    a = [&](double r) { return g(r) + 2. / r - f_prime(r); };

    b = [&](double r) { return 2. + f(r); };

    c = [&](double r) { return 2. / r; };

    d = [&](double r) { return imag * q * Q / (r * r); };
  }

  std::function<value_type(double)> a;
  std::function<value_type(double)> b;
  std::function<value_type(double)> c;
  std::function<value_type(double)> d;

private:

  /* Private functions: */

  std::function<value_type(double)> f;
  std::function<value_type(double)> f_prime;
  std::function<value_type(double)> g;

  /* Parameters: */

  double M;
  double Q;
  double q;
};



int main()
{
  Coefficients coefficients;

  ParameterAcceptor::initialize("gravwave.prm");

  return 0;
}


#if 0
#endif
