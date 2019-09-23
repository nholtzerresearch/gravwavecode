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
    R_0 = 0.001;
    add_parameter("R_0", R_0, "inner radius of the computational domain");

    R_1 = 0.0186;
    add_parameter("R_1", R_1, "outer radius of the computational domain");

    Psi_0 = 0.001;
    add_parameter("Psi_0", Psi_0, "inner boundary value");

    Psi_1 = 0.001;
    add_parameter("Psi_1", Psi_1, "outer boundary value");

    M = 0.2;
    add_parameter("M", M, "mass of the black hole");

    Q = 0.08;
    add_parameter("Q", Q, "total charge of the black hole");

    q = 1.;
    add_parameter("q", q, "charge density");

    Mu = 2.;
    add_parameter("Mu", Mu, "mu parameter of the initial envelope");


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

    initial_values = [&](double r) {
      Assert(r >= R_0 && r <= R_1, ExcMessage("Radius r is out of range"));
      return Psi_0 * (r - R_1) / (R_0 - R_1) + Psi_1 * (r - R_0) / (R_1 - R_0) +
             Mu * std::cos(M_PI * (R_1 - R_0) / 2. * (r - (R_1 + R_0) / 2.));
    };
  }

  /* Publicly readable parameters: */

  double R_0;
  double R_1;

  double Psi_0;
  double Psi_1;

  double Mu;

  /* Publicly readable functions: */

  std::function<value_type(double)> a;
  std::function<value_type(double)> b;
  std::function<value_type(double)> c;
  std::function<value_type(double)> d;

  std::function<value_type(double)> initial_values;

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
