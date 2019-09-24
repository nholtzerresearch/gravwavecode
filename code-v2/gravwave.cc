/* ---------------------------------------------------------------------
 *
 * This file aims to solve a GR problem utilizing a Reissner-Nordstrom metric
 *
 * --------------------------------------------------------------------- */

#include <deal.II/base/config.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "helper.h"

using namespace dealii;


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */


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
             Mu * std::cos(M_PI / (R_1 - R_0) * (r - (R_1 + R_0) / 2.));
    };
  }

  /* Publicly readable parameters: */

  double R_0;
  double R_1;

  value_type Psi_0;
  value_type Psi_1;

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


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */


template <int dim>
class Discretization : public ParameterAcceptor
{
public:
  static_assert(dim == 1, "Only implemented for 1D");

  Discretization(const Coefficients &coefficients)
      : ParameterAcceptor("B - Discretization")
      , p_coefficients(&coefficients)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&Discretization<dim>::parse_parameters_callback, this));

    refinement = 5;
    add_parameter(
        "refinement", refinement, "refinement of the spatial geometry");

    order_mapping = 1;
    add_parameter("order mapping", order_mapping, "order of the mapping");

    order_finite_element = 1;
    add_parameter("order finite element",
                  order_finite_element,
                  "polynomial order of the finite element space");

    order_quadrature = 3;
    add_parameter(
        "order quadrature", order_quadrature, "order of the quadrature rule");
  }

  void parse_parameters_callback()
  {
    p_triangulation.reset(new Triangulation<dim>);
    auto &triangulation = *p_triangulation;

    GridGenerator::hyper_cube(
        triangulation, p_coefficients->R_0, p_coefficients->R_1);

    triangulation.begin_active()->face(0)->set_boundary_id(0); // FIXME
    triangulation.begin_active()->face(1)->set_boundary_id(0); // FIXME

    triangulation.refine_global(refinement);

    p_mapping.reset(new MappingQ<dim>(order_mapping));

    /* create a system with real part and imaginary part: */
    p_finite_element.reset(
        new FESystem<dim>(FE_Q<dim>(order_finite_element), 2));

    p_quadrature.reset(new QGauss<dim>(order_quadrature));
  }

  const Triangulation<dim> &triangulation() const
  {
    return *p_triangulation;
  }

  const Mapping<dim> &mapping() const
  {
    return *p_mapping;
  }

  const FiniteElement<dim> &finite_element() const
  {
    return *p_finite_element;
  }

  const Quadrature<dim> &quadrature() const
  {
    return *p_quadrature;
  }

private:
  SmartPointer<const Coefficients> p_coefficients;

  std::unique_ptr<Triangulation<dim>> p_triangulation;
  std::unique_ptr<const Mapping<dim>> p_mapping;
  std::unique_ptr<const FiniteElement<dim>> p_finite_element;
  std::unique_ptr<const Quadrature<dim>> p_quadrature;

  unsigned int refinement;

  unsigned int order_finite_element;
  unsigned int order_mapping;
  unsigned int order_quadrature;
};


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */


template <int dim>
class OfflineData : public Subscriptor
{
public:
  OfflineData(const Coefficients &coefficients,
              const Discretization<dim> &discretization)
      : p_coefficients(&coefficients)
      , p_discretization(&discretization)
  {
  }

  void prepare()
  {
    setup();
    assemble();
  }

  void setup();
  void assemble();

  void setup_constraints();

  DoFHandler<dim> dof_handler;
  SparsityPattern sparsity_pattern;

  AffineConstraints<double> affine_constraints;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> stiffness_matrix;

  SmartPointer<const Coefficients> p_coefficients;
  SmartPointer<const Discretization<dim>> p_discretization;
};


template <int dim>
void OfflineData<dim>::setup()
{
  std::cout << "OfflineData<dim>::setup_system()" << std::endl;

  const auto &triangulation = p_discretization->triangulation();
  const auto &finite_element = p_discretization->finite_element();

  dof_handler.initialize(triangulation, finite_element);

  std::cout << "        " << dof_handler.n_dofs() << " DoFs" << std::endl;

  DoFRenumbering::Cuthill_McKee(dof_handler);

  setup_constraints();

  DynamicSparsityPattern c_sparsity(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(
      dof_handler, c_sparsity, affine_constraints, false);

  sparsity_pattern.copy_from(c_sparsity);

  mass_matrix.reinit(sparsity_pattern);
  stiffness_matrix.reinit(sparsity_pattern);
}


template <int dim>
void OfflineData<dim>::setup_constraints()
{
  std::cout << "OfflineData<dim>::setup_constraints()" << std::endl;
  affine_constraints.clear();

  const auto &coefficients = *p_coefficients;

  const auto lambda = [&](const Point<dim> &p, const unsigned int component) {
    Assert(component <= 1, ExcMessage("need exactly two components"));

    if (p[0] <= coefficients.R_0 + 1.e-6) {
      if (component == 0)
        return coefficients.Psi_0.real();
      else
        return coefficients.Psi_0.imag();

    } else if (p[0] >= coefficients.R_1 - 1.e-6) {
      if (component == 0)
        return coefficients.Psi_1.real();
      else
        return coefficients.Psi_1.imag();
    }

    Assert(false, ExcMessage("Not at boundary!"));
    __builtin_trap();
  };

  VectorTools::interpolate_boundary_values(
      dof_handler,
      1,
      to_function<dim, /*components*/ 2>(lambda),
      affine_constraints);

  DoFTools::make_hanging_node_constraints(dof_handler, affine_constraints);

  affine_constraints.close();
}


template <int dim>
void OfflineData<dim>::assemble()
{
  std::cout << "OfflineData<dim>::assemble_system()" << std::endl;

  mass_matrix = 0.;
  stiffness_matrix = 0.;

  const auto &mapping = p_discretization->mapping();
  const auto &finite_element = p_discretization->finite_element();
  const auto &quadrature = p_discretization->quadrature();

  const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  FEValues<dim> fe_values(mapping,
                          finite_element,
                          quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEValuesViews::Scalar<dim> real_part(fe_values, 0);
  FEValuesViews::Scalar<dim> imag_part(fe_values, 1);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const unsigned int n_q_points = quadrature.size();

  for (auto cell : dof_handler.active_cell_iterators()) {

    cell_mass_matrix = 0;
    cell_stiffness_matrix = 0.;

    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    const auto &quadrature_points = fe_values.get_quadrature_points();

    const auto imag = Coefficients::imag;
    const auto rot_x = 1.;
    const auto rot_y = -imag;

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

      const auto position = quadrature_points[q_point];
      const auto r = position[0];

      const auto a = p_coefficients->a(r);
      const auto b = p_coefficients->b(r);
      const auto c = p_coefficients->c(r);
      const auto d = p_coefficients->d(r);

      const auto JxW = fe_values.JxW(q_point);

      // index i for test space, index j for ansatz space
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {

        const auto value_i = rot_x * real_part.value(i, q_point) +
                             rot_y * imag_part.value(i, q_point);

        const auto grad_i = rot_x * real_part.gradient(i, q_point) +
                            rot_y * imag_part.gradient(i, q_point);

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          const auto value_j = rot_x * real_part.value(j, q_point) +
                               rot_y * imag_part.value(j, q_point);

          const auto grad_j = real_part.gradient(j, q_point) +
                              imag * imag_part.gradient(j, q_point);

          const auto mass_term = (c * value_i - 2. * grad_i[0]) * value_j * JxW;
          cell_mass_matrix(i, j) += mass_term.real();

          const auto stiffness_term =
              (a * value_i - b * grad_i[0]) * grad_j[0] * JxW +
              d * value_i * value_j * JxW;
          cell_stiffness_matrix(i, j) += stiffness_term.real();

        } // for j
      }   // for i
    }     // for q_point

    affine_constraints.distribute_local_to_global(
        cell_mass_matrix, local_dof_indices, mass_matrix);

    affine_constraints.distribute_local_to_global(
        cell_stiffness_matrix, local_dof_indices, stiffness_matrix);
  }
}


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

template <int dim>
class TimeStep : public ParameterAcceptor
{
public:
  TimeStep(const OfflineData<dim> &offline_data)
      : ParameterAcceptor("C - TimeStep")
      , p_offline_data(&offline_data)
  {
    kappa = 0.1;
    add_parameter("kappa", kappa, "time step size");

    theta = 0.5;
    add_parameter("theta", theta, "theta paramter of the theta scheme");
  }

  void prepare();

  /* Updates the vector with the solution for the next time step: */
  void step(Vector<double> &solution) const;

  SmartPointer<const OfflineData<dim>> p_offline_data;
  double kappa;
  double theta;

private:

  SparseMatrix<double> L;
  SparseMatrix<double> R;

  SparseDirectUMFPACK L_inverse;
};


template <int dim>
void TimeStep<dim>::prepare()
{
  std::cout << "TimeStep<dim>::prepare()" << std::endl;

  const auto &offline_data = *p_offline_data;

  L.reinit(offline_data.sparsity_pattern);
  R.reinit(offline_data.sparsity_pattern);

  const auto &M = offline_data.mass_matrix;
  const auto &S = offline_data.stiffness_matrix;

  /*
   * We want to solve the equation:
   * M ( U^n+1 - U^n ) + kappa * (1-\theta) S U^n+1 + kappa * theta S U^n = 0
   */

  /* R = M - theta * kappa * S */
  R.copy_from(M);
  R.add(-theta * kappa, S);

  /* L = M + (1. - theta) * kappa * S */
  L.copy_from(M);
  L.add((1. - theta) * kappa, S);

  /* L_inverse = L^-1 */
  L_inverse.initialize(L);
}


template <int dim>
void TimeStep<dim>::step(Vector<double> &solution) const
{
  GrowingVectorMemory<Vector<double>> vector_memory;
  typename VectorMemory<Vector<double>>::Pointer p_tmp(vector_memory);
  auto &tmp = *p_tmp;

  tmp.reinit(solution);

  R.vmult(tmp, /* old */ solution);
  L_inverse.vmult(/* new */ solution, tmp);

  return;
}


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */


template <int dim>
class TimeLoop : public ParameterAcceptor
{
public:
  TimeLoop(const TimeStep<dim> &time_step)
      : ParameterAcceptor("D - TimeLoop")
      , p_time_step(&time_step)
  {
    t_end = 1.0;
    add_parameter("final time", t_end, "final time of the simulation");
  }

  void run();

private:
  SmartPointer<const TimeStep<dim>> p_time_step;

  double t_end;
};


template <int dim>
void TimeLoop<dim>::run()
{
  std::cout << "TimeLoop<dim>::run()" << std::endl;

  const auto &time_step = *p_time_step;
  const auto &offline_data = *time_step.p_offline_data;
  const auto &coefficients = *offline_data.p_coefficients;
  const auto &discretization = *offline_data.p_discretization;

  const auto &mapping = discretization.mapping();
  const auto &dof_handler = offline_data.dof_handler;

  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());

  const double kappa = time_step.kappa;


  unsigned int n = 0;
  double t = 0;
  for (; t <= t_end + 1.0e-10;) {

    std::cout << "time step n = " << n << "\tt = " << t << std::endl;

    if (n == 0) {
      std::cout << "    interpolate initial values" << std::endl;

      /* interpolate initial conditions: */

      const auto &initial_values = coefficients.initial_values;

      const auto lambda = [&](const Point<dim> &p, const unsigned int component) {
        Assert(component <= 1, ExcMessage("need exactly two components"));

        const auto value = initial_values(/* r = */ p[0]);

        if (component == 0)
          return value.real();
        else
          return value.imag();
      };

      VectorTools::interpolate(mapping,
                               dof_handler,
                               to_function<dim, /*components*/ 2>(lambda),
                               solution);
    } else {

      time_step.step(solution);

    }

    {
      /* output: */
      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      data_out.build_patches(/* FIXME: adjust for DEGREE of ansatz */);

      std::string name = std::string("solution-") + std::to_string(n) +
                         std::string(".gnuplot");
      std::ofstream output(name);

      data_out.write_gnuplot(output);
    }

    n += 1;
    t += kappa;

  } /* for */
}


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */


int main()
{
  constexpr int dim = 1;

  Coefficients coefficients;
  Discretization<dim> discretization(coefficients);
  OfflineData<dim> offline_data(coefficients, discretization);
  TimeStep<dim> time_step(offline_data);
  TimeLoop<dim> time_loop(time_step);

  ParameterAcceptor::initialize("gravwave.prm");

  offline_data.prepare();
  time_step.prepare();

  time_loop.run();

  return 0;
}

