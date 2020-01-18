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
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/solver_gmres.h>
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
    R_0 = 2.;
    add_parameter("R_0", R_0, "inner radius of the computational domain");

    R_1 = 200.;
    add_parameter("R_1", R_1, "outer radius of the computational domain");

    Psi_0 = 0.001;
    add_parameter("Psi_0", Psi_0, "inner boundary value");

    Psi_1 = 0.001;
    add_parameter("Psi_1", Psi_1, "outer boundary value");

    M = 0.5;
    add_parameter("M", M, "mass of the black hole");

    Q = 0.0;
    add_parameter("Q", Q, "total charge of the black hole");

    q = 1.;
    add_parameter("q", q, "charge density");

    Mu = 2.;
    add_parameter("Mu", Mu, "mu parameter of the initial envelope");

    coef1 = 1.;
    add_parameter("coef1", coef1, "coefficient on r in the manufactured sol.");

    coef2 = 1.;
    add_parameter("coef2", coef2, "coefficient on t in the manufactured sol.");

    a = [&](double r) {return (2.* M * pow(r,3.)-Q*Q*pow(r,2.)+2.*M*r-pow(Q,2.))/(-r*r - 2.*M*r + Q*Q);};
    b = [&](double r) { return (pow(r,4.)-2*M*pow(r,3.)-Q*Q*pow(r,2.))/(-r*r-2.*M*r+Q*Q); }; 
    c = [&](double r) { return (2.*M*pow(r,2.)-2.*imag*q*Q*pow(r,3.))/(-r*r-2.*M*r+Q*Q); };
    d = [&](double r) { return (2.*r*r-2.*M*r)/(-r*r-2.*M*r+Q*Q); };
    e = [&](double r) {return (r*r *q*q*Q*Q)/(-r*r-2.*M*r+Q*Q);};
    aprime = [&](double r) { return -2.*r*(pow(Q,4.)+pow(Q,2.)*(1.-4.*M*r)+M*r*(-1. + 4.*M*r+ r*r))/((Q*Q-r*(2.*M+r))*(Q*Q-r*(2.*M+r))); };

    bprime = [&](double r) { return -2.*r*(pow(Q,4.)+2.*pow(Q,2.)*(M-r)*r+pow(r,2.)*(-4.*pow(M,2.)+2.*M*r+pow(r,2.)))/((Q*Q-r*(2.*M+r))*(Q*Q-r*(2.*M+r))); };
    
    dprime = [&](double r) { return 4.*pow(Q,2.)*r-2.*M*(pow(Q,2.)+3.*pow(r,2.)) /((Q*Q-r*(2.*M+r))*(Q*Q-r*(2.*M+r)));};

    initial_values_u = [&](double r) {
        return 0.;
	//return 10. * std::exp(-(r-10.1)*(r-10.1));
	//return 10. * std::exp(-(r-100.)*(r - 100.)/2) ;
    };
    initial_values_v = [&](double r) {

	//return 10. * std::exp(-(r-10.)*(r-10.));
	return 0.;
    };
    /*Boundary values resulting from manufactured solution*/
    boundary_values = [&](double r, double t) {
       //return 0.;
       if (r == R_0)
	 return 0.;
	 //return 10. * std::exp(-(R_0-10.)*(R_0-10.));
	 //return 10. * std::exp(-(R_0-10.)*(R_0 - 10.)/3.);
      // if (r == R_1)
	// return 0.;
        // else
	 //return 10. * std::exp((R_1-10.)*((R_1-10.)/2.)) ;//works for diffusion
    };

    //RHS resulting from ansatz solution: Psi=sin(ar - bt)
    manufactured_solution_rhs = [&](double r, double t) {
	return 0.;
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
  std::function<value_type(double)> e;
  std::function<value_type(double)> aprime;
  std::function<value_type(double)> bprime;
  std::function<value_type(double)> dprime;
  std::function<value_type(double)> initial_values_u;
  std::function<value_type(double)> initial_values_v;

  std::function<value_type(double, double)> boundary_values;

  std::function<value_type(double, double)> manufactured_solution_rhs;

private:
  /* Parameters: */

  double M;
  double Q;
  double q;
  double coef1;
  double coef2;
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

    triangulation.begin_active()->face(0)->set_boundary_id(1); // FIXME
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
/*----------------------------------------------------------------------------*//*----------------------------------------------------------------------------*/

template <int dim>
class ManufacturedSolution : public Subscriptor
{
public:
  ManufacturedSolution(const Coefficients &coefficients,
              const Discretization<dim> &discretization)
      : p_coefficients(&coefficients)
      , p_discretization(&discretization)
     
  {
  }

  void set_time(double new_t) 
  {
    t=new_t;
  }

  Coefficients::value_type
  get_manufactured_source(double r, double t);
  

  void prepare()
  {
    setup();
    assemble();
  }

  void setup();
  void assemble();

  DoFHandler<dim> dof_handler;
  SparsityPattern sparsity_pattern;

  AffineConstraints<double> affine_constraints;

  Vector<double> right_hand_side;


  SmartPointer<const Coefficients> p_coefficients;
  SmartPointer<const Discretization<dim>> p_discretization;
private:
  double t; 
};


template <int dim>
Coefficients::value_type 
ManufacturedSolution<dim>::get_manufactured_source(double r, double t) 
{
   Coefficients coefficients;

   auto &manufactured_solution_rhs = coefficients.manufactured_solution_rhs;
   auto value = manufactured_solution_rhs(/* r = */ r, /* t = */ t);
   return value;
}

template <int dim>
void ManufacturedSolution<dim>::setup()
{
  std::cout << "ManufacturedSolution<dim>::setup_right_hand_side()" << std::endl;

  const auto &triangulation = p_discretization->triangulation();
  const auto &finite_element = p_discretization->finite_element();

  dof_handler.initialize(triangulation, finite_element);
  DoFRenumbering::Cuthill_McKee(dof_handler);
 // setup_constraints();

  DynamicSparsityPattern c_sparsity(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(
      dof_handler, c_sparsity, affine_constraints, true);

  sparsity_pattern.copy_from(c_sparsity);

  right_hand_side.reinit(dof_handler.n_dofs());
}

template <int dim>
void ManufacturedSolution<dim>::assemble()
{
  std::cout << "ManufacturedSolution<dim>::assemble_right_hand_side()" << std::endl;

  right_hand_side = 0.;

  const auto &mapping = p_discretization->mapping();
  const auto &finite_element = p_discretization->finite_element();
  const auto &quadrature = p_discretization->quadrature();
  

  const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

  Vector<double> cell_right_hand_side(dofs_per_cell);

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

    cell_right_hand_side = 0;

    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    const auto &quadrature_points = fe_values.get_quadrature_points();

    const auto imag = Coefficients::imag;
    const auto rot_x = 1.;
    const auto rot_y = -imag;

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

      const auto position = quadrature_points[q_point];
      const auto r = position[0];

      const auto JxW = fe_values.JxW(q_point);
      const auto source = get_manufactured_source(r, t); 


      // index i for test space, index j for ansatz space
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {

        const auto value_i = rot_x * real_part.value(i, q_point) +
                             rot_y * imag_part.value(i, q_point);

        const auto rhs_vec =
              source * value_i * JxW;
          cell_right_hand_side(i) += rhs_vec.real();

      }   // for i
    }     // for q_point


    affine_constraints.distribute_local_to_global(
        cell_right_hand_side, local_dof_indices, right_hand_side);

   /* mass_matrix_unconstrained.add(local_dof_indices, cell_mass_matrix);

    affine_constraints.distribute_local_to_global(
        cell_stiffness_matrix, local_dof_indices, stiffness_matrix);

    stiffness_matrix_unconstrained.add(local_dof_indices,
                                       cell_stiffness_matrix);*/
  } // cell

    std::cout<<"rhs:( "<<right_hand_side(0)<<")"<<std::endl;
}






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

  void set_time(double new_t) 
  {
    t=new_t;
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

  SparseMatrix<double> nodal_mass_matrix;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> mass_matrix_unconstrained;

  SparseMatrix<double> mass_matrix_newv;
  SparseMatrix<double> mass_matrix_oldv;
  SparseMatrix<double> mass_matrix_newu;
  SparseMatrix<double> mass_matrix_oldu;
  SparseMatrix<double> mass_matrix_newv_unconstrained;
  SparseMatrix<double> mass_matrix_oldv_unconstrained;
  SparseMatrix<double> mass_matrix_oldu_unconstrained;
  SparseMatrix<double> mass_matrix_newu_unconstrained;
  SparseMatrix<double> q1_matrix_newv;
  SparseMatrix<double> q1_matrix_oldv;
  SparseMatrix<double> q1_matrix_newu;
  SparseMatrix<double> q2_matrix_newu; 
  SparseMatrix<double> q2_matrix_oldu; 
  SparseMatrix<double> q1_matrix_newv_unconstrained;
  SparseMatrix<double> q1_matrix_oldv_unconstrained;
  SparseMatrix<double> q1_matrix_newu_unconstrained;
  SparseMatrix<double> q2_matrix_newu_unconstrained;
  SparseMatrix<double> q2_matrix_oldu_unconstrained;
  SparseMatrix<double> stiffness_matrix_oldu;
  SparseMatrix<double> stiffness_matrix_newu;
  SparseMatrix<double> stiffness_matrix_oldu_unconstrained;
  SparseMatrix<double> stiffness_matrix_newu_unconstrained;

  SmartPointer<const Coefficients> p_coefficients;
  SmartPointer<const Discretization<dim>> p_discretization;
private:
  double t;
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
      dof_handler, c_sparsity, affine_constraints, true);

  sparsity_pattern.copy_from(c_sparsity);

  nodal_mass_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);
  mass_matrix_unconstrained.reinit(sparsity_pattern);
  mass_matrix_newv.reinit(sparsity_pattern);
  mass_matrix_newv_unconstrained.reinit(sparsity_pattern);
  mass_matrix_oldv.reinit(sparsity_pattern);
  mass_matrix_oldv_unconstrained.reinit(sparsity_pattern);
  mass_matrix_newu.reinit(sparsity_pattern);
  mass_matrix_newu_unconstrained.reinit(sparsity_pattern);
  mass_matrix_oldu.reinit(sparsity_pattern);
  mass_matrix_oldu_unconstrained.reinit(sparsity_pattern);
  q1_matrix_newv.reinit(sparsity_pattern);
  q1_matrix_newv_unconstrained.reinit(sparsity_pattern);
  q1_matrix_oldv.reinit(sparsity_pattern);
  q1_matrix_oldv_unconstrained.reinit(sparsity_pattern);
  q1_matrix_newu.reinit(sparsity_pattern);
  q1_matrix_newu_unconstrained.reinit(sparsity_pattern);
  q2_matrix_newu.reinit(sparsity_pattern);
  q2_matrix_newu_unconstrained.reinit(sparsity_pattern);
  q2_matrix_oldu.reinit(sparsity_pattern);
  q2_matrix_oldu_unconstrained.reinit(sparsity_pattern);
  stiffness_matrix_newu.reinit(sparsity_pattern);
  stiffness_matrix_newu_unconstrained.reinit(sparsity_pattern);
  stiffness_matrix_oldu.reinit(sparsity_pattern);
  stiffness_matrix_oldu_unconstrained.reinit(sparsity_pattern);
}


template <int dim>
void OfflineData<dim>::setup_constraints()
{

  std::cout << "OfflineData<dim>::setup_constraints()" << std::endl;
  affine_constraints.clear();

  const auto &mapping = p_discretization->mapping();
  Coefficients coefficients;
  const auto &boundary_values = coefficients.boundary_values;
  std::map<types::global_dof_index, double> boundary_value_map;

  const auto lambda = [&](const Point<dim> &p, const unsigned int component) {
    Assert(component <= 1, ExcMessage("need exactly two components"));

    const auto value = boundary_values(/* r = */ p[0], /* t = */ t);

    if (component == 0)
      return value.real();
    else
      return value.imag();
  };

  const auto boundary_value_function =
      to_function<dim, /*components*/ 2>(lambda);

  VectorTools::interpolate_boundary_values(
      mapping,
      dof_handler,
      {{0, &boundary_value_function}},
      boundary_value_map);
  DoFTools::make_hanging_node_constraints(dof_handler, affine_constraints);

  affine_constraints.close();
}

template <int dim>
void OfflineData<dim>::assemble()
{
  std::cout << "OfflineData<dim>::assemble_system()" << std::endl;

  nodal_mass_matrix = 0.;
  mass_matrix = 0.;
  mass_matrix_unconstrained = 0.;

  mass_matrix_newv = 0.;
  mass_matrix_oldv = 0.;
  mass_matrix_newu = 0.;
  mass_matrix_oldu = 0.;
  mass_matrix_newv_unconstrained = 0.;
  mass_matrix_oldv_unconstrained = 0.;
  mass_matrix_newu_unconstrained = 0.;
  mass_matrix_oldu_unconstrained = 0.;

  q1_matrix_newv = 0.;
  q1_matrix_oldv = 0.;
  q1_matrix_newu = 0.;
  q1_matrix_newv_unconstrained = 0.;
  q1_matrix_oldv_unconstrained = 0.;
  q1_matrix_newu_unconstrained = 0.;
  q2_matrix_newu = 0.;
  q2_matrix_oldu = 0.;
  q2_matrix_newu_unconstrained = 0.;
  q2_matrix_oldu_unconstrained = 0.;
  stiffness_matrix_newu = 0.;
  stiffness_matrix_oldu = 0.;
  stiffness_matrix_newu_unconstrained = 0.;
  stiffness_matrix_oldu_unconstrained = 0.;

  const auto &mapping = p_discretization->mapping();
  const auto &finite_element = p_discretization->finite_element();
  const auto &quadrature = p_discretization->quadrature();
  

  const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

  FullMatrix<double> cell_nodal_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix_newv(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix_oldv(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix_newu(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix_oldu(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_q1_matrix_newv(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_q1_matrix_oldv(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_q1_matrix_newu(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_q2_matrix_oldu(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_q2_matrix_newu(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix_oldu(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix_newu(dofs_per_cell, dofs_per_cell);

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

    cell_nodal_mass_matrix = 0;
    cell_mass_matrix = 0;
    cell_mass_matrix_newv = 0;
    cell_mass_matrix_oldv = 0;
    cell_mass_matrix_newu = 0;
    cell_mass_matrix_oldu = 0;
    cell_q1_matrix_newv = 0;
    cell_q1_matrix_oldv = 0;
    cell_q1_matrix_newu = 0;
    cell_q2_matrix_oldu = 0;
    cell_q2_matrix_newu = 0;
    cell_stiffness_matrix_oldu = 0.;
    cell_stiffness_matrix_newu = 0.;

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
      const auto e = p_coefficients->e(r);
      const auto aprime = p_coefficients->aprime(r);
      const auto bprime = p_coefficients->bprime(r);
      const auto dprime = p_coefficients->dprime(r);
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

          const auto nodal_mass_term = value_i * value_j * JxW;
          cell_nodal_mass_matrix(i, j) += nodal_mass_term.real();

	  const auto mass_term_newv = ((c - aprime)) * value_i * value_j * JxW;   
          cell_mass_matrix_newv(i, j) += mass_term_newv.real();

	  const auto mass_term_oldv = (c + aprime) * value_i * value_j * JxW;   
          cell_mass_matrix_oldv(i, j) += mass_term_oldv.real();

	  const auto mass_term_newu = (e - dprime) * value_i * value_j * JxW;   
          cell_mass_matrix_newu(i, j) += mass_term_newu.real();

          cell_mass_matrix_oldu(i, j) += mass_term_newu.real();

	  //const auto nodal_q1_mass_term = value_i * grad_j[0] * JxW;
	  //cell_nodal_q1_mass_matrix(i,j) += nodal_q1_mass_term.real(); //DO I NEED THIS FOR OTHER MATRICES BESIDES MASS
	  //const auto nodal_q2_mass_term = grad_i[0] * value_j * JxW;
	  //cell_nodal_q2_mass_matrix(i,j) += nodal_q2_mass_term.real();
	  //
	  const auto q1_term_newv = a * value_i * grad_j[0] * JxW;
	  cell_q1_matrix_newv(i,j) += q1_term_newv.real();
	  cell_q1_matrix_oldv(i,j) += q1_term_newv.real();

	  const auto q1_term_newu = d * value_i * grad_j[0] * JxW;
	  cell_q1_matrix_newu(i,j) += q1_term_newu.real();
	  
	  const auto q2_term_newu = dprime * grad_i[0] * value_j * JxW;
	  cell_q2_matrix_newu(i,j) += q2_term_newu.real();

	  
	  const auto q2_term_oldu = (d + bprime) * grad_i[0] * value_j * JxW;
	  cell_q2_matrix_oldu(i,j) += q2_term_oldu.real();


          const auto stiffness_term_newu = d * grad_i[0]* grad_j[0] * JxW;
	  cell_stiffness_matrix_newu(i,j) += stiffness_term_newu.real();

          const auto stiffness_term_oldu = b * grad_i[0]* grad_j[0] * JxW;
	  cell_stiffness_matrix_oldu(i,j) += stiffness_term_oldu.real();
        } // for j
      }   // for i
    }     // for q_point

    affine_constraints.distribute_local_to_global(
        cell_nodal_mass_matrix, local_dof_indices, nodal_mass_matrix);

    affine_constraints.distribute_local_to_global(
        cell_mass_matrix, local_dof_indices, mass_matrix);

    mass_matrix_unconstrained.add(local_dof_indices, cell_mass_matrix);
    

    affine_constraints.distribute_local_to_global(
		    cell_mass_matrix_newv, local_dof_indices, mass_matrix_newv);
    mass_matrix_newv_unconstrained.add(local_dof_indices, cell_q1_matrix_newv);

    affine_constraints.distribute_local_to_global(
		    cell_mass_matrix_oldv, local_dof_indices, mass_matrix_oldv);
    mass_matrix_newv_unconstrained.add(local_dof_indices, cell_mass_matrix_oldv);

    affine_constraints.distribute_local_to_global(
		    cell_mass_matrix_newv, local_dof_indices, mass_matrix_newu);
    mass_matrix_newv_unconstrained.add(local_dof_indices, cell_mass_matrix_newu);

    affine_constraints.distribute_local_to_global(
		    cell_mass_matrix_oldu, local_dof_indices, mass_matrix_oldu);
    mass_matrix_oldu_unconstrained.add(local_dof_indices, cell_mass_matrix_oldu);

    affine_constraints.distribute_local_to_global(
		    cell_q1_matrix_newv, local_dof_indices, q1_matrix_newv);
    q1_matrix_newv_unconstrained.add(local_dof_indices, cell_q1_matrix_newv);

    affine_constraints.distribute_local_to_global(
		    cell_q1_matrix_oldv, local_dof_indices, q1_matrix_oldv);
    q1_matrix_oldv_unconstrained.add(local_dof_indices, cell_q1_matrix_oldv);
    
    affine_constraints.distribute_local_to_global(
		    cell_q1_matrix_newu, local_dof_indices, q1_matrix_newu);
    q1_matrix_newu_unconstrained.add(local_dof_indices, cell_q1_matrix_newu);

    affine_constraints.distribute_local_to_global(
		    cell_q2_matrix_newu, local_dof_indices, q2_matrix_newu);
    q2_matrix_newu_unconstrained.add(local_dof_indices, cell_q2_matrix_newu);

    affine_constraints.distribute_local_to_global(
		    cell_q2_matrix_oldu, local_dof_indices, q2_matrix_oldu);
    q2_matrix_oldu_unconstrained.add(local_dof_indices, cell_q2_matrix_oldu);

    affine_constraints.distribute_local_to_global(
        cell_stiffness_matrix_newu, local_dof_indices, stiffness_matrix_newu);
    stiffness_matrix_newu_unconstrained.add(local_dof_indices,
                                       cell_stiffness_matrix_newu);

    affine_constraints.distribute_local_to_global(
        cell_stiffness_matrix_oldu, local_dof_indices, stiffness_matrix_oldu);
    stiffness_matrix_oldu_unconstrained.add(local_dof_indices,
                                       cell_stiffness_matrix_oldu);
  } // cell
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

template <int dim>
class TimeStep : public ParameterAcceptor
{
public:
  TimeStep(const OfflineData<dim> &offline_data, ManufacturedSolution<dim> &manufactured)
      : ParameterAcceptor("C - TimeStep")
      , p_offline_data(&offline_data)
      , p_manufactured(&manufactured)
  {
    kappa = 0.1;
    add_parameter("kappa", kappa, "time step size");

    theta = 0.5;
    add_parameter("theta", theta, "theta paramter of the theta scheme");

    linear_solver_limit = 1000.;
    add_parameter("linear_solver_limit", linear_solver_limit, "max number of iterations in newton scheme for linear (left) side");

    linear_solver_tol = 1.0e-12;
    add_parameter("linear_solver_tol", linear_solver_tol, "tolerance to determine if newton step should keep running on left");

    nonlinear_solver_limit = 100.;
    add_parameter("nonlinear_solver_limit", nonlinear_solver_limit, "max number of iterations in newton scheme for nonlinear (right) side");

    nonlinear_solver_tol = 1.0e-12;
    add_parameter("nonlinear_solver_tol", nonlinear_solver_tol, "tolerance to determine if newton step should keep running on right");
  }

  void prepare();

  /* Updates the vector with the solution for the next time step: */
  void step(Vector<double> &old_solution_u,Vector<double> &old_solution_v, double new_t) const;

  SmartPointer<const OfflineData<dim>> p_offline_data;
  SmartPointer<ManufacturedSolution<dim>> p_manufactured;
  double kappa;
  double theta;

  unsigned int linear_solver_limit;
  double linear_solver_tol;
  unsigned int nonlinear_solver_limit;
  double nonlinear_solver_tol;

private:
  SparseMatrix<double> linear_part_u;
  SparseDirectUMFPACK linear_part_u_inverse;

  SparseMatrix<double> linear_part_v;
  SparseDirectUMFPACK linear_part_v_inverse;
};


template <int dim>
void TimeStep<dim>::prepare()
{
  std::cout << "TimeStep<dim>::prepare()" << std::endl;
  const auto &offline_data = *p_offline_data;
  //const auto &manufactured = *p_manufactured;

  linear_part_u.reinit(offline_data.sparsity_pattern);
  linear_part_v.reinit(offline_data.sparsity_pattern);
  const auto &M_c = offline_data.mass_matrix;
  const auto &M_newv_c = offline_data.mass_matrix_newv;
  const auto &M_oldv_c = offline_data.mass_matrix_oldv;
  const auto &M_newu_c = offline_data.mass_matrix_newu;
  const auto &M_oldu_c = offline_data.mass_matrix_oldu;
  const auto &Q1_newv_c = offline_data.q1_matrix_newv;
  const auto &Q1_oldv_c = offline_data.q1_matrix_oldv;
  const auto &Q1_newu_c = offline_data.q1_matrix_newu;
  const auto &Q2_newu_c = offline_data.q2_matrix_newu;
  const auto &Q2_oldu_c = offline_data.q2_matrix_oldu;
  const auto &S_newu_c = offline_data.stiffness_matrix_newu;
  const auto &S_oldu_c = offline_data.stiffness_matrix_oldu;
  /* linear_part_u = M_c  ASK ABOUT WHETHER THIS SHOULD INCLUDE V^n AS WELL*/
  linear_part_u.copy_from(M_c);
  //linear_part_u.add(kappa, Q_c);
  //linear_part_u.add((1. - theta) * kappa * kappa, S_c);

  linear_part_u_inverse.initialize(linear_part_u);

  /* linear_part_v = M + kappa * theta *M_newv_c - kappa * theta * Q1_newv_c */
  linear_part_v.copy_from(M_c);
  linear_part_v.add(theta * kappa, M_newv_c);
  linear_part_v.add(theta *-kappa, Q1_newv_c);

  linear_part_v_inverse.initialize(linear_part_v);
}


template<int dim>
void apply_boundary_values(const OfflineData<dim> &offline_data,
                           const Coefficients &coefficients,
                           double t,
                           Vector<double> &vector)
{
  const auto &mapping = offline_data.p_discretization->mapping();
  const auto &dof_handler = offline_data.dof_handler;

  const auto &boundary_values = coefficients.boundary_values;

  std::map<types::global_dof_index, double> boundary_value_map;

  const auto lambda = [&](const Point<dim> &p, const unsigned int component) {
    Assert(component <= 1, ExcMessage("need exactly two components"));

    const auto value = boundary_values(/* r = */ p[0], /* t = */ t);

    if (component == 0)
      return value.real();
    else
      return value.imag();
  };

  const auto boundary_value_function =
      to_function<dim, /*components*/ 2>(lambda);

  VectorTools::interpolate_boundary_values(
      mapping,
      dof_handler,
      {{0, &boundary_value_function}},
      boundary_value_map);

  for (auto it : boundary_value_map) {
    vector[it.first] = it.second;
  }
}


template <int dim>
void TimeStep<dim>::step(Vector<double> &old_solution_u,Vector<double> &old_solution_v, double new_t) const
{
  const auto &offline_data = *p_offline_data;
  const auto &coefficients = *offline_data.p_coefficients;
  const auto &affine_constraints = offline_data.affine_constraints;
  auto &manufactured = *p_manufactured;

  const auto M_c = linear_operator(offline_data.mass_matrix);
  const auto M_newv_c = linear_operator(offline_data.mass_matrix_newv);
  const auto M_oldv_c = linear_operator(offline_data.mass_matrix_oldv);
  const auto M_newu_c = linear_operator(offline_data.mass_matrix_newu);
  const auto M_oldu_c = linear_operator(offline_data.mass_matrix_oldu);
  const auto Q1_newv_c = linear_operator(offline_data.q1_matrix_newv);
  const auto Q1_oldv_c = linear_operator(offline_data.q1_matrix_oldv);
  const auto Q1_newu_c = linear_operator(offline_data.q1_matrix_newu);
  const auto Q2_newu_c = linear_operator(offline_data.q2_matrix_newu);
  const auto Q2_oldu_c = linear_operator(offline_data.q2_matrix_oldu);
  const auto S_newu_c = linear_operator(offline_data.stiffness_matrix_newu);
  const auto S_oldu_c = linear_operator(offline_data.stiffness_matrix_oldu);

  const auto M_u = linear_operator(offline_data.mass_matrix_unconstrained);
  const auto M_newv_u = linear_operator(offline_data.mass_matrix_newv_unconstrained);
  const auto M_oldv_u = linear_operator(offline_data.mass_matrix_oldv_unconstrained);
  const auto M_newu_u = linear_operator(offline_data.mass_matrix_newu_unconstrained);
  const auto M_oldu_u = linear_operator(offline_data.mass_matrix_oldu_unconstrained);
  const auto Q1_newv_u = linear_operator(offline_data.q1_matrix_newv_unconstrained);
  const auto Q1_oldv_u = linear_operator(offline_data.q1_matrix_oldv_unconstrained);
  const auto Q1_newu_u = linear_operator(offline_data.q1_matrix_newu_unconstrained);
  const auto Q2_newu_u = linear_operator(offline_data.q2_matrix_newu_unconstrained);
  const auto Q2_oldu_u = linear_operator(offline_data.q2_matrix_oldu_unconstrained);
  const auto S_newu_u = linear_operator(offline_data.stiffness_matrix_newu_unconstrained);
  const auto S_oldu_u = linear_operator(offline_data.stiffness_matrix_oldu_unconstrained);

  GrowingVectorMemory<Vector<double>> vector_memory_u;
  GrowingVectorMemory<Vector<double>> vector_memory_v;
  typename VectorMemory<Vector<double>>::Pointer p_new_solution_u(vector_memory_u);

  typename VectorMemory<Vector<double>>::Pointer p_new_solution_v(vector_memory_v);

  auto &new_solution_u = *p_new_solution_u;
  auto &new_solution_v = *p_new_solution_v;

  new_solution_u = old_solution_u;
  new_solution_v = old_solution_v;

  apply_boundary_values(offline_data, coefficients, new_t, new_solution_u);
  apply_boundary_values(offline_data, coefficients, new_t, new_solution_v);
  
  manufactured.set_time(new_t);
  manufactured.prepare();

  for (unsigned int m = 0; m < nonlinear_solver_limit; ++m) {
    Vector<double> residual_u = M_u * (new_solution_u - old_solution_u) - kappa * theta * M_u * new_solution_v - kappa * (1. - theta) * M_u * old_solution_v;

    Vector<double> residual_v = (M_u + kappa * theta * M_newv_u - kappa * theta * Q1_newv_u) * new_solution_v - kappa * (1-theta) * (M_oldv_u + Q1_oldv_u) * old_solution_v + kappa * theta * (M_newu_u - Q1_newu_u - Q2_newu_u - S_newu_u) * new_solution_u + kappa * (1-theta) * (M_oldu_u - Q2_oldu_u - S_oldu_u) * old_solution_u;

    affine_constraints.set_zero(residual_v);
    affine_constraints.set_zero(residual_v);

    if ((residual_u.linfty_norm() < nonlinear_solver_tol)||(residual_v.linfty_norm() < nonlinear_solver_tol))
      break;

    const auto system_matrix_u = linear_operator(linear_part_u);
    const auto system_matrix_v = linear_operator(linear_part_v);

    SolverControl solver_control(linear_solver_limit, linear_solver_tol);
    SolverGMRES<> solver(solver_control);

    const auto system_matrix_u_inverse =
        inverse_operator(system_matrix_u, solver, linear_part_u_inverse);

    Vector<double> update_u = system_matrix_u_inverse * (-1. * residual_u);

    const auto system_matrix_v_inverse =
        inverse_operator(system_matrix_v, solver, linear_part_v_inverse);

    Vector<double> update_v = system_matrix_v_inverse * (-1. * residual_v);

    affine_constraints.set_zero(update_u);
    affine_constraints.set_zero(update_v);

    new_solution_u += update_u;
    new_solution_v += update_v;
  }

  {
    Vector<double> residual_u = M_u * (new_solution_u - old_solution_u) - kappa  * theta * M_u * new_solution_v - kappa * (1.-theta) * M_u * old_solution_v;
    
    Vector<double> residual_v = (M_u + kappa * theta * M_newv_u - kappa * theta * Q1_newv_u) * new_solution_v - kappa * (1-theta) * (M_oldv_u + Q1_oldv_u) * old_solution_v + kappa * theta * (M_newu_u - Q1_newu_u - Q2_newu_u - S_newu_u) * new_solution_u + kappa * (1-theta) * (M_oldu_u - Q2_oldu_u - S_oldu_u) * old_solution_u;

    affine_constraints.set_zero(residual_u);
    affine_constraints.set_zero(residual_v);


    std::cout<<"norm of residual_u: "<<residual_u.linfty_norm()<<std::endl;
    std::cout<<"norm of residual_v: "<<residual_v.linfty_norm()<<std::endl;
    if ((residual_u.linfty_norm() > nonlinear_solver_tol)||(residual_v.linfty_norm() > nonlinear_solver_tol))
      throw ExcMessage("non converged");
  }

  old_solution_u = new_solution_u;
  old_solution_v = new_solution_v;
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
    t_end = 1000.;
    add_parameter("final time", t_end, "final time of the simulation");

    basename_u = "solution_u";
    add_parameter("base name_u", basename_u, "base name for 'u' output files");


    basename_v = "solution_v";
    add_parameter("base name_v", basename_v, "base name for 'v' output files");
  }

  void run();

private:
  SmartPointer<const TimeStep<dim>> p_time_step;

  double t_end;
  std::string basename_u;
  std::string basename_v;
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

  Vector<double> solution_u;
  Vector<double> solution_v;
  solution_u.reinit(dof_handler.n_dofs());
  solution_v.reinit(dof_handler.n_dofs());

  Vector<double> oldsolution_u;
  Vector<double> oldsolution_v;

  oldsolution_u.reinit(dof_handler.n_dofs());
  oldsolution_v.reinit(dof_handler.n_dofs());

  const double kappa = time_step.kappa;

  unsigned int n = 0;
  double t = 0;
  for (; t <= t_end + 1.0e-10;) {

    std::cout << "time step n = " << n << "\tt = " << t << std::endl;

    if (n == 0) {
      std::cout << "    interpolate first initial values" << std::endl;

      const auto &initial_values_u = coefficients.initial_values_u;
      const auto &initial_values_v = coefficients.initial_values_v;

      const auto lambda = [&](const Point<dim> &p, const unsigned int component) {
        Assert(component <= 1, ExcMessage("need exactly two components"));

        const auto value_u = initial_values_u(/* r = */ p[0]);
        const auto value_v = initial_values_v(/* r = */ p[0]);

        if (component == 0)
          return value_u.real();
        else
          return value_u.imag();
	
        if (component == 0)
          return value_v.real();
        else
          return value_v.imag();
      };
      VectorTools::interpolate(mapping,
                               dof_handler,
                               to_function<dim, /*components*/ 2>(lambda),
                               solution_u);

      VectorTools::interpolate(mapping,
                               dof_handler,
                               to_function<dim, /*components*/ 2>(lambda),
                               solution_v);
    } else {

      time_step.step(solution_u,solution_v,t);
     // time_step.step(solution_v,t);

    }

    {
      /* output: */
      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution_u, "solution_u");
      data_out.add_data_vector(solution_v, "solution_v");

      data_out.build_patches(/* FIXME: adjust for DEGREE of ansatz */);

      std::string name  = "Solution-" + std::to_string(n);
      

      {
        std::ofstream output(name + std::string(".vtk"));
        data_out.write_vtk(output);
      }
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
  ManufacturedSolution<dim> manufactured(coefficients, discretization);
  OfflineData<dim> offline_data(coefficients, discretization);
  TimeStep<dim> time_step(offline_data, manufactured);
  TimeLoop<dim> time_loop(time_step);
  ParameterAcceptor::initialize("gravwave1.prm");
  manufactured.prepare();
  offline_data.prepare();
  time_step.prepare();

  time_loop.run();

  return 0;
}

