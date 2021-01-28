/* ---------------------------------------------------------------------
 *
 * This file aims to solve the standard wave equation u_tt-u_rr=0
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
#include <fstream>
#include <complex>
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
    R_0 = 1.0;
    add_parameter("R_0", R_0, "inner radius of the computational domain");

    R_1 = 10.0;
    add_parameter("R_1", R_1, "outer radius of the computational domain");

    A_param = 0.1;
    add_parameter("A_param", A_param, "height of initial gaussian");

    b_param = 5.0;
    add_parameter("b_param", b_param, "position of peak of initial gaussian");

    c_param = 1.0;
    add_parameter("c_param", c_param, "wave speed of initial gaussian");

    d_param = 1.0;
    add_parameter("d_param", d_param, "standard deviation of initial gaussian");

    /* IC = A exp[-{((x-b)+cT)^2/2d^2}], initial_values1 = IC(0) */
   
    initial_values1 = [&](double r) {
        return A_param*std::exp(- (r-b_param)*(r-b_param)/(2.* pow(d_param,2.));
    };

    //Need to solve for u^2(0) = initial_values1 + kappa d/dT(IC)(0)
    //IC_T = d/dT(IC)(0)

    IC_T = [&](double r) {
        return - c_param / pow(d_param, 2.) * (r-b_param) * initial_values1;
    };

    boundary_values = [&](double r, double t) {
       if (r == R_0)
	 return 0.; 
	//return t;
       if (r == R_1)
	 return 0.;
    };

  }

  /* Publicly readable parameters: */
  double R_0;
  double R_1;
  double A_param;
  double b_param;
  double c_param;
  double d_param;

  /* Publicly readable functions: */
  std::function<value_type(double)> initial_values1;
  std::function<value_type(double)> IC_T;
  std::function<value_type(double, double)> boundary_values;

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
  unsigned int refinement;

  void parse_parameters_callback()
  {
    p_triangulation.reset(new Triangulation<dim>);
    auto &triangulation = *p_triangulation;

    GridGenerator::hyper_cube(
        triangulation, p_coefficients->R_0, p_coefficients->R_1);
    /* IDS 0-left, 1-right */

    triangulation.begin_active()->face(0)->set_boundary_id(0); // FIXME
    triangulation.begin_active()->face(1)->set_boundary_id(1); // FIXME

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

  //unsigned int refinement;

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
  SparseMatrix<double> stiffness_matrix;
  SparseMatrix<double> stiffness_matrix_unconstrained;

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
  stiffness_matrix.reinit(sparsity_pattern);
  stiffness_matrix_unconstrained.reinit(sparsity_pattern);

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
      boundary_value_map); // '0':= enforce dirichlet bc for ID 0/ID 1 nat BC
 
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
  stiffness_matrix = 0.;
  stiffness_matrix_unconstrained = 0.;


  const auto &mapping = p_discretization->mapping();
  const auto &finite_element = p_discretization->finite_element();
  const auto &quadrature = p_discretization->quadrature();
  

  const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

  FullMatrix<double> cell_nodal_mass_matrix(dofs_per_cell, dofs_per_cell);
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

    cell_nodal_mass_matrix = 0.;
    cell_mass_matrix = 0.;
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

      const auto JxW = fe_values.JxW(q_point);
      dealii::Tensor<1,dim,double> u_grad;

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

          const auto mass_term = value_i * value_j * JxW;
          cell_mass_matrix(i, j) += mass_term.real();

          const auto stiffness_term = grad_i[0] * grad_j[0] * JxW;//add this in for additional +u term in pde - value_i * value_j * JxW;
	  cell_stiffness_matrix(i,j) += stiffness_term.real();
        } // for j
      }   // for i
    }     // for q_point



    affine_constraints.distribute_local_to_global(
        cell_nodal_mass_matrix, local_dof_indices, nodal_mass_matrix);
    

    affine_constraints.distribute_local_to_global(
        cell_mass_matrix, local_dof_indices, mass_matrix);


    mass_matrix_unconstrained.add(local_dof_indices, cell_mass_matrix);


    affine_constraints.distribute_local_to_global(
        cell_stiffness_matrix, local_dof_indices, stiffness_matrix);

    stiffness_matrix_unconstrained.add(local_dof_indices,cell_stiffness_matrix);


  } // cell
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

template <int dim>
class TimeStep : public ParameterAcceptor
{
public:
  TimeStep(const OfflineData<dim> &offline_data,const Discretization<dim> &discretization)
      : ParameterAcceptor("C - TimeStep")
      , p_offline_data(&offline_data)
      , p_discretization(&discretization)
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
  void step(Vector<double> &old_solution, Vector<double> &oldest_solution, double new_t);


  SmartPointer<const Discretization<dim>> p_discretization;
  SmartPointer<const OfflineData<dim>> p_offline_data;
  double kappa;
  double theta;
  unsigned int linear_solver_limit;
  double linear_solver_tol;
  unsigned int nonlinear_solver_limit;
  double nonlinear_solver_tol;

  SparseMatrix<double> linear_part;
  SparseDirectUMFPACK linear_part_inverse;



private:

};


template <int dim>
void TimeStep<dim>::prepare()
{
  std::cout << "TimeStep<dim>::prepare()" << std::endl;
  const auto &offline_data = *p_offline_data;
  const auto &finite_element = p_discretization->finite_element();


  linear_part.reinit(offline_data.sparsity_pattern);
  auto &M_c = offline_data.mass_matrix;
  auto &S_c = offline_data.stiffness_matrix;


  const unsigned int dofs_per_cell = finite_element.dofs_per_cell;

  linear_part.copy_from(M_c);

  linear_part.add((1. - theta) * pow(kappa,2.) / 2., S_c);

  linear_part_inverse.initialize(linear_part);


}


template<int dim>
void apply_boundary_values(const OfflineData<dim> &offline_data,
                           const Coefficients &coefficients,
                           double t,
                           Vector<double> &vector_new,
			                     Vector<double> &vector_old)
{
  const auto &mapping = offline_data.p_discretization->mapping();
  const auto &dof_handler = offline_data.dof_handler;

  const auto &boundary_values = coefficients.boundary_values;
  const auto &r_0 = coefficients.R_0;
  const auto &r_1 = coefficients.R_1;
  const auto &refine = offline_data.p_discretization->refinement;
  
  std::cout<<"My refinement is " << refine << std::endl;


  std::map<types::global_dof_index, double> boundary_value_map;
  t = (vector_old[2]-vector_old[1])/((r_1-r_0)/pow(2.,refine));

  const auto lambda = [&](const Point<dim> &p, const unsigned int component) {
    Assert(component <= 1, ExcMessage("need exactly two components"));

    const auto value = boundary_values(/* r = */ p[0], /* t = */ t);
    std::cout<<"boundary value function output is: "<<value<<std::endl;

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

    //std::cout<<"global dofs: "<<boundary_value_map[0]<<std::endl;
    //vector[it.first] = it.second;

      vector_new[it.first] = it.second;

  }

}


template <int dim>
//void TimeStep<dim>::step(Vector<double> &old_solution, double new_t) 
void TimeStep<dim>::step(Vector<double> &old_solution,Vector<double> &oldest_solution,double new_t) 

{
  const auto &offline_data = *p_offline_data;
  const auto &coefficients = *offline_data.p_coefficients;
  const auto &affine_constraints = offline_data.affine_constraints;
  const auto &finite_element = p_discretization->finite_element();

  const auto M_c = linear_operator(offline_data.mass_matrix);
  const auto S_c = linear_operator(offline_data.stiffness_matrix);
  const auto M_u = linear_operator(offline_data.mass_matrix_unconstrained);
  const auto S_u = linear_operator(offline_data.stiffness_matrix_unconstrained);
  const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
  Vector<double> temp;
  Vector<double> temp1;
  GrowingVectorMemory<Vector<double>> vector_memory;
  typename VectorMemory<Vector<double>>::Pointer p_new_solution(vector_memory);
 //typename VectorMemory<Vector<double>>::Pointer p_old_solution(vector_memory);
  auto &new_solution = *p_new_solution;
  //auto &old_solution = *p_old_solution;


  old_solution = oldest_solution;
  new_solution = old_solution;
 
  //apply_boundary_values(offline_data, coefficients, new_t, new_solution);
  
  apply_boundary_values(offline_data, coefficients, new_t, new_solution, old_solution);

  for (unsigned int m = 0; m < nonlinear_solver_limit; ++m) {
    Vector<double> residual = M_u * (new_solution - 2.* old_solution + oldest_solution) +
                              pow(kappa,2.) * (1. - theta)/2. * S_u * (new_solution + old_solution) +
                              theta * pow(kappa,2.) * S_u * old_solution;


    affine_constraints.set_zero(residual);

    if (residual.linfty_norm() < nonlinear_solver_tol)
      break;
    
    auto system_matrix = linear_operator(linear_part); 


    SolverControl solver_control(linear_solver_limit, linear_solver_tol);
    SolverGMRES<> solver(solver_control);

    auto system_matrix_inverse =
        inverse_operator(system_matrix, solver, linear_part_inverse);//deleted const

    Vector<double> update = system_matrix_inverse * (-1. * residual);

    affine_constraints.set_zero(update);


    new_solution += update;

    }

  {
    Vector<double> residual = M_u * (new_solution - 2. * old_solution + oldest_solution) +
                              pow(kappa,2.) * (1. - theta) / 2.* S_u * (new_solution + old_solution) +
                              theta * pow(kappa,2.) * S_u * old_solution;

    affine_constraints.set_zero(residual);
    std::cout<<"norm of residual: "<<residual.linfty_norm()<<std::endl;
    if (residual.linfty_norm() > nonlinear_solver_tol)
      throw ExcMessage("non converged");
  }

  old_solution = new_solution;
  oldest_solution = old_solution;

}


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */


template <int dim>
class TimeLoop : public ParameterAcceptor
{
public:
  TimeLoop(TimeStep<dim> &time_step)
      : ParameterAcceptor("D - TimeLoop")
      , p_time_step(&time_step)// deleted const before TimeStep
  {
    t_end = 1.0;
    add_parameter("final time", t_end, "final time of the simulation");

    basename = "solution";
    add_parameter("base name", basename, "base name for output files");
  }

  void run();

private:
  SmartPointer<TimeStep<dim>> p_time_step;//deleted const before TimeStep

  double t_end;
  std::string basename;
};


template <int dim>
void TimeLoop<dim>::run()
{
  std::cout << "TimeLoop<dim>::run()" << std::endl;

  auto &time_step = *p_time_step;//deleted const before auto
  const auto &offline_data = *time_step.p_offline_data;
  const auto &coefficients = *offline_data.p_coefficients;
  const auto &discretization = *offline_data.p_discretization;

  const auto &mapping = discretization.mapping();
  const auto &dof_handler = offline_data.dof_handler;

  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());

  Vector<double> older_solution;
  older_solution.reinit(dof_handler.n_dofs());
  const double kappa = time_step.kappa;

  unsigned int n = 0;
  double t = 0;
  for (; t <= t_end + 1.0e-10;) {

    std::cout << "time step n = " << n << "\tt = " << t << std::endl;

    if (n == 0) {
      std::cout << "    interpolate first initial values" << std::endl;
      /* interpolate first initial conditions: */

      const auto &initial_values1 = coefficients.initial_values1;
      
      const auto lambda = [&](const Point<dim> &p, const unsigned int component) {
        Assert(component <= 1, ExcMessage("need exactly two components"));

        const auto value1 = initial_values1(/* r = */ p[0]);

        if (component == 0)
          return value1.real();
        else
          return value1.imag();
      };
      VectorTools::interpolate(mapping,
                               dof_handler,
                               to_function<dim, /*components*/ 2>(lambda),
                               older_solution);

    }
 
    if (n == 1) {
      std::cout << "    interpolate second initial values" << std::endl;
      /* interpolate second initial conditions: */
      
      const auto &initial_values1 = coefficients.initial_values1;
      const auto &IC_T = coefficients.IC_T;

      const auto lambda = [&](const Point<dim> &p, const unsigned int component) {
        Assert(component <= 1, ExcMessage("need exactly two components"));

        const auto value2 = initial_values1(/* r = */ p[0]) + kappa * IC_T(/* r = */ p[0]);

        if (component == 0)
          return value2.real();
        else
          return value2.imag();
      };
      VectorTools::interpolate(mapping,
                               dof_handler,
                               to_function<dim, /*components*/ 2>(lambda),
                               solution);
    }
    else {
      time_step.step(solution,older_solution,t);

    }

//  Only run output every x steps
    if((n > 1))// && (n % 100 == 0))
    {
      /* output: */
      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");


      data_out.build_patches(/* FIXME: adjust for DEGREE of ansatz */);

      std::string name = basename + std::to_string(n);

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
  OfflineData<dim> offline_data(coefficients, discretization);
  TimeStep<dim> time_step(offline_data,discretization);
  TimeLoop<dim> time_loop(time_step);
  ParameterAcceptor::initialize("waveeqn_test.prm");
  offline_data.prepare();
  time_step.prepare();

  time_loop.run();

  return 0;
}

