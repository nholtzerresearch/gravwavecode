/* ---------------------------------------------------------------------
 *
 * This file aims to solve a GR problem utilizing a Reissner-Nordstrom metric
 *
 * ---------------------------------------------------------------------
 */


//Include files
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

//Standard C++ libraries
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <complex>

using namespace dealii;

typedef std::complex<double> value_type; //Gives complex<double> type the alias "value_type"

static constexpr value_type imag{0., 1.};

//{The GravWave class}
template <int dim>
class GravWave
{
public:
  GravWave(); //Class Constructor

  void run()
  {
    create_triangulation();
    setup_system();
    assemble_system();
    solve();
    output_result();
  }

  void create_triangulation();
  void setup_system();
  void setup_constraints();
  void assemble_system();
  void output_result();
  
  void solve();

 dealii::FESystem<dim>      finite_element_;
 dealii::MappingQ<dim>      mapping_;
 dealii::QGauss<dim>        quadrature_;
  

 dealii::Triangulation<dim> triangulation_;

 dealii::DoFHandler<dim>           dof_handler_;
 dealii::SparsityPattern           sparsity_pattern_;
 dealii::ConstraintMatrix          affine_constraints_;
//};
  // PARAMETER
  std::function<value_type(const Point<dim> &point)>f_func=   
   [](const Point<dim> & point) {
    double M = 1.0; //need to change before running
    double Q = 1.0;
    return  1.0-(2*M/point[0])+(Q*Q)/(point[0]*point[0]) + 0.0 * imag;
    };

  std::function<value_type(const Point<dim> &point)>fprime_func=
   [](const Point<dim> & point) {
    double M = 1.0; //need to change before running
    double Q = 1.0;
    return (2/(point[0]*point[0]))*(M-((Q*Q)/point[0])) + 0.0 * imag ;
    };
    
 std::function<value_type(const Point<dim> &point)>g_func=
   [&](const Point<dim> & point) {
   //
   double Q = 1.0;
   double q = 1.0;
  // return (2.0*f_func(point))/point[0] +(2.0 * imag*(q*Q)/point[0])+fprime_func(point);
  // return Q*q+f_func(point);
   return (2.0*f_func(point))/point[0] + 2.0 * imag*(q*Q)/point[0] +fprime_func(point);
   };

 std::function<value_type(const Point<dim> &point)>A_coefficient=
   [&](const Point<dim> & point) {
   //
   double Q = 1.0;
   double q = 1.0;

   return (2.0/point[0])*(f_func(point) +1.0) + (2.0 * imag*(q*Q))/point[0] ;
   };

  std::function<value_type(const Point<dim> &point)>B_coefficient=
    [&](const Point<dim> & point) {
    //
    return (2.0 + f_func(point)) + 0.0 * imag;
    };

 std::function<value_type(const Point<dim> &point)>C_coefficient=
   [&](const Point<dim> & point) {
   //
   return (2.0/point[0]) + 0.0 * imag;
   };

  std::function<value_type(const Point<dim> &point)>D_coefficient=
    [&](const Point<dim> & point) {
    //
    double Q=1.0;
    double q=1.0;
    return 0.0 + ( imag*(q*Q))/(point[0]*point[0]) ;
    };
   

    dealii::SparseMatrix<double> system_matrix_;

    dealii::Vector<double> system_right_hand_side_;
    dealii:: Vector<double> solution_;

   };



//@sect{GravWave::GravWave}

template <int dim>

GravWave<dim>::GravWave()
   : 
   finite_element_(FE_Q<dim>(1), 2),
   mapping_(1),    // PARAMETER
   quadrature_(3) // PARAMETER
{} 


// @sect{GravWave::create_triangulation}

template <int dim>
void

GravWave<dim>::create_triangulation()

{
  std::cout << "GravWave<dim>::create_triangulation()" << std::endl;

  GridGenerator::hyper_cube(triangulation_);

  triangulation_.refine_global(7); // PARAMETER

}

// @sect{GravWave::setup_system}

template <int dim>
void

GravWave<dim>::setup_system()
{
 std::cout << "GravWave<dim>::setup_system()" << std::endl;

 dof_handler_.initialize(triangulation_, finite_element_);
 std::cout << "        " << dof_handler_.n_dofs() << " DoFs" << std::endl;

 DoFRenumbering::Cuthill_McKee(dof_handler_);

  setup_constraints();
 
 DynamicSparsityPattern c_sparsity(dof_handler_.n_dofs(),

                                   dof_handler_.n_dofs());
 DoFTools::make_sparsity_pattern(dof_handler_,
                                 c_sparsity,
                                 affine_constraints_,
                                 false);
 sparsity_pattern_.copy_from(c_sparsity);
 
 system_matrix_.reinit(sparsity_pattern_);
 system_right_hand_side_.reinit(dof_handler_.n_dofs());
 solution_.reinit(dof_handler_.n_dofs());
}
// @sect{GravWave::setup_constraints}

template <int dim>
void
GravWave<dim>::setup_constraints()
{
 // deallog << "GravWave<dim>::setup_constraints()" << std::endl;
  std::cout << "GravWave<dim>::setup_constraints()" << std::endl;
  affine_constraints_.clear();

  VectorTools::interpolate_boundary_values(dof_handler_,
                                           0,
                                           ZeroFunction<dim>(2),
                                           affine_constraints_);

  DoFTools::make_hanging_node_constraints(dof_handler_, affine_constraints_);

  affine_constraints_.close();
}

// @sect{GravWave::assemble_system}

template <int dim>
void

GravWave<dim>::assemble_system()
{
  deallog << "GravWave<dim>::assemble_system()" << std::endl;

  this->system_matrix_          = 0.;
  this->system_right_hand_side_ = 0.;
  this->solution_               = 0.;

  const unsigned int dofs_per_cell = finite_element_.dofs_per_cell;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  FEValues<dim> fe_values(mapping_,
                          finite_element_,
                          quadrature_,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  FEValuesViews::Scalar<dim> real_part(fe_values, 0);
  FEValuesViews::Scalar<dim> imag_part(fe_values, 1);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const unsigned int n_q_points = quadrature_.size();

  for (auto cell : dof_handler_.active_cell_iterators())
    {
      cell_matrix = 0.;
      cell_rhs    = 0.;
      
      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      const auto &quadrature_points = fe_values.get_quadrature_points();

      const auto rot_x = 1.;
      const auto rot_y = -imag;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const auto position = quadrature_points[q_point];
          const auto A_coef = A_coefficient(position);

          const auto JxW = fe_values.JxW(q_point);

          // index i for test space, index j for ansatz space
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const auto value_i = rot_x * real_part.value(i, q_point) +
                                   rot_y * imag_part.value(i, q_point);

              const auto grad_i = rot_x * real_part.gradient(i, q_point) +
                                  rot_y * imag_part.gradient(i, q_point);

              cell_rhs(i) += (value_i * JxW).real();

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const auto grad_j = real_part.gradient(j, q_point) +
                                      imag * imag_part.gradient(j, q_point);

                  cell_matrix(i, j) += (grad_j * A_coef * grad_i).real();

                } // for j 

            }     // for i 

        }         // for q_point

      affine_constraints_.distribute_local_to_global(cell_matrix,
                                                     cell_rhs,
                                                     local_dof_indices,
                                                     system_matrix_,
                                                     system_right_hand_side_);
    } // loop over dof_handler iterators 
  }






// @sect{GravWave:: solve}

template <int dim>
void

GravWave<dim>::solve()
{
  std::cout << "GravWave<dim>::solve()" << std::endl;

  affine_constraints_.set_zero(solution_);

  SparseDirectUMFPACK solver;
  solver.initialize(system_matrix_);
  solver.vmult(solution_, system_right_hand_side_);

  affine_constraints_.distribute(solution_);
}

// @sect{GravWave:: output_results}

template <int dim>
void

GravWave<dim>::output_result()
{
  std::cout << "GravWave<dim>::output_result()" << std::endl;
 
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_);

  data_out.add_data_vector(system_right_hand_side_, "rhs");
  data_out.add_data_vector(solution_, "solution");

  data_out.build_patches();

  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);
}

//}



//@sect{The <code>main</code> function}
int main()

{
  static constexpr int dim = 2;

  GravWave<dim> gravwave_problem;

 // gravwave_problem.run();

  return 0;
}
                                           
