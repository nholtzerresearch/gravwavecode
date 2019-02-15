/* ---------------------------------------------------------------------
 *
 * This file aims to solve a GR problem utilizing a Reissner-Nordstrom metric
 *
 * ---------------------------------------------------------------------
 */


//Include files
#include <deal.II/base/utilities.h>
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
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

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
    define_boundary_conds(); //ADDED FOR DIRICH. BCS
    setup_system();
    assemble_system();
    solve();
    output_result();
  }

  void create_triangulation();
  void define_boundary_conds(); //ADDED FOR DIRICH. BCS
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

 SparseMatrix<double>              mass_matrix_;
 SparseMatrix<double>              laplace_matrix_;
 SparseMatrix<double>		   system_matrix_;
 SparseMatrix<double>              system_RHS_;
 std::vector<double>               nodeLocation;//ADDED FOR DIRICH BCS
 std::map<unsigned int,double>     boundary_values;//ADDED FOR DIRICH BCS
 

 dealii::Vector<value_type>        old_solution_;
 dealii::Vector<double>		   solution_;
 dealii::Vector<double>            system_right_hand_side_; 
 
// dealii::SparseMatrix<double> system_matrix_;

// dealii::Vector<double> system_right_hand_side_;
// dealii:: Vector<double> solution_;


 double R, bc_left, bc_right;
 double time_step_, time_;
 unsigned int timestep_number_;
 const double theta_;

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

 std::function<value_type(const Point<dim> &point)>a_coefficient=
   [&](const Point<dim> & point) {
   //
   double Q = 1.0;
   double q = 1.0;

   return (2.0/point[0])*(f_func(point) +1.0) + (2.0 * imag*(q*Q))/point[0] ;
   };

  std::function<value_type(const Point<dim> &point)>b_coefficient=
    [&](const Point<dim> & point) {
    //
    return (2.0 + f_func(point)) + 0.0 * imag;
    };

 std::function<value_type(const Point<dim> &point)>c_coefficient=
   [&](const Point<dim> & point) {
   //
   return (2.0/point[0]) + 0.0 * imag;
   };

  std::function<value_type(const Point<dim> &point)>d_coefficient=
    [&](const Point<dim> & point) {
    //
    double Q=1.0;
    double q=1.0;
    return 0.0 + ( imag*(q*Q))/(point[0]*point[0]) ;
    };
   

    //dealii::SparseMatrix<double> system_matrix_;

   // dealii::Vector<double> system_right_hand_side_;
   // dealii:: Vector<double> solution_;

   };
//@sect{IC,RHS,BCs}
//Here we declare three more classes for the implementation of the IC, RHS, and non-homogeneous Dirichlet BCs.
template<int dim>
class Initial : public Function<dim>
{
public:
  Initial (const unsigned int n_components =1, 
               const double time=0.0): Function<dim>(n_components, time){}
  virtual double value (const Point<dim> &p,
                        const unsigned int component =0) const;
};

template<int dim>
double Initial<dim>::value (const Point<dim> &p,
                            const unsigned int ) const
{
  double t = this->get_time ();

       double a = 1.0;
       double mu = 1.0;
       double sigma = 1.0;
      
       return a.*std::exp(-((p[0]-mu)-t)*((p[0]-mu)-t)/(2.*sigma*sigma));
}

template<int dim>
class InitialValues : public Function<dim>
{
public:
  InitialValues (const unsigned int n_components =1,
	         const double time =0.0)
    :
    Function<dim>(n_components, time)
   {}

   virtual double value (const Point<dim> &p,
                         const unsigned int component) const;
};

template <int dim>
double InitialValues<dim>::value (const Point<dim> &p,
             			  const unsigned int component) const
{
  return Initial<dim>(1, this->get_time()).value (p, component);
} 


template <int dim>
class RightHandSide : public Function<dim>
{
public:
 RightHandSide () : Function<dim>() {}

 virtual double value (const Point<dim> &p,
                       const unsigned int component=0) const;
};

/*template <int dim>
class BoundaryValues : public Function <dim>
{
public:
 BoundaryValues () : Function<dim>() {}

 virtual double value (const Point<dim> &p,
                       const unsigned int component=0) const;
};*/

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int) const
{
  double return_value = 0.0;

  return return_value;
}


/*template<int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int) const
{
  if (p[0]=0.001){
    return bc_left; //PARAMETER TO CHANGE POSSIBLY
  }
  if(p[0]=1){
    return bc_right;//PARAMETER
  }
}*/
//@sect{GravWave::GravWave}

template <int dim>

GravWave<dim>::GravWave()
   : 
   finite_element_(FE_Q<dim>(1), 2),
   mapping_(1),    // PARAMETER
   quadrature_(3), // PARAMETER
   time_step_(1./64), //PARAMETER
   time_(time_step_),
   timestep_number_(1),
   theta_(0.5)
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


// @sect{GravWave::define_boundary_conds} ADDED SECTION FOR DIRICH BCS

template <int dim>
void GravWave<dim>::define_boundary_conds(){

const unsigned int totalNodes = dof_handler_.n_dofs(); //Total number of nodes

 for(unsigned int globalNode=0; globalNode<totalNodes; globalNode++){
    if(nodeLocation[globalNode] == 0){
      boundary_values[globalNode] = bc_left;;
    }
    if(nodeLocation[globalNode] == R){ 
      boundary_values[globalNode] = bc_right;
      }
    }
}


// @sect{GravWave::setup_system}

template <int dim>
void

GravWave<dim>::setup_system()
{
  //Define constants for problem (Dirichlet boundary values) ADDED FOR DIRICH BC
 bc_left =0.001 ; bc_right =1 ;
 std::cout << "GravWave<dim>::setup_system()" << std::endl;

 dof_handler_.initialize(triangulation_, finite_element_);
 std::cout << "        " << dof_handler_.n_dofs() << " DoFs" << std::endl;

 DoFRenumbering::Cuthill_McKee(dof_handler_);
 
 define_boundary_conds(); //ADDED FOR DIRICH BCs (call the function) 
 
 setup_constraints();
 
 DynamicSparsityPattern c_sparsity(dof_handler_.n_dofs(),

                                   dof_handler_.n_dofs());
 DoFTools::make_sparsity_pattern(dof_handler_,
                                 c_sparsity,
                                 affine_constraints_,
                                 false);
 sparsity_pattern_.copy_from(c_sparsity);
 
// mass_matrix_.reinit(sparsity_pattern_);
// laplace_matrix_.reinit(sparsity_pattern_);
 system_matrix_.reinit(sparsity_pattern_);
 system_RHS_.reinit(sparsity_pattern_);
// MatrixCreator::create_mass_matrix(dof_handler_, quadrature_, mass_matrix_);

// MatrixCreator::create_laplace_matrix(dof_handler_, quadrature_, laplace_matrix_);

 system_right_hand_side_.reinit(dof_handler_.n_dofs());
 solution_.reinit(dof_handler_.n_dofs());
 old_solution_.reinit(dof_handler_.n_dofs());
}
// @sect{GravWave::setup_constraints}

template <int dim>
void
GravWave<dim>::setup_constraints()
{
 // deallog << "GravWave<dim>::setup_constraints()" << std::endl;
  std::cout << "GravWave<dim>::setup_constraints()" << std::endl;
  affine_constraints_.clear();

 /* VectorTools::interpolate_boundary_values(dof_handler_,
                                           0,
                                           ZeroFunction<dim>(2),
                                           affine_constraints_);*/

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
  this->system_RHS_             = 0.;
  this->system_right_hand_side_ = 0.;
  this->solution_               = 0.;

  const unsigned int dofs_per_cell = finite_element_.dofs_per_cell;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M_matrix_one(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M_matrix_two(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> K_matrix_one(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> K_matrix_two(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> A_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_rhs_matrix(dofs_per_cell, dofs_per_cell);


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
      cell_rhs_matrix = 0;
      M_matrix_one = 0.;
      M_matrix_two = 0.;
      K_matrix_one = 0.; 
      K_matrix_two = 0.;
      A_matrix = 0.;
      cell_rhs    = 0.;
      
      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      const auto &quadrature_points = fe_values.get_quadrature_points();

      const auto rot_x = 1.;
      const auto rot_y = -imag;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const auto position = quadrature_points[q_point];
          const auto a_coef = a_coefficient(position);
          const auto b_coef = b_coefficient(position);
          const auto c_coef = c_coefficient(position);
          const auto d_coef = d_coefficient(position);

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
                  const auto value_j = rot_x * real_part.value(j, q_point) +
                                   rot_y * imag_part.value(j, q_point);

                  const auto grad_j = real_part.gradient(j, q_point) +
                                      imag * imag_part.gradient(j, q_point);
                  M_matrix_one(i,j) += (value_j * (c_coef+time_step_*theta_) * value_i).real();
                  M_matrix_two(i,j) += (value_j * (c_coef- time_step_* (1.-theta_)*d_coef) * value_i).real();
                  K_matrix_one(i,j) += (value_j *  2.* grad_i).real(); 
                  K_matrix_two(i,j) += (value_j * (2. + time_step_ * theta_ * a_coef) * grad_i).real();
                  A_matrix(i,j)     += (grad_j * (time_step_ * theta_ * b_coef) * grad_i).real();
                  cell_matrix(i, j) += (M_matrix_one(i,j) + K_matrix_one(i,j));
                  cell_rhs_matrix(i,j) += (M_matrix_two(i,j) + K_matrix_two(i,j) - A_matrix(i,j));
                } // for j 

            }     // for i 

        }         // for q_point

      affine_constraints_.distribute_local_to_global(cell_matrix,
                                                     cell_rhs,
                                                     local_dof_indices,
                                                     system_matrix_,
                                                     system_right_hand_side_);
       /* affine_constraints_.distribute_local_to_global(cell_matrix,
                                                     cell_rhs_matrix,
                                                     local_dof_indices,
                                                     system_matrix_,
                                                     system_RHS_);*/

    } // loop over dof_handler iterators 
   // MatrixTools::apply_boundary_values (boundary_values, cell_matrix, false);ADDED FOR BCS
      MatrixTools::apply_boundary_values (boundary_values,
                                    system_matrix_,//ADDED FOR BCS
                                    solution_,
                                    system_right_hand_side_);

//  }

//std::map<types::global_dof_index,double> boundary_values;
/*VectorTools::interpolate_boundary_values (dof_handler_,
					  0,
					  BoundaryValues<dim>(),
					  boundary_values);*/
/*MatrixTools::apply_boundary_values (boundary_values, 
				    system_matrix_,//ADDED FOR BCS
				    solution_,
				    system_right_hand_side_);*/
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
  //solver.vmult(solution_, system_RHS_);
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
 // static constexpr int dim = 2;
 // GravWave<dim> gravwave_problem;

 // gravwave_problem.run();

  return 0;
}
                                           
