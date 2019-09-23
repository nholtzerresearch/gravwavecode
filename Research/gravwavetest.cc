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
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/identity_matrix.h>
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
#include <algorithm>
using namespace dealii;
typedef std::complex<double> value_type; //Gives complex<double> type the alias "value_type"

static constexpr value_type imag{0., 1.};

//{The GravWave class}
template <int dim>
class GravWave
{
public:
  GravWave(); //Class Constructor

  void run();

  void create_triangulation();
  void setup_system();
  void setup_constraints();
  void assemble_system();
  void output_result(const unsigned int timestep_number);
  void solve();

 dealii::FESystem<dim>      finite_element_;
 dealii::MappingQ<dim>      mapping_;
 dealii::QGauss<dim>        quadrature_;
  

 dealii::Triangulation<dim> triangulation_;

 dealii::DoFHandler<dim>    dof_handler_;
 dealii::SparsityPattern    sparsity_pattern_;
 dealii::ConstraintMatrix   affine_constraints_;
 SparseMatrix<double>       system_matrix_;
 SparseMatrix<double>       system_RHS_;
 

 dealii::Vector<double>      old_solution_;
 dealii::Vector<double>      solution_update_,solution_;
 dealii::Vector<double>      system_right_hand_side_; 
 



 double bc_left, bc_right, time_step_, final_time_;
 double time_;
 const double theta_, R0_, Rf_;
};
//@sect{IC,RHS,BCs}
//Here we declare three more classes for the implementation of the IC, RHS, and non-homogeneous Dirichlet BCs.
  template<int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues (): Function<dim>(2){}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
  };

  template<int dim>
  inline
  void InitialValues<dim>::vector_value (const Point<dim> &p,
                                         Vector<double>   &values) const

    {  
      	values(0) = 0.5;
      	values(1) = 0;
    }

    template<int dim>
    void InitialValues<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                std::vector<Vector<double> >   &value_list) const
  {
    
    for (unsigned int p=0; p<points.size(); ++p)
      InitialValues<dim>::vector_value (points[p], value_list[p]);
  }
template <int dim>
class RightHandSide : public Function<dim>
{
public:
 RightHandSide () : Function<dim>() {}

 virtual double value (const Point<dim> &p,
                       const unsigned int component=0) const;
};

template <int dim>
class BoundaryValues : public Function <dim>
{
public:
 BoundaryValues () : Function<dim>(2) {}

 virtual double value (const Point<dim> &p,
                       const unsigned int component=0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int) const
{
  double return_value = 0.0;

  return return_value;
}
template<int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int component) const
{
  // MM FIXME: These values should be set in the constructor:
  double R0_      = 0.001;
  double Rf_      = 0.0186;
  double bc_left  = 0.001;
  double bc_right = 0.001;;

  if (p[0] <= R0_ + 1.e-6){
    if (component == 0)
      return bc_left; //PARAMETER TO CHANGE POSSIBLY
    else
      return 0.;
  }

 if (p[0] >= Rf_ - 1.e-6){
    if (component == 0)
      return bc_right; //PARAMETER TO CHANGE POSSIBLY
    else
      return 0.;
  }

  return 0;
}


//@sect{GravWave::GravWave}

template <int dim>

GravWave<dim>::GravWave()
   : 
   finite_element_(FE_Q<dim>(1), 2),
   mapping_(1),    // PARAMETER
   quadrature_(4), // PARAMETER
   time_step_(1./64), //PARAMETER
   final_time_(1./32),
   time_(0.0),
   theta_(1.0),
   //try changing euler scheme
   R0_(0.001),
   Rf_(0.0186)
{} 


// @sect{GravWave::create_triangulation}

template <int dim>
void

GravWave<dim>::create_triangulation()

{
  std::cout << "GravWave<dim>::create_triangulation()" << std::endl;

  GridGenerator::hyper_cube(triangulation_,R0_,Rf_);
 
  // Set boundary indicators:
  triangulation_.begin_active()->face(0)->set_boundary_id(1);
  triangulation_.begin_active()->face(1)->set_boundary_id(1);

  triangulation_.refine_global(4); // PARAMETER

}


// @sect{GravWave::setup_system}

template <int dim>
void

GravWave<dim>::setup_system()
{
 std::cout << "GravWave<dim>::setup_system()" << std::endl;

 dof_handler_.initialize(triangulation_, finite_element_);
 std::cout << "        " << dof_handler_.n_dofs() << " DoFs" << std::endl;
 std::cout << "        " <<  "bc_left = " << bc_left  << std::endl;
 std::cout << "        " <<  "bc_right = " << bc_right  << std::endl;  
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
 system_RHS_.reinit(sparsity_pattern_);

 system_right_hand_side_.reinit(dof_handler_.n_dofs());
 solution_update_.reinit(dof_handler_.n_dofs());
 solution_.reinit(dof_handler_.n_dofs());
 old_solution_.reinit(dof_handler_.n_dofs());
}
// @sect{GravWave::setup_constraints}

template <int dim>
void
GravWave<dim>::setup_constraints()
{
  deallog << "GravWave<dim>::setup_constraints()" << std::endl;
  affine_constraints_.clear();

 VectorTools::interpolate_boundary_values(dof_handler_,
                                          1,
                                          BoundaryValues<dim>(),
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
  this->system_RHS_             = 0.;
  this->system_right_hand_side_ = 0.;
  this->solution_               = 0.;

  const unsigned int dofs_per_cell = finite_element_.dofs_per_cell;
  std::cout << "dofs_per_cell" << dofs_per_cell <<std::endl;
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_rhs_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M_matrix_one(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> K_matrix_two(dofs_per_cell, dofs_per_cell);
  Tensor<1,dim,value_type>  NewGrad_i, NewGrad1_i;
  NewGrad_i[0] = 2.;
  NewGrad1_i[0] = 2. + time_step_ * theta_;
  Vector<double> cell_rhs(dofs_per_cell);
  Vector<double>     tmp_vec(dofs_per_cell);
  Vector<double> temp_i(dofs_per_cell);
  Vector<double> temp_j(dofs_per_cell);
  //temp_i[0]=1.;
  //temp_j[0]=1.;
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
      cell_matrix  = 0.;
      cell_rhs     = 0.;
      M_matrix_one = 0.;
      K_matrix_two = 0.;
      cell_rhs     = 0.;
      tmp_vec      = 0.; 
      temp_i       = 0.;
      temp_j       = 0.;
      temp_i[0]=1.;
      temp_j[0]=1.;
      /*for (int v=0; v<8; v++) {
               std::cout << "temp_i[" << v <<"]: ";
               std::cout <<temp_i[v]<<std::endl;
            }*/


      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      const auto &quadrature_points = fe_values.get_quadrature_points();

      const auto rot_x = 1.;
      const auto rot_y = -imag;

       for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const auto position = quadrature_points[q_point];
          const auto JxW = fe_values.JxW(q_point);
          // index i for test space, index j for ansatz space
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            { 
             // cell_rhs(i) += 1.0; 
              const auto value_i = rot_x * real_part.value(i, q_point) +
                                   rot_y * imag_part.value(i, q_point);
              const auto grad_i = rot_x * real_part.gradient(i, q_point) +
                                  rot_y * imag_part.gradient(i, q_point);
              const auto newgrad1_i = (NewGrad_i)*grad_i;

              const auto newgrad2_i = (NewGrad1_i)*grad_i;
             

		for (unsigned int j = 0; j < dofs_per_cell; ++j)
                { 
                  const auto value_j = rot_x * real_part.value(j, q_point) +
                                       rot_y * imag_part.value(j, q_point);

                  const auto grad_j = real_part.gradient(j, q_point) +
                                      imag * imag_part.gradient(j, q_point);
		  if (j == i){
		    M_matrix_one[i][j] = 1.;
		    K_matrix_two[i][j] = 1.;
		    }
                  //K_matrix_two(i,j) += (value_j * newgrad2_i).real();
                  cell_matrix(i, j) += M_matrix_one(i,j);
                  cell_rhs_matrix(i,j) += (K_matrix_two(i,j));

		  //if (j==i){
                   // cell_matrix(i, j) += 1.real();;
		 // }
		  //std::cout<<"size of value_j "<<sizeof(value_j.real())<<std::endl;

                } // for j
	    //std::cout<<"size of value_j "<<sizeof(value_j)<<std::endl;
            }     // for i 
        }         // for q_point
       for (unsigned int i = 0; i == cell_rhs.size()+1; ++i)
         {
         
           tmp_vec(i)=old_solution_(local_dof_indices[i]);
         }


       cell_rhs_matrix.vmult(cell_rhs, tmp_vec);
       for (int v=0; v<8; v++) {
	    cell_rhs[v]=.5;
	}
/*	for (int w=0; w<8; w++) {
      std::cout<<"cell rhs[ "<< w << "]"<< cell_rhs[w] <<std::endl;
        }*/
       affine_constraints_.distribute_local_to_global( cell_matrix,
                                                       cell_rhs,
                                                       local_dof_indices,
                                                       system_matrix_,
                                                       system_right_hand_side_);

}
	 std::cout<<"rows: "<<system_matrix_.m()<< std::endl;
	 std::cout<<"cols: "<<system_matrix_.n()<< std::endl;
         std::cout<<"rhs length: "<<system_right_hand_side_.size()<<std::endl;
	 for (int i=0; i< system_matrix_.m(); i++){
	 	system_matrix_.set(i,i,1.0);
	 }	
	 for (int j=0; j< system_right_hand_side_.size(); j++){
                system_right_hand_side_[j]=0.5;
         }


	 std::cout << "K_matrix_two size: " <<K_matrix_two.size() << std::endl;
	 std::string name1 = std::string("outfile");
	 std::ofstream outfile(name1);

	 system_matrix_.print(outfile, false, true);
	 
	 std::string name2 = std::string("outfilerhs");
         std::ofstream outfilerhs(name2);

         system_right_hand_side_.print(outfilerhs, false, true);

	
         //std::cout << "system_matrix_: " << system_matrix_.print(StreamType &out,									    const bool across = false,  const bool diagonal_first=true) const <<std::endl;
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

GravWave<dim>::output_result(const unsigned int timestep_number)
{
  std::cout << "GravWave<dim>::output_result(" << timestep_number << ")"
            << std::endl;
 
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_);
  data_out.add_data_vector(system_right_hand_side_, "rhs");
  data_out.add_data_vector(solution_, "solution");

  data_out.build_patches();
  
  std::string name = std::string("solution-") + std::to_string(timestep_number);
  
  name += std::string(".vtk");

  std::ofstream output(name);

  //data_out.write_gnuplot(output);
  data_out.write_vtk(output);

}

// @sect{GravWave:: run}
template <int dim>
void

GravWave<dim>::run()
{

    create_triangulation();
    setup_system();

    VectorTools::interpolate(dof_handler_, 
                             InitialValues<dim>(), 
                             solution_); 

    output_result(0);
    unsigned int timestep_number = 1;
    time_+=time_step_;
    std::cout << "time: " << time_ <<std::endl;
    std::cout << "final_time: " << final_time_ <<std::endl;
    while (time_ <= final_time_)
      {
	std::cout << "GravWave<dim>::Time Stepping()" << std::endl;
         old_solution_ = solution_;

	 std::cout << std::endl
	           << "Time step #" << timestep_number << "; " 
	           << "advancing to t = " << time_ << "." << std::endl;
 
         assemble_system();

         solve();

         output_result(timestep_number);
      
         time_ += time_step_;
         timestep_number++;
       }
 } 
//@sect{The <code>main</code> function}
int main()

{
  static constexpr int dim = 2;
  GravWave<dim> gravwave_problem;

  gravwave_problem.run();

  return 0;
}
