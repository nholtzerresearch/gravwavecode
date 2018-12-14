ass InitialValues : public Function<dim>
  {
  public:
    InitialValues (const unsigned int n_components = 1,
                   const double time = 0.)
      :
      Function<dim>(n_components, time)
    {}

    virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;
  };

  template <int dim>
  double InitialValues<dim>::value (const Point<dim> &p,
                                    const unsigned int  component/)// const
  {
    const double a = 1.0;
    const double mu = 1.0;
    const double sigma = 1.0;
    double ic;
    ic = a.*std::exp(-((p[0]-mu)*(p[0]-mu))/(2.*sigma*sigma));
    return ic;
  }
  //@sect{Boundary Conditions and RHS}
   template<int dim>
   class RightHandSide : public Function<dim>
   {
    public: 
    RightHandSide (const unsigned int n_components =1, 
                   const double time = 0.)
    :
    Function<dim>(n_components, time) 
    {}
     virtual double value(const Point<dim> &p,
                          const unsigned int component = 0) const;
     };

  //template<int dim>
 // class BoundaryValues : public Function<dim>
 // {
// public:
  // BoundaryValues () : Function<dim>() {}
  // virtual double value(const Point<dim> &p,
   //                     const unsigned int component = 0) const;
 // };

 template<int dim>
 double RightHandSide<dim>:: value (const Point<dim> &p,
                                    const unsigned int component) const 
    {
       double rhs;
       const double lambda= 0.0;//EDIT value later
       rhs = lambda*p[0]; //EDIT this function
      return rhs;
     }


//template<int dim>
  // double BoundaryValues<dim>:: value (const Point<dim> &p,
   //                     const unsigned int component)// const;
 // {
   // bnd_value=//EDIT this function
   // return bnd_value;
 // };









//template <int dim>
 // GravWave<dim>::define_boundary_conds(){
//  const unsigned int totalNodes=dof_handler.n_dofs();//total number of nodes

 // for(unsigned int globalNode=0; globalNode<totalNodes; globalNode++){
   // if(nodeLocation[globalNode] ==0){
     // boundary_values[globalNode]=c1;
  //  }
  //  if(nodeLocation[globalNode] ==L){
    //  boundary_values[globalNode] = c2;
   // }
//}
//}
/*


