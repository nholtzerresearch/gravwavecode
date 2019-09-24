#ifndef HELPER_H
#define HELPER_H

#include <deal.II/base/function.h>

namespace
{
  template <int dim, unsigned int components, typename Callable>
  class ToFunction : public dealii::Function<dim>
  {
  public:
    ToFunction(const Callable &callable)
        : dealii::Function<dim>(components)
        , callable_(callable)
    {
    }

    virtual double value(const dealii::Point<dim> &point,
                         unsigned int component) const
    {
      return callable_(point, component);
    }

  private:
    const Callable callable_;
  };
} // namespace

template <int dim, unsigned int components, typename Callable>
ToFunction<dim, components, Callable> to_function(const Callable &callable)
{
  return {callable};
}

#endif /* HELPER_H */
