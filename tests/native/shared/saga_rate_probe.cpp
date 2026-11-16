// Standalone scalar-policy driver; pytest supplies independent expected values.
#include "common/classifier/saga_rate.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
  long n, b, m;
  double data, maximum, ridge;
  while (std::cin >> n >> b >> m >> data >> maximum >> ridge) {
    try {
      const auto result = ltm::saga::calibration::rate(n, b, m, data, maximum, ridge);
      std::cout << std::setprecision(17) << result.rate << ' ' << result.practical << ' '
                << result.batch << '\n';
    } catch (const std::invalid_argument&) {
      std::cout << "invalid\n";
    }
  }
}
