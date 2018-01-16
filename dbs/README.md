# Multialternative Decision by Sampling

This is an implementation of the DbS in C++ with R interface.

- Noguchi, T., & Stewart, N. (in press). Multialternative decision by sampling:
  A model of decision making constrained by process data. *Psychological Review*.

## Description

- dbs.hpp and dbs.cpp: main model implementation
- test.hpp and test.cpp: test for the model implementation
- libdbs_r.cpp: R interface
- script.R: example with the R interface
- walk\_through.py: The walk through calculation as used in Appendix C in the above paper. This script depends only on numpy, and does not depend on other files in this repo.

## Dependencies

- C++11 standard libraries
- Eigen (http://eigen.tuxfamily.org/)
- Rcpp (http://rcpp.org/) for R interface


## Licence

Copyright (C) 2015-2018 Takao Noguchi (tkngch@runbox.com)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
