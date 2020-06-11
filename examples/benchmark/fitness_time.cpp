/*
Copyright (c) 2018, Jairo Rojas-Delgado <jrdelgado@uci.cu>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>
#include <chrono>

#include <common.h>
#include <dnn_opt.h>

using namespace std;
using namespace std::chrono;

#ifdef ENABLE_CUDA
using namespace dnn_opt::cuda;
#elif ENABLE_COPT
using namespace dnn_opt::copt;
#elif ENABLE_CORE
using namespace dnn_opt::core;
#endif

int main(int argc, char** argv)
{
  /* command line argument collection */

  int n = input("-n", 100, argc, argv);
  int p = input("-p", 40, argc, argv);
  int eta = input("-eta", 1000, argc, argv);
  int solution_type = input("-s", 8, argc, argv);

  auto* generator = generators::uniform::make(-10.0f, 10.0f);
  auto* solution = create_solution(solution_type, n, generator);
  
  solution->generate();

  auto start = high_resolution_clock::now();
  
  for(int i = 0; i < eta; i++)
  {
    solution->fitness();
    solution->set_modified(true);
  }
  
  auto end = high_resolution_clock::now();
  float time = duration_cast<milliseconds>(end - start).count();

  cout << time << endl;

  delete solution;
  delete generator;

  return 0;
}
