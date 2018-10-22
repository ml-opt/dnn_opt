#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <core/base/sampler.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{

sampler* sampler::make(reader* reader, float sample_proportion)
{
  return new sampler(reader, sample_proportion);
}

sampler** sampler::make(reader* reader, int folds)
{
  sampler** samplers = new sampler*[folds];
  float* random = new float[reader->size()];
  auto* generator = generators::uniform::make(0, 1);
  float proportion = 1.0f / folds;

  generator->generate(reader->size(), random);

  for(int i = 0; i < folds; i++)
  {
    bool* mask = new bool[reader->size()];

    for(int j = 0; j < reader->size(); j++)
    {
      mask[j] = random[j] >= i * proportion && random[j] < (i + 1) * proportion;
    }

    samplers[i] = new sampler(reader, mask);
  }

  delete[] random;
  delete generator;

  return samplers;
}

float* const sampler::in_data()
{
  return _in_data;
}

float* const sampler::out_data()
{
  return _out_data;
}

int sampler::get_in_dim() const
{
  return _reader->get_in_dim();
}

int sampler::get_out_dim() const
{
  return _reader->get_out_dim();
}

int sampler::size() const
{
  return _samples;
}

sampler* sampler::difference(sampler* other)
{
  assert(_reader == other->_reader);

  bool* mask = new bool[size()];

  for(int i = 0; i < size(); i++)
  {
    mask[i] = _mask[i] && !other->_mask[i];
  }

  return new sampler(_reader, mask);
}

void sampler::save_to_file(std::string file_name)
{//FIX THIS
  std::ofstream file(file_name, std::ofstream::out);

  file << size() << " " << get_in_dim() << " " << get_out_dim() << std::endl;

  for(int i = 0; i < size(); i++)
  {
    for (int j = 0; j < get_in_dim(); j++)
    {
      file << _in_data[i * get_in_dim() + j] << " ";
    }
    for (int j = 0; j < get_out_dim(); j++)
    {
      file << _out_data[i * get_out_dim() + j] << " ";
    }
    file << std::endl;
  }


  file.close();
}

void sampler::sample()
{
  for(int i = 0; i < get_in_dim(); i++)
  {
    int sampled = 0;
    int col_start = i * _reader->size();

    for(int j = 0; j < _reader->size(); j++)
    {
      if(_mask[j] == true)
      {        
        _in_data[col_start + sampled] = _reader->in_data()[col_start + j];
        sampled += 1;
      }
    }
  }

  for(int i = 0; i < get_out_dim(); i++)
  {
    int sampled = 0;
    int col_start = i * _reader->size();

    for(int j = 0; j < _reader->size(); j++)
    {
      if(_mask[j] == true)
      {
        _out_data[col_start + sampled] = _reader->out_data()[col_start + j];
        sampled += 1;
      }
    }
  }
}

sampler::sampler(reader* reader, float sample_proportion)
{
  float* random = new float[reader->size()];
  auto* generator = generators::uniform::make(0, 1);

  generator->generate(reader->size(), random);

  _samples = 0;
  _mask = new bool[reader->size()];

  for(int i = 0; i < reader->size(); i++)
  {
    _mask[i] = random[i] <= sample_proportion;
    _samples += _mask[i] == true ? 1 : 0;
  }

  _reader = reader;
  _in_data = new float[size() * get_in_dim()];
  _out_data = new float[size() * get_out_dim()];

  delete[] random;
  delete generator;
}

sampler::sampler(reader* reader, bool* mask)
{
  _samples = 0;
  _mask = mask;

  for(int i = 0; i < reader->size(); i++)
  {
    _samples += _mask[i] == true ? 1 : 0;
  }

  _reader = reader;
  _in_data = new float[size() * get_in_dim()];
  _out_data = new float[size() * get_out_dim()];
}

sampler::~sampler()
{
  delete[] _mask;
  delete[] _in_data;
  delete[] _out_data;

  _mask = 0;
  _in_data = 0;
  _out_data = 0;
}

}
}
