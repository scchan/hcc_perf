

#ifndef _READFILE_HPP
#define _READFILE_HPP

#include <string>
#include <iostream>
#include <fstream>

// read a CL file into a string
static std::string readFile(const std::string& filename) {
  std::ifstream file;
  file.exceptions(std::ifstream::failbit|std::ifstream::badbit);
  file.open(filename.c_str());
  std::string s = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  file.close();
  return s;
}

#endif  /* #ifndef _READFILE_HPP */
