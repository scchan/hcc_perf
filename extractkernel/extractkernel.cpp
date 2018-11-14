
#include <unistd.h>
#include <iostream>
#include <unordered_map>

#include "elfio/elfio.hpp"
#include "code_object_bundle/code_object_bundle.hpp"

using namespace ELFIO;
using namespace std;
using namespace code_object_bundle;

template <typename P>
inline section* find_section_if(elfio& reader, P p) {
    const auto it = find_if(reader.sections.begin(), reader.sections.end(), move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

int main(int argc, char* argv[]) {

  char* input_file = nullptr;
  int s; 
  while ((s = getopt(argc, argv, "i:"))!=-1) {
    switch (s) {
      case 'i':
        input_file = optarg;
        break;
      case '?':
        abort();
      default:
        abort();
    }
  }
  if (!input_file) {
    cerr << "No input file specified" << endl;
    exit(1);
  }

  elfio reader;
  if (!reader.load(input_file)) {
    cerr << "Error reading " << input_file << endl;
    abort();
  }

  constexpr const char kernel_section_name[] = ".kernel";
  constexpr const char kernel_ir_section_name[] = ".kernel_ir";
  
  auto kernel_section = find_section_if(reader, 
                                        [](const section* x) { return x->get_name() == kernel_section_name; });
  
  if (kernel_section) {
    vector<char> kernel_blobs;
    kernel_blobs.insert(kernel_blobs.end(), kernel_section->get_data(), 
                          kernel_section->get_data() + kernel_section->get_size());

    std::unordered_map<std::string, std::vector<std::vector<char>>> classified_blobs;

    for (auto sub_blob = kernel_blobs.begin(); sub_blob != kernel_blobs.end(); ++sub_blob) {
      Bundled_code_header tmp(sub_blob, kernel_blobs.end());
      if (valid(tmp)) {
        for (auto&& bundle : bundles(tmp)) {
          classified_blobs[bundle.triple].push_back(bundle.blob);
        }
      }
    }

    // dump all the bundles
    for (auto b = classified_blobs.begin(); b != classified_blobs.end(); ++b) {
      std::cout << "arch: " << b->first << std::endl;
    }
  }
  else {
    cout << "No " << kernel_section_name << " found." << endl;
  }

  return 0;
}
