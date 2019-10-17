
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

void check_hsa_error(hsa_status_t s) {
    if (s != HSA_STATUS_SUCCESS) {
        const char *error = nullptr;
        auto ss = hsa_status_string(s, &error);
        if (ss == HSA_STATUS_SUCCESS)
            std::cerr << "HSA Error: " << error << std::endl;
        else
            std::cerr << "HSA not initialized or invalid status error" << std::endl;
        abort();
    }
}

class hsa_env {
public:
    hsa_env() {
        check_hsa_error(hsa_init());
    }
    ~hsa_env() {
        for(auto& e : executables) {
            check_hsa_error(
                hsa_executable_destroy(e));
        }
        for(auto& r : code_object_readers) {
            check_hsa_error(
                hsa_code_object_reader_destroy(r));
        }
        check_hsa_error(hsa_shut_down());
    }

    void get_agents() {
        auto f = [](hsa_agent_t agent, void* data) {
            hsa_env* this_ptr = reinterpret_cast<hsa_env*>(data);
            hsa_device_type_t dtype;
            check_hsa_error(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dtype));
            switch (dtype) {
              case HSA_DEVICE_TYPE_GPU:
                this_ptr->gpu_agents.push_back(agent);
                break;
              case HSA_DEVICE_TYPE_CPU:
                this_ptr->host_agents.push_back(agent);
                break;
            };
            return HSA_STATUS_SUCCESS;
        };
        check_hsa_error(hsa_iterate_agents(f, this));
#if 1
        std::cout << "found " << gpu_agents.size() << " GPUs" << std::endl;
        std::cout << "found " << host_agents.size() << " Host Agents" << std::endl;
#endif
    }

    inline std::string symbol_name(hsa_executable_symbol_t x) {
        std::uint32_t sz = 0u;
        hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &sz);

        std::string r(sz, '\0');
        hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &r.front());

        return r;
    }

    hsa_executable_t load_code_object(hsa_agent_t agent, std::vector<char>& code_blob) {

        hsa_code_object_reader_t r = {};
        check_hsa_error(
        hsa_code_object_reader_create_from_memory(code_blob.data(), code_blob.size(), &r));

        hsa_executable_t exe = {};

        check_hsa_error(
         hsa_executable_create_alt(
                                HSA_PROFILE_FULL,
                                HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                nullptr,
                                &exe));     
        check_hsa_error(
          hsa_executable_load_agent_code_object(exe, agent, r, nullptr, nullptr));
        check_hsa_error(
          hsa_executable_freeze(exe, nullptr));

        auto get_kernel_symbols = [](hsa_executable_t, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data) {
            hsa_env* this_ptr = reinterpret_cast<hsa_env*>(data);
            hsa_symbol_kind_t r = {};
            hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &r);
            if (r == HSA_SYMBOL_KIND_KERNEL) {
#if 1
                std::cerr << "adding symbol: " << this_ptr->symbol_name(symbol) << std::endl;
#endif
                this_ptr->agent_symbols.push_back(symbol);
            }
            return HSA_STATUS_SUCCESS;
        };
        check_hsa_error(
            hsa_executable_iterate_agent_symbols(
                exe, agent, get_kernel_symbols, this));
        code_object_readers.push_back(r);
        executables.push_back(exe);

#if 1
        std::cout << "load_code_object: " << agent_symbols.size() << " kernel symbols" << std::endl;
#endif
    }

    std::vector<hsa_agent_t> host_agents;
    std::vector<hsa_agent_t> gpu_agents;

    std::vector<hsa_executable_symbol_t> agent_symbols;
    std::vector<hsa_executable_t> executables;
    std::vector<hsa_code_object_reader_t> code_object_readers;
};

bool read_file(std::string file, std::vector<char>& file_content) {
  std::ifstream s(file, std::ios::binary|std::ios::in);
  file_content.clear();
  if (!s.is_open()) {
    //td::cerr << "Can't open " << file << std::endl;
    return false;
  }

  s.seekg(0, s.end);
  file_content.resize(s.tellg());
  s.seekg(0, s.beg);
  s.read(file_content.data(), file_content.size());
  return true;
}

bool write_file(std::string file, std::vector<char>& file_content) {
  std::ofstream s(file, std::ios::binary|std::ios::out|std::ios::trunc);
  if (!s.is_open()) {
    std::cerr << "Can't open " << file << std::endl;
    return false;
  }
  s.write(file_content.data(), file_content.size());
  s.close();
  return true;
}

int main(int argc, char* argv[]) {

  if (argc != 2) {
      return -1;
  }

#if 0
  std::cout << "read " << file_content.size() << " bytes" << std::endl;
  std::cout << "file content: " << std::endl;
  std::cout << std::string(file_content.begin(), file_content.end()) << std::endl;
#endif

  hsa_env hsa;
  hsa.get_agents();

  std::string file_name = argv[1];
  std::vector<char> file_content;
  if (!read_file(file_name, file_content)) {
    std::cerr << "Error reading " << file_name << std::endl;
  }
    
  hsa.load_code_object(hsa.gpu_agents[0], file_content);

  write_file("./test_outputfile", file_content);

  return 0;
}