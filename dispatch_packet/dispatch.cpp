
#include <cstdint>


#define HSA_LARGE_MODEL 1

typedef uint32_t hsa_signal_t;

typedef struct hsa_kernel_dispatch_packet_s {
  uint16_t header ;
  uint16_t setup;
  uint16_t workgroup_size_x ;
  uint16_t workgroup_size_y ;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x ;
  uint32_t grid_size_y ;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object ;
#ifdef HSA_LARGE_MODEL
  void * kernarg_address;
#elif defined HSA_LITTLE_ENDIAN
  void * kernarg_address;
  uint32_t reserved1;
#else
  uint32_t reserved1;
  void * kernarg_address;
#endif
  uint64_t reserved2;
  hsa_signal_t completion_signal;
} hsa_kernel_dispatch_packet_t;



extern "C" int __hsail_get_num_groups(hsa_kernel_dispatch_packet_t* p, int i) {
  
  uint32_t num_groups = 0;
  uint32_t gs = 0;
  uint32_t ws = 1;
  switch(i) {
    case 0: 
      {
        gs = p->grid_size_x;
        ws = p->workgroup_size_x;
      }
    break;
    case 1:
      {
        gs = p->grid_size_y;
        ws = p->workgroup_size_y;
      }
    break;
    case 2:
      {
        gs = p->grid_size_z;
        ws = p->workgroup_size_z;
      }
    break;
    default:
      break;
  };

  num_groups = gs/ws;
  num_groups += (gs%ws)==0?0:1;
  return num_groups;
}
