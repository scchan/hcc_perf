
extern "C" [[hc]] __attribute__((constant)) int __oclc_ISA_version;

[[hc]] static inline int get_isa_version() {
  return __oclc_ISA_version;
}

[[hc]] static inline bool is_gfx900() {
  if (__oclc_ISA_version == 900)
    return true;
  else
    return false;
}

[[hc]] static inline bool is_gfx906() {
  if (__oclc_ISA_version == 906)
    return true;
  else
    return false;
}

