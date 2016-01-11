
#include <random>
#include <vector>
#include <iostream>
#include <hc.hpp>


template<typename T, typename S> 
void multiply(hc::array_view<T,1>& av_in1
             ,hc::array_view<T,1>& av_in2
             ,hc::array_view<T,1>& av_out) {
  
  hc::parallel_for_each(av_in1.get_extent(),[=](hc::index<1> i) [[hc]] {
    S x = (S)av_in1[i];
    S y = (S)av_in2[i];
    S z = x * y;
    av_out[i] = (T)z;
  });

}

template<typename T, typename S> 
void multiply_add(hc::array_view<T,1>& av_in1
             ,hc::array_view<T,1>& av_in2
             ,hc::array_view<T,1>& av_out) {
  
  hc::parallel_for_each(av_in1.get_extent(),[=](hc::index<1> i) [[hc]] {
    S x = (S)av_in1[i];
    S y = (S)av_in2[i];
    S z = (S)av_out[i];

    av_out[i] = (T)((S)(x*y+z));
  });

}


template<typename T, typename S>
void test_multiply(const unsigned int extent) {

  
  hc::array_view<T ,1> in1(extent);
  hc::array_view<T ,1> in2(extent);
  hc::array_view<T ,1> out(extent);

  std::vector<T> cpu(extent);

  std::default_random_engine generator;
  std::uniform_int_distribution<T> distribution((T)0x00000000, (T)0xFFFFFFFF);

  for (unsigned int i = 0; i < extent; i++) {
    in1[i] = distribution(generator);
    in2[i] = distribution(generator);
    out[i] = distribution(generator);
    cpu[i] = (T)((S) ((S)in1[i] * (S)in2[i]));
  }
  
  multiply<T,S>(in1, in2, out);

  int error = 0;
  for (unsigned int i = 0; i < extent; i++) {
    if (cpu[i] != out[i])
      error++;
  }
  
  std::cout << __FUNCTION__ << ": " << error << " errors\n" << std::endl;
}



template<typename T, typename S>
void test_multiply_add(const unsigned int extent) {

  
  hc::array_view<T ,1> in1(extent);
  hc::array_view<T ,1> in2(extent);
  hc::array_view<T ,1> out(extent);

  std::vector<T> cpu(extent);

  std::default_random_engine generator;
  std::uniform_int_distribution<T> distribution((T)0x00000000, (T)0xFFFFFFFF);

  for (unsigned int i = 0; i < extent; i++) {
    in1[i] = distribution(generator);
    in2[i] = distribution(generator);
    out[i] = distribution(generator);
    cpu[i] = (T)((S) ((S)in1[i] * (S)in2[i] + (S)out[i]));
  }
  
  multiply_add<T,S>(in1, in2, out);

  int error = 0;
  for (unsigned int i = 0; i < extent; i++) {
    if (cpu[i] != out[i])
      error++;
  }
  
  std::cout << __FUNCTION__ << ": " << error << " errors\n" << std::endl;
}


int main(int argc, char* argv[]) {

  const int n = 1024;

  test_multiply<int,int>(n);
  test_multiply<int,short>(n);

  test_multiply<unsigned int, unsigned int>(n);
  test_multiply<unsigned int, unsigned short>(n);



  test_multiply_add<int,int>(n);
  test_multiply_add<int,short>(n);

  test_multiply_add<unsigned int, unsigned int>(n);
  test_multiply_add<unsigned int, unsigned short>(n);





  return 0;
}



