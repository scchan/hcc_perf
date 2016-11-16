
#ifndef _SIMPLETIMER_H
#define _SIMPLETIMER_H

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <linux/limits.h>
#include <unistd.h>
#endif

#include <string>
#include <iostream>

class SimpleTimer {

public:
    SimpleTimer(const char* n) {
      name = std::string(n);
#ifdef _WIN32
      startTime = GetTickCount64();
#else
      struct timeval s;
      gettimeofday(&s, 0);
      startTime = (long long)s.tv_sec * (long long)1.0E3 +
                  (long long)s.tv_usec / (long long)1.0E3;
#endif
    }

    ~SimpleTimer() {
      long long total;
#ifdef _WIN32
      total = GetTickCount64() - startTime;
#else
      struct timeval s;
      gettimeofday(&s, 0);
      total  = (long long)s.tv_sec * (long long)1.0E3 +
               (long long)s.tv_usec / (long long)1.0E3;
      total -= startTime; 
#endif
      std::cout << "Timer " << name << ": " << total << " ms" << std::endl;
    }

private:
    std::string name;
    long long startTime;
};

#endif   // #endif _SIMPLETIMER_H
