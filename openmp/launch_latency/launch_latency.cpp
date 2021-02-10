#include <chrono>
#include <iostream>
#include <utility>
#include <vector>

template<class chrono_clock_type>
class timer {
   typedef chrono_clock_type clock;
   typedef typename clock::time_point time_point;
   public:
        timer(uint32_t prealloc_timeslots) {
            intervals.reserve(prealloc_timeslots);
        }
        timer() : timer(__default_timeslots) { }
        void start() {
            intervals[next_start++].first = now();
        }
        void stop() {
            intervals[next_end++].second = now();
        }
        float get_avg_milliseconds() {
            float t = 0.0f;
            uint32_t i = 0;
            for (; i < next_end; ++i)
                t += std::chrono::duration<float, std::milli>(delta(intervals[i])).count();
            if (i > 0)
                t = t/(float)i;
            return t;
        }
    private:
        constexpr static uint32_t __default_timeslots = 1024;
        time_point now() { return clock::now(); }
        typedef std::pair<time_point, time_point> __interval;
        auto delta(const __interval &i) {
            return i.second - i.first;
        }
        std::vector<__interval> intervals;
        uint32_t next_start = 0;
        uint32_t next_end = 0;
};


static void launch_empty(const int n) {
    #pragma omp target teams distribute parallel for
    for(int i = 0; i < n; ++i) {
    }
}

int main() {
    // warm up
    launch_empty(1);

    constexpr int N = 1000;
    timer<std::chrono::high_resolution_clock> timer(N);
    for (int i = 0; i < N; ++i) {
        timer.start();
        launch_empty(1);
        timer.stop();
    }
    std::cout << "Average latency: " << timer.get_avg_milliseconds() * 1000.0f << " ms" << std::endl;
    return 0;
}