//
// Utility functions for timing and performance monitoring
//

#ifndef IDMRG_UTIL_TIMER_H
#define IDMRG_UTIL_TIMER_H

#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

namespace idmrg {

//
// Simple timer class for profiling
//
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;
    
    Timer() : running_(false), elapsed_(0) {}
    
    void start() {
        if (!running_) {
            start_ = Clock::now();
            running_ = true;
        }
    }
    
    void stop() {
        if (running_) {
            auto end = Clock::now();
            elapsed_ += Duration(end - start_).count();
            running_ = false;
        }
    }
    
    void reset() {
        elapsed_ = 0;
        running_ = false;
    }
    
    double elapsed() const {
        if (running_) {
            auto now = Clock::now();
            return elapsed_ + Duration(now - start_).count();
        }
        return elapsed_;
    }
    
    // Return elapsed time in formatted string
    std::string elapsedStr() const {
        double t = elapsed();
        if (t < 1.0) {
            return std::to_string(static_cast<int>(t * 1000)) + " ms";
        } else if (t < 60.0) {
            return std::to_string(t) + " s";
        } else if (t < 3600.0) {
            int min = static_cast<int>(t / 60);
            double sec = t - min * 60;
            return std::to_string(min) + " min " + 
                   std::to_string(static_cast<int>(sec)) + " s";
        } else {
            int hr = static_cast<int>(t / 3600);
            int min = static_cast<int>((t - hr * 3600) / 60);
            return std::to_string(hr) + " hr " + std::to_string(min) + " min";
        }
    }

private:
    bool running_;
    TimePoint start_;
    double elapsed_;
};

//
// Scoped timer that automatically stops when destroyed
//
class ScopedTimer {
public:
    ScopedTimer(Timer& timer) : timer_(timer) {
        timer_.start();
    }
    
    ~ScopedTimer() {
        timer_.stop();
    }

private:
    Timer& timer_;
};

//
// Global timer registry for named timers
//
class TimerRegistry {
public:
    static TimerRegistry& instance() {
        static TimerRegistry reg;
        return reg;
    }
    
    Timer& get(const std::string& name) {
        return timers_[name];
    }
    
    void start(const std::string& name) {
        timers_[name].start();
    }
    
    void stop(const std::string& name) {
        timers_[name].stop();
    }
    
    void reset(const std::string& name) {
        timers_[name].reset();
    }
    
    void resetAll() {
        for (auto& [name, timer] : timers_) {
            timer.reset();
        }
    }
    
    void report(std::ostream& os = std::cout) const {
        os << "\n===== Timer Report =====\n";
        for (const auto& [name, timer] : timers_) {
            os << std::setw(30) << std::left << name << ": "
               << timer.elapsedStr() << "\n";
        }
        os << "========================\n";
    }

private:
    std::map<std::string, Timer> timers_;
    TimerRegistry() = default;
};

// Convenience macros
#define IDMRG_TIME_START(name) idmrg::TimerRegistry::instance().start(name)
#define IDMRG_TIME_STOP(name) idmrg::TimerRegistry::instance().stop(name)
#define IDMRG_TIME_REPORT() idmrg::TimerRegistry::instance().report()

//
// Memory usage tracking (Linux only)
//
inline size_t getCurrentRSS() {
#ifdef __linux__
    long rss = 0L;
    FILE* fp = nullptr;
    if ((fp = fopen("/proc/self/statm", "r")) == nullptr) {
        return 0;
    }
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return static_cast<size_t>(rss) * static_cast<size_t>(sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

inline std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

} // namespace idmrg

#endif // IDMRG_UTIL_TIMER_H
