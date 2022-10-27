#ifndef CLOCK_H
#define CLOCK_H
#include <Windows.h>
#include <intrin.h>
typedef unsigned __int64 prof_time_t;

inline void start_measure(prof_time_t & cycle_read)
{
	SetProcessAffinityMask(GetCurrentProcess(), 1);
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	int a[4], b=1;
	__cpuid(a, b);
	cycle_read = __rdtsc();
}

void inline end_measure(prof_time_t& cycle_read)
{
	unsigned int c;
	cycle_read = __rdtscp(&c);
	int a[4], b=1;
	__cpuid(a, b);		
};

#endif
