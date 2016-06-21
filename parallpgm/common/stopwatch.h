#include <stdio.h>
#include <inttypes.h>
#include <sys/time.h>

static struct timeval beg_time, end_time;

struct timeval *stopwatch_restart()
{
    gettimeofday(&beg_time, NULL);
    return &beg_time;
}

uint64_t stopwatch_record()
{
    gettimeofday(&end_time, NULL);
    time_t delta_s = end_time.tv_sec - beg_time.tv_sec;
    time_t delta_us = end_time.tv_usec - beg_time.tv_usec;
    return delta_s * 1000000L + delta_us;
}

