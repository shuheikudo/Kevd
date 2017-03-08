#ifndef TIMED_H
#define TIMED_H

#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/time.h>
#include <time.h>

// for VS
#ifdef _MSC_VER
#include "common.h"
typedef clock_t Timed_t;
inline Timed_t* epoch(int p)
{
	static Timed_t start;
	if (p) start = clock();
	return &start;
}
inline void current_time(Timed_t* cur)
{
	*cur = clock();
}
inline double duration_in_sec(const Timed_t*lhs, const Timed_t*rhs)
{
	return (double)(*lhs - *rhs) / CLOCKS_PER_SEC;
}
inline int64_t duration_in_usec(const Timed_t* lhs, const Timed_t*rhs)
{
	return (int64_t)(duration_in_sec(lhs, rhs) * 1000000);
}
#else
typedef struct timespec Timed_t;
Timed_t* epoch(int p);
void current_time(Timed_t* cur);
double duration_in_sec(const Timed_t* lhs, const Timed_t* rhs);
int64_t duration_in_usec(const Timed_t*, const Timed_t* rhs);

#endif

// write log to @filename.
// the logs before calling log_file will be omitted.
// if @filename == 1ull, the logs will be dumped to stderr.
FILE* log_file(const char* filename);
void log_close(void);
void log_comment(const char* comment, ...);
#ifdef _MSC_VER
__declspec(noreturn)
#endif
void log_abort(const char* comment, ...);
void log_current_time(unsigned id, unsigned indent, const char* tag);

unsigned get_log_id(void);
unsigned inc_indent(void);
unsigned dec_indent(void);


#define MEASURE(TAG, FUN, ...)\
	do{\
		int my_id = get_log_id();\
		log_current_time(my_id, inc_indent(), TAG "_st");\
		FUN(__VA_ARGS__);\
		log_current_time(my_id, dec_indent(), TAG "_ed");\
				}while(0)

#define MEASUREI(INFO, TAG, FUN, ...)\
	do{\
		int my_id = get_log_id();\
		log_current_time(my_id, inc_indent(), TAG "_st");\
		INFO = FUN(__VA_ARGS__);\
		log_current_time(my_id, dec_indent(), TAG "_ed");\
			}while(0)

#endif
