// Copyright by Shuhei Kudo, May 2015.
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

#include "timed.h"

#ifndef _MSC_VER
Timed_t* epoch(int p)
{
	static Timed_t start;
	if (p) gettimeofday(&start, NULL);
	return &start;
}
void current_time(Timed_t* cur)
{
	gettimeofday(cur, NULL);
}
double duration_in_sec(const Timed_t* lhs, const Timed_t* rhs)
{
	Timed_t res;
	timersub(lhs, rhs, &res);
	return (double)res.tv_sec + (double)res.tv_usec / 1000000;
}
int64_t duration_in_usec(const Timed_t* lhs, const Timed_t* rhs)
{
	Timed_t res;
	timersub(lhs, rhs, &res);
	return ((int64_t)res.tv_sec) * 1000000LL + res.tv_usec;
}
#endif

FILE* log_file(const char* filename)
{
	static FILE* fp = 0;
	if (!filename) return fp;
	else if (filename == (char*)(intptr_t)1) {
		fp = stderr;
		return fp;
	}
	else if(!fp) {
		epoch(0);
#ifdef _MSC_VER
		if (fopen_s(&fp, filename, "w")) abort();
#else
		if((fp = fopen(filename, "w")) == NULL) abort();
#endif
	}
	return fp;
}

void log_close()
{
	FILE* fp = log_file(0);
	if (fp) {
		fflush(fp);
		fclose(fp);
	}
}

void log_comment(const char* comment, ...)
{
	va_list arg;
	va_start(arg, comment);
	FILE* fp = log_file(0);
	if (fp) {
		fprintf(fp, "# %d ", get_log_id());
		vfprintf(fp, comment, arg);
		fputs("\n", fp);
	}
}

void log_abort(const char* comment, ...)
{
	va_list arg;
	va_start(arg, comment);
	FILE* fp = log_file(0);
	if (fp){
		fprintf(fp, "# %d ABORT ", get_log_id());
		vfprintf(fp, comment, arg);
		fputs("\n", fp);
	}
	log_close();
	abort();
}

void log_current_time(unsigned id, unsigned indent, const char* tag)
{
	Timed_t t;
	FILE* fp = log_file(0);
	current_time(&t);
	if (fp) fprintf(fp, "%d, %d, %s, %lld\n", id, indent, tag, duration_in_usec(&t, epoch(0)));
}


static unsigned log_id = 0u;
static unsigned log_indent = 0u;

unsigned get_log_id(void)
{
	return log_id++;
}

unsigned inc_indent(void)
{
	return log_indent++;
}
unsigned dec_indent(void)
{
	return --log_indent;
}