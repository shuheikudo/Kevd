#--------------------------------------------
#For FX-10
#--------------------------------------------
FC		= mpifrtpx
CC		= mpifccpx
VFC		= vtfrtpx
VCC		= vtfccpx
CCFLAG	= -Kfast,openmp,ocl -Xg

BLAS	= -SSL2BLAMP

LD		= $(FC)
LDFLAG  = $(CCFLAG) $(BLAS) -mlcmain=main

MAIN	= test
LOAD	= test

OBJS = common.o dormtrb.o dsyevdt.o dsytdc.o r24mv.o timed.o dlaed3x.o

.PHONY : clean
.PRECIOUS : $(LOAD)
.SUFFIXES : .o .c .f .f90


all : $(LOAD)

lib : $(OBJS)
	rm -f libdsyevd.a
	ar r libdsyevd.a $(OBJS)

clean :
	rm -f *.o *.out

$(LOAD) : $(OBJS) test.o
	rm -f $(LOAD)
	$(LD) -o $(LOAD) $(OBJS) $(LDFLAG) $(DEF)
	@echo "$(LOAD) is now up-to-date"


.c.o : 
	$(CC) $(CCFLAG) -c $< $(DEF)

.f.o : 
	$(FC) $(CCFLAG) -c $< $(DEF)

.f90.o : 
	$(FC) $(CCFLAG) -c $< $(DEF)

