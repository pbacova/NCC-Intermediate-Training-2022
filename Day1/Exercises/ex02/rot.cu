#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>

/***
 * coords structure
 ***/
typedef struct {
  float x;
  float y;
} coords;

/***
 * Print usage
 ***/
void
usage(char *argv[])
{
  fprintf(stderr, "usage: %s N\n", argv[0]);
  return;
}

/***
 * Allocate memory; print error if NULL is returned
 ***/
void *
ualloc(size_t size)
{
  void *ptr = malloc(size);
  if(ptr == NULL) {
    fprintf(stderr, "malloc() returned null; quitting...\n");
    exit(-2);
  }
  return ptr;
}

/***
 * Return seconds elapsed since t0, with t0 = 0 the epoch
 ***/
double
stop_watch(double t0)
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec/1e6 - t0;
}

/***
 * Read coords from binary file
 ***/
void
read_coords(coords *r, unsigned long int n, const char *fname)
{
  FILE *fp = fopen(fname, "r");
  for(int i=0; i<n; i++) {
    fread(&r[i].x, sizeof(float), 2, fp);
  }
  fclose(fp);
  return;
}

/***
 * Write coords to binary file
 ***/
void
write_coords(const char *fname, coords *r, unsigned long int n)
{
  FILE *fp = fopen(fname, "w");
  for(int i=0; i<n; i++) {
    fwrite(&r[i].x, sizeof(float), 2, fp);
  }
  fclose(fp);
  return;
}

/***
 * Do r' <- U*r + s on the CPU using OpenMP
 ***/
void
rotate(int n, coords *out, float theta, coords *r, coords *s)
{
  float ct = cos(theta);
  float st = sin(theta);
#pragma omp parallel for
  for(int i=0; i<n; i++) {
    out[i].x = ct*r[i].x - st*r[i].y + s[i].x;
    out[i].y = st*r[i].x + ct*r[i].y + s[i].y;
  }
  return;
}

int
main(int argc, char *argv[])
{
  /*
   * If number of arguments are not as expected, print usage and exit
   */
  if(argc != 2) {
    usage(argv);
    return 1;
  }

  unsigned long int n = atol(argv[1]);

  coords *r = (coords *)ualloc(sizeof(coords)*n);
  coords *s = (coords *)ualloc(sizeof(coords)*n);
  coords *v0 = (coords *)ualloc(sizeof(coords)*n);
  coords *v1 = (coords *)ualloc(sizeof(coords)*n);

  /*
   * Read from file
   */
  read_coords(r, n, "points.bin");
  read_coords(s, n, "shifts.bin");
    
  /*
   * The angle
   */
  float theta = (M_PI)*(10.0/180.0);
    
  /*
   * A: Run rotate(), return to v0, report performance
   */
  {
    double t0 = stop_watch(0);
    rotate(n, v0, theta, r, s);
    t0 = stop_watch(t0);

    double n_flop = 8;
    double n_io = 6*sizeof(float);
#pragma omp parallel
    {
#pragma omp single
      {
	int nthr = omp_get_num_threads();
	printf(" CPU: nthr = %4d   t0 = %6.4lf sec   P = %7.3lf Gflop/s   B = %7.3lf GB/s\n",
	       nthr, t0, n_flop*n/1e9/t0, n_io*n/1e9/t0);
      }
    }
  }

  /*
   * B: Run rotate(), return to v1, report performance
   */
  {
    double t0 = stop_watch(0);
    rotate(n, v1, theta, r, s);
    t0 = stop_watch(t0);

    double n_flop = 8;
    double n_io = 6*sizeof(float);
#pragma omp parallel
    {
#pragma omp single
      {
	int nthr = omp_get_num_threads();
	printf(" CPU: nthr = %4d   t0 = %6.4lf sec   P = %7.3lf Gflop/s   B = %7.3lf GB/s\n",
	       nthr, t0, n_flop*n/1e9/t0, n_io*n/1e9/t0);
      }
    }
  }

  /* Compare v1 and v0 */
  double diff = 0;
  double norm = 0;
  for(int i=0; i<n; i++) {
    float dx = v0[i].x-v1[i].x;
    float dy = v0[i].y-v1[i].y;
    diff += dx*dx + dy*dy;
    norm += v0[i].x*v0[i].x;
    norm += v0[i].y*v0[i].y;
  }
  printf(" Diff = %e\n", diff/norm);

  /*
   * Write to file
   */
  write_coords("points-rot.bin", v0, n);


  
  /*
   * Free arrays
   */
  free(r);
  free(s);
  free(v0);
  free(v1);
  return 0;
}
