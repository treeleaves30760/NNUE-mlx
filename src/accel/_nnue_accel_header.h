#ifndef _NNUE_ACCEL_HEADER_H
#define _NNUE_ACCEL_HEADER_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>

#ifdef USE_NEON
#include <arm_neon.h>
#endif

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#define MAX_INDICES 64        /* Max features per perspective (well above 30 for chess) */
#define DEFAULT_MAX_STACK 128

#endif /* _NNUE_ACCEL_HEADER_H */
