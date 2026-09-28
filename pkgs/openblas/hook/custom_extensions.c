// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef __APPLE__
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <ctype.h>

#define LAPACK_DISABLE_NAN_CHECK
#define LAPACK_COMPLEX_STRUCTURE
#define LAPACK_FORTRAN_STRLEN_END
#define FORTRAN_STRLEN size_t
#define LAPACK_GLOBAL(lcname,UCNAME) lcname##_

typedef int32_t lapack_int;
typedef int32_t lapack_logical;
typedef struct { float real, imag; } _lapack_complex_float;
typedef struct { double real, imag; } _lapack_complex_double;
#define lapack_complex_float  _lapack_complex_float
#define lapack_complex_double _lapack_complex_double

typedef lapack_logical (*LAPACK_S_SELECT2)( const float*, const float* );
typedef lapack_logical (*LAPACK_S_SELECT3)( const float*, const float*, const float* );
typedef lapack_logical (*LAPACK_D_SELECT2)( const double*, const double* );
typedef lapack_logical (*LAPACK_D_SELECT3)( const double*, const double*, const double* );
typedef lapack_logical (*LAPACK_C_SELECT1)( const lapack_complex_float* );
typedef lapack_logical (*LAPACK_C_SELECT2)( const lapack_complex_float*, const lapack_complex_float* );
typedef lapack_logical (*LAPACK_Z_SELECT1)( const lapack_complex_double* );
typedef lapack_logical (*LAPACK_Z_SELECT2)( const lapack_complex_double*, const lapack_complex_double* );

#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102
#define LAPACK_WORK_MEMORY_ERROR       -1010
#define LAPACK_TRANSPOSE_MEMORY_ERROR  -1011

#define LAPACK_C2INT( x ) (lapack_int)(*((float*)&x ))
#define LAPACK_Z2INT( x ) (lapack_int)(*((double*)&x ))

#ifndef ABS
#define ABS(x) (((x) < 0) ? -(x) : (x))
#endif
#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif
#ifndef MAX3
#define MAX3(x,y,z) (((x) > MAX(y,z)) ? (x) : MAX(y,z))
#endif
#ifndef MIN3
#define MIN3(x,y,z) (((x) < MIN(y,z)) ? (x) : MIN(y,z))
#endif

#define LAPACKE_malloc( size ) malloc( size )
#define LAPACKE_free( p )      free( p )

void LAPACKE_xerbla( const char *name, lapack_int info ) {
    (void)name;
    (void)info;
}

lapack_logical LAPACKE_lsame( char ca, char cb ) {
    return (tolower((unsigned char)ca) == tolower((unsigned char)cb));
}


#define LAPACK_sgetrf LAPACK_GLOBAL(sgetrf,SGETRF)
lapack_int LAPACK_sgetrf(
    lapack_int const* m, lapack_int const* n,
    float* A, lapack_int const* lda, lapack_int* ipiv,
    lapack_int* info );

#define LAPACK_dgetrf LAPACK_GLOBAL(dgetrf,DGETRF)
lapack_int LAPACK_dgetrf(
    lapack_int const* m, lapack_int const* n,
    double* A, lapack_int const* lda, lapack_int* ipiv,
    lapack_int* info );

#define LAPACK_cgetrf LAPACK_GLOBAL(cgetrf,CGETRF)
lapack_int LAPACK_cgetrf(
    lapack_int const* m, lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda, lapack_int* ipiv,
    lapack_int* info );

#define LAPACK_zgetrf LAPACK_GLOBAL(zgetrf,ZGETRF)
lapack_int LAPACK_zgetrf(
    lapack_int const* m, lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda, lapack_int* ipiv,
    lapack_int* info );

#define LAPACK_sgetri LAPACK_GLOBAL(sgetri,SGETRI)
void LAPACK_sgetri(
    lapack_int const* n,
    float* A, lapack_int const* lda, lapack_int const* ipiv,
    float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_dgetri LAPACK_GLOBAL(dgetri,DGETRI)
void LAPACK_dgetri(
    lapack_int const* n,
    double* A, lapack_int const* lda, lapack_int const* ipiv,
    double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_cgetri LAPACK_GLOBAL(cgetri,CGETRI)
void LAPACK_cgetri(
    lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda, lapack_int const* ipiv,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_zgetri LAPACK_GLOBAL(zgetri,ZGETRI)
void LAPACK_zgetri(
    lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda, lapack_int const* ipiv,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_sgesv LAPACK_GLOBAL(sgesv,SGESV)
lapack_int LAPACK_sgesv(
    lapack_int const* n, lapack_int const* nrhs,
    float* A, lapack_int const* lda, lapack_int* ipiv,
    float* B, lapack_int const* ldb,
    lapack_int* info );

#define LAPACK_dgesv LAPACK_GLOBAL(dgesv,DGESV)
lapack_int LAPACK_dgesv(
    lapack_int const* n, lapack_int const* nrhs,
    double* A, lapack_int const* lda, lapack_int* ipiv,
    double* B, lapack_int const* ldb,
    lapack_int* info );

#define LAPACK_cgesv LAPACK_GLOBAL(cgesv,CGESV)
lapack_int LAPACK_cgesv(
    lapack_int const* n, lapack_int const* nrhs,
    lapack_complex_float* A, lapack_int const* lda, lapack_int* ipiv,
    lapack_complex_float* B, lapack_int const* ldb,
    lapack_int* info );

#define LAPACK_zgesv LAPACK_GLOBAL(zgesv,ZGESV)
lapack_int LAPACK_zgesv(
    lapack_int const* n, lapack_int const* nrhs,
    lapack_complex_double* A, lapack_int const* lda, lapack_int* ipiv,
    lapack_complex_double* B, lapack_int const* ldb,
    lapack_int* info );

#define LAPACK_sgeev_base LAPACK_GLOBAL(sgeev,SGEEV)
void LAPACK_sgeev_base(
    char const* jobvl, char const* jobvr,
    lapack_int const* n,
    float* A, lapack_int const* lda,
    float* WR,
    float* WI,
    float* VL, lapack_int const* ldvl,
    float* VR, lapack_int const* ldvr,
    float* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_sgeev(...) LAPACK_sgeev_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_sgeev(...) LAPACK_sgeev_base(__VA_ARGS__)
#endif

#define LAPACK_dgeev_base LAPACK_GLOBAL(dgeev,DGEEV)
void LAPACK_dgeev_base(
    char const* jobvl, char const* jobvr,
    lapack_int const* n,
    double* A, lapack_int const* lda,
    double* WR,
    double* WI,
    double* VL, lapack_int const* ldvl,
    double* VR, lapack_int const* ldvr,
    double* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_dgeev(...) LAPACK_dgeev_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_dgeev(...) LAPACK_dgeev_base(__VA_ARGS__)
#endif

#define LAPACK_cgeev_base LAPACK_GLOBAL(cgeev,CGEEV)
void LAPACK_cgeev_base(
    char const* jobvl, char const* jobvr,
    lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* W,
    lapack_complex_float* VL, lapack_int const* ldvl,
    lapack_complex_float* VR, lapack_int const* ldvr,
    lapack_complex_float* work, lapack_int const* lwork,
    float* rwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_cgeev(...) LAPACK_cgeev_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_cgeev(...) LAPACK_cgeev_base(__VA_ARGS__)
#endif

#define LAPACK_zgeev_base LAPACK_GLOBAL(zgeev,ZGEEV)
void LAPACK_zgeev_base(
    char const* jobvl, char const* jobvr,
    lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* W,
    lapack_complex_double* VL, lapack_int const* ldvl,
    lapack_complex_double* VR, lapack_int const* ldvr,
    lapack_complex_double* work, lapack_int const* lwork,
    double* rwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_zgeev(...) LAPACK_zgeev_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_zgeev(...) LAPACK_zgeev_base(__VA_ARGS__)
#endif

#define LAPACK_spotrf_base LAPACK_GLOBAL(spotrf,SPOTRF)
lapack_int LAPACK_spotrf_base(
    char const* uplo,
    lapack_int const* n,
    float* A, lapack_int const* lda,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_spotrf(...) LAPACK_spotrf_base(__VA_ARGS__, 1)
#else
    #define LAPACK_spotrf(...) LAPACK_spotrf_base(__VA_ARGS__)
#endif

#define LAPACK_dpotrf_base LAPACK_GLOBAL(dpotrf,DPOTRF)
lapack_int LAPACK_dpotrf_base(
    char const* uplo,
    lapack_int const* n,
    double* A, lapack_int const* lda,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_dpotrf(...) LAPACK_dpotrf_base(__VA_ARGS__, 1)
#else
    #define LAPACK_dpotrf(...) LAPACK_dpotrf_base(__VA_ARGS__)
#endif

#define LAPACK_cpotrf_base LAPACK_GLOBAL(cpotrf,CPOTRF)
lapack_int LAPACK_cpotrf_base(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_cpotrf(...) LAPACK_cpotrf_base(__VA_ARGS__, 1)
#else
    #define LAPACK_cpotrf(...) LAPACK_cpotrf_base(__VA_ARGS__)
#endif

#define LAPACK_zpotrf_base LAPACK_GLOBAL(zpotrf,ZPOTRF)
lapack_int LAPACK_zpotrf_base(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_zpotrf(...) LAPACK_zpotrf_base(__VA_ARGS__, 1)
#else
    #define LAPACK_zpotrf(...) LAPACK_zpotrf_base(__VA_ARGS__)
#endif

#define LAPACK_sgeqrf LAPACK_GLOBAL(sgeqrf,SGEQRF)
void LAPACK_sgeqrf(
    lapack_int const* m, lapack_int const* n,
    float* A, lapack_int const* lda,
    float* tau,
    float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_dgeqrf LAPACK_GLOBAL(dgeqrf,DGEQRF)
void LAPACK_dgeqrf(
    lapack_int const* m, lapack_int const* n,
    double* A, lapack_int const* lda,
    double* tau,
    double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_cgeqrf LAPACK_GLOBAL(cgeqrf,CGEQRF)
void LAPACK_cgeqrf(
    lapack_int const* m, lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_zgeqrf LAPACK_GLOBAL(zgeqrf,ZGEQRF)
void LAPACK_zgeqrf(
    lapack_int const* m, lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_sorgqr LAPACK_GLOBAL(sorgqr,SORGQR)
void LAPACK_sorgqr(
    lapack_int const* m, lapack_int const* n, lapack_int const* k,
    float* A, lapack_int const* lda,
    float const* tau,
    float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_dorgqr LAPACK_GLOBAL(dorgqr,DORGQR)
void LAPACK_dorgqr(
    lapack_int const* m, lapack_int const* n, lapack_int const* k,
    double* A, lapack_int const* lda,
    double const* tau,
    double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_cungqr LAPACK_GLOBAL(cungqr,CUNGQR)
void LAPACK_cungqr(
    lapack_int const* m, lapack_int const* n, lapack_int const* k,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float const* tau,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_zungqr LAPACK_GLOBAL(zungqr,ZUNGQR)
void LAPACK_zungqr(
    lapack_int const* m, lapack_int const* n, lapack_int const* k,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double const* tau,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_ssyev_base LAPACK_GLOBAL(ssyev,SSYEV)
void LAPACK_ssyev_base(
    char const* jobz, char const* uplo,
    lapack_int const* n,
    float* A, lapack_int const* lda,
    float* W,
    float* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_ssyev(...) LAPACK_ssyev_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_ssyev(...) LAPACK_ssyev_base(__VA_ARGS__)
#endif

#define LAPACK_dsyev_base LAPACK_GLOBAL(dsyev,DSYEV)
void LAPACK_dsyev_base(
    char const* jobz, char const* uplo,
    lapack_int const* n,
    double* A, lapack_int const* lda,
    double* W,
    double* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_dsyev(...) LAPACK_dsyev_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_dsyev(...) LAPACK_dsyev_base(__VA_ARGS__)
#endif

#define LAPACK_cheev_base LAPACK_GLOBAL(cheev,CHEEV)
void LAPACK_cheev_base(
    char const* jobz, char const* uplo,
    lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda,
    float* W,
    lapack_complex_float* work, lapack_int const* lwork,
    float* rwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_cheev(...) LAPACK_cheev_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_cheev(...) LAPACK_cheev_base(__VA_ARGS__)
#endif

#define LAPACK_zheev_base LAPACK_GLOBAL(zheev,ZHEEV)
void LAPACK_zheev_base(
    char const* jobz, char const* uplo,
    lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda,
    double* W,
    lapack_complex_double* work, lapack_int const* lwork,
    double* rwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_zheev(...) LAPACK_zheev_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_zheev(...) LAPACK_zheev_base(__VA_ARGS__)
#endif

#define LAPACK_ssyevd_base LAPACK_GLOBAL(ssyevd,SSYEVD)
void LAPACK_ssyevd_base(
    char const* jobz, char const* uplo,
    lapack_int const* n,
    float* A, lapack_int const* lda,
    float* W,
    float* work, lapack_int const* lwork,
    lapack_int* iwork, lapack_int const* liwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_ssyevd(...) LAPACK_ssyevd_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_ssyevd(...) LAPACK_ssyevd_base(__VA_ARGS__)
#endif

#define LAPACK_dsyevd_base LAPACK_GLOBAL(dsyevd,DSYEVD)
void LAPACK_dsyevd_base(
    char const* jobz, char const* uplo,
    lapack_int const* n,
    double* A, lapack_int const* lda,
    double* W,
    double* work, lapack_int const* lwork,
    lapack_int* iwork, lapack_int const* liwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_dsyevd(...) LAPACK_dsyevd_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_dsyevd(...) LAPACK_dsyevd_base(__VA_ARGS__)
#endif

#define LAPACK_cheevd_base LAPACK_GLOBAL(cheevd,CHEEVD)
void LAPACK_cheevd_base(
    char const* jobz, char const* uplo,
    lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda,
    float* W,
    lapack_complex_float* work, lapack_int const* lwork,
    float* rwork, lapack_int const* lrwork,
    lapack_int* iwork, lapack_int const* liwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_cheevd(...) LAPACK_cheevd_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_cheevd(...) LAPACK_cheevd_base(__VA_ARGS__)
#endif

#define LAPACK_zheevd_base LAPACK_GLOBAL(zheevd,ZHEEVD)
void LAPACK_zheevd_base(
    char const* jobz, char const* uplo,
    lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda,
    double* W,
    lapack_complex_double* work, lapack_int const* lwork,
    double* rwork, lapack_int const* lrwork,
    lapack_int* iwork, lapack_int const* liwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_zheevd(...) LAPACK_zheevd_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_zheevd(...) LAPACK_zheevd_base(__VA_ARGS__)
#endif

#define LAPACK_sgesvd_base LAPACK_GLOBAL(sgesvd,SGESVD)
void LAPACK_sgesvd_base(
    char const* jobu, char const* jobvt,
    lapack_int const* m, lapack_int const* n,
    float* A, lapack_int const* lda,
    float* S,
    float* U, lapack_int const* ldu,
    float* VT, lapack_int const* ldvt,
    float* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_sgesvd(...) LAPACK_sgesvd_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_sgesvd(...) LAPACK_sgesvd_base(__VA_ARGS__)
#endif

#define LAPACK_dgesvd_base LAPACK_GLOBAL(dgesvd,DGESVD)
void LAPACK_dgesvd_base(
    char const* jobu, char const* jobvt,
    lapack_int const* m, lapack_int const* n,
    double* A, lapack_int const* lda,
    double* S,
    double* U, lapack_int const* ldu,
    double* VT, lapack_int const* ldvt,
    double* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_dgesvd(...) LAPACK_dgesvd_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_dgesvd(...) LAPACK_dgesvd_base(__VA_ARGS__)
#endif

#define LAPACK_cgesvd_base LAPACK_GLOBAL(cgesvd,CGESVD)
void LAPACK_cgesvd_base(
    char const* jobu, char const* jobvt,
    lapack_int const* m, lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda,
    float* S,
    lapack_complex_float* U, lapack_int const* ldu,
    lapack_complex_float* VT, lapack_int const* ldvt,
    lapack_complex_float* work, lapack_int const* lwork,
    float* rwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_cgesvd(...) LAPACK_cgesvd_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_cgesvd(...) LAPACK_cgesvd_base(__VA_ARGS__)
#endif

#define LAPACK_zgesvd_base LAPACK_GLOBAL(zgesvd,ZGESVD)
void LAPACK_zgesvd_base(
    char const* jobu, char const* jobvt,
    lapack_int const* m, lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda,
    double* S,
    lapack_complex_double* U, lapack_int const* ldu,
    lapack_complex_double* VT, lapack_int const* ldvt,
    lapack_complex_double* work, lapack_int const* lwork,
    double* rwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_zgesvd(...) LAPACK_zgesvd_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_zgesvd(...) LAPACK_zgesvd_base(__VA_ARGS__)
#endif

#define LAPACK_sgels_base LAPACK_GLOBAL(sgels,SGELS)
void LAPACK_sgels_base(
    char const* trans,
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    float* A, lapack_int const* lda,
    float* B, lapack_int const* ldb,
    float* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_sgels(...) LAPACK_sgels_base(__VA_ARGS__, 1)
#else
    #define LAPACK_sgels(...) LAPACK_sgels_base(__VA_ARGS__)
#endif

#define LAPACK_dgels_base LAPACK_GLOBAL(dgels,DGELS)
void LAPACK_dgels_base(
    char const* trans,
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    double* A, lapack_int const* lda,
    double* B, lapack_int const* ldb,
    double* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_dgels(...) LAPACK_dgels_base(__VA_ARGS__, 1)
#else
    #define LAPACK_dgels(...) LAPACK_dgels_base(__VA_ARGS__)
#endif

#define LAPACK_cgels_base LAPACK_GLOBAL(cgels,CGELS)
void LAPACK_cgels_base(
    char const* trans,
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* B, lapack_int const* ldb,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_cgels(...) LAPACK_cgels_base(__VA_ARGS__, 1)
#else
    #define LAPACK_cgels(...) LAPACK_cgels_base(__VA_ARGS__)
#endif

#define LAPACK_zgels_base LAPACK_GLOBAL(zgels,ZGELS)
void LAPACK_zgels_base(
    char const* trans,
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* B, lapack_int const* ldb,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_zgels(...) LAPACK_zgels_base(__VA_ARGS__, 1)
#else
    #define LAPACK_zgels(...) LAPACK_zgels_base(__VA_ARGS__)
#endif

#define LAPACK_sgelsy LAPACK_GLOBAL(sgelsy,SGELSY)
void LAPACK_sgelsy(
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    float* A, lapack_int const* lda,
    float* B, lapack_int const* ldb, lapack_int* JPVT,
    float const* rcond, lapack_int* rank,
    float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_dgelsy LAPACK_GLOBAL(dgelsy,DGELSY)
void LAPACK_dgelsy(
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    double* A, lapack_int const* lda,
    double* B, lapack_int const* ldb, lapack_int* JPVT,
    double const* rcond, lapack_int* rank,
    double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_cgelsy LAPACK_GLOBAL(cgelsy,CGELSY)
void LAPACK_cgelsy(
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* B, lapack_int const* ldb, lapack_int* JPVT,
    float const* rcond, lapack_int* rank,
    lapack_complex_float* work, lapack_int const* lwork,
    float* rwork,
    lapack_int* info );

#define LAPACK_zgelsy LAPACK_GLOBAL(zgelsy,ZGELSY)
void LAPACK_zgelsy(
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* B, lapack_int const* ldb, lapack_int* JPVT,
    double const* rcond, lapack_int* rank,
    lapack_complex_double* work, lapack_int const* lwork,
    double* rwork,
    lapack_int* info );

#define LAPACK_sgelsd LAPACK_GLOBAL(sgelsd,SGELSD)
void LAPACK_sgelsd(
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    float* A, lapack_int const* lda,
    float* B, lapack_int const* ldb,
    float* S,
    float const* rcond, lapack_int* rank,
    float* work, lapack_int const* lwork,
    lapack_int* iwork,
    lapack_int* info );

#define LAPACK_dgelsd LAPACK_GLOBAL(dgelsd,DGELSD)
void LAPACK_dgelsd(
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    double* A, lapack_int const* lda,
    double* B, lapack_int const* ldb,
    double* S,
    double const* rcond, lapack_int* rank,
    double* work, lapack_int const* lwork,
    lapack_int* iwork,
    lapack_int* info );

#define LAPACK_cgelsd LAPACK_GLOBAL(cgelsd,CGELSD)
void LAPACK_cgelsd(
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* B, lapack_int const* ldb,
    float* S,
    float const* rcond, lapack_int* rank,
    lapack_complex_float* work, lapack_int const* lwork,
    float* rwork,
    lapack_int* iwork,
    lapack_int* info );

#define LAPACK_zgelsd LAPACK_GLOBAL(zgelsd,ZGELSD)
void LAPACK_zgelsd(
    lapack_int const* m, lapack_int const* n, lapack_int const* nrhs,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* B, lapack_int const* ldb,
    double* S,
    double const* rcond, lapack_int* rank,
    lapack_complex_double* work, lapack_int const* lwork,
    double* rwork,
    lapack_int* iwork,
    lapack_int* info );

#define LAPACK_sgees_base LAPACK_GLOBAL(sgees,SGEES)
void LAPACK_sgees_base(
    char const* jobvs, char const* sort, LAPACK_S_SELECT2 select,
    lapack_int const* n,
    float* A, lapack_int const* lda, lapack_int* sdim,
    float* WR,
    float* WI,
    float* VS, lapack_int const* ldvs,
    float* work, lapack_int const* lwork, lapack_logical* BWORK,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_sgees(...) LAPACK_sgees_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_sgees(...) LAPACK_sgees_base(__VA_ARGS__)
#endif

#define LAPACK_dgees_base LAPACK_GLOBAL(dgees,DGEES)
void LAPACK_dgees_base(
    char const* jobvs, char const* sort, LAPACK_D_SELECT2 select,
    lapack_int const* n,
    double* A, lapack_int const* lda, lapack_int* sdim,
    double* WR,
    double* WI,
    double* VS, lapack_int const* ldvs,
    double* work, lapack_int const* lwork, lapack_logical* BWORK,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_dgees(...) LAPACK_dgees_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_dgees(...) LAPACK_dgees_base(__VA_ARGS__)
#endif

#define LAPACK_cgees_base LAPACK_GLOBAL(cgees,CGEES)
void LAPACK_cgees_base(
    char const* jobvs, char const* sort, LAPACK_C_SELECT1 select,
    lapack_int const* n,
    lapack_complex_float* A, lapack_int const* lda, lapack_int* sdim,
    lapack_complex_float* W,
    lapack_complex_float* VS, lapack_int const* ldvs,
    lapack_complex_float* work, lapack_int const* lwork,
    float* rwork, lapack_logical* BWORK,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_cgees(...) LAPACK_cgees_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_cgees(...) LAPACK_cgees_base(__VA_ARGS__)
#endif

#define LAPACK_zgees_base LAPACK_GLOBAL(zgees,ZGEES)
void LAPACK_zgees_base(
    char const* jobvs, char const* sort, LAPACK_Z_SELECT1 select,
    lapack_int const* n,
    lapack_complex_double* A, lapack_int const* lda, lapack_int* sdim,
    lapack_complex_double* W,
    lapack_complex_double* VS, lapack_int const* ldvs,
    lapack_complex_double* work, lapack_int const* lwork,
    double* rwork, lapack_logical* BWORK,
    lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
    , FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_zgees(...) LAPACK_zgees_base(__VA_ARGS__, 1, 1)
#else
    #define LAPACK_zgees(...) LAPACK_zgees_base(__VA_ARGS__)
#endif

#define LAPACK_sgehrd LAPACK_GLOBAL(sgehrd,SGEHRD)
void LAPACK_sgehrd(
    lapack_int const* n, lapack_int const* ilo, lapack_int const* ihi,
    float* A, lapack_int const* lda,
    float* tau,
    float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_dgehrd LAPACK_GLOBAL(dgehrd,DGEHRD)
void LAPACK_dgehrd(
    lapack_int const* n, lapack_int const* ilo, lapack_int const* ihi,
    double* A, lapack_int const* lda,
    double* tau,
    double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_cgehrd LAPACK_GLOBAL(cgehrd,CGEHRD)
void LAPACK_cgehrd(
    lapack_int const* n, lapack_int const* ilo, lapack_int const* ihi,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_zgehrd LAPACK_GLOBAL(zgehrd,ZGEHRD)
void LAPACK_zgehrd(
    lapack_int const* n, lapack_int const* ilo, lapack_int const* ihi,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_sorghr LAPACK_GLOBAL(sorghr,SORGHR)
void LAPACK_sorghr(
    lapack_int const* n, lapack_int const* ilo, lapack_int const* ihi,
    float* A, lapack_int const* lda,
    float const* tau,
    float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_dorghr LAPACK_GLOBAL(dorghr,DORGHR)
void LAPACK_dorghr(
    lapack_int const* n, lapack_int const* ilo, lapack_int const* ihi,
    double* A, lapack_int const* lda,
    double const* tau,
    double* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_cunghr LAPACK_GLOBAL(cunghr,CUNGHR)
void LAPACK_cunghr(
    lapack_int const* n, lapack_int const* ilo, lapack_int const* ihi,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float const* tau,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );

#define LAPACK_zunghr LAPACK_GLOBAL(zunghr,ZUNGHR)
void LAPACK_zunghr(
    lapack_int const* n, lapack_int const* ilo, lapack_int const* ihi,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double const* tau,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );


void LAPACKE_sge_trans( int matrix_layout, lapack_int m, lapack_int n, const float* in, lapack_int ldin, float* out, lapack_int ldout );
void LAPACKE_dge_trans( int matrix_layout, lapack_int m, lapack_int n, const double* in, lapack_int ldin, double* out, lapack_int ldout );
void LAPACKE_cge_trans( int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_float* in, lapack_int ldin, lapack_complex_float* out, lapack_int ldout );
void LAPACKE_zge_trans( int matrix_layout, lapack_int m, lapack_int n, const lapack_complex_double* in, lapack_int ldin, lapack_complex_double* out, lapack_int ldout );
void LAPACKE_str_trans( int matrix_layout, char uplo, char diag, lapack_int n, const float *in, lapack_int ldin, float *out, lapack_int ldout );
void LAPACKE_dtr_trans( int matrix_layout, char uplo, char diag, lapack_int n, const double *in, lapack_int ldin, double *out, lapack_int ldout );
void LAPACKE_ctr_trans( int matrix_layout, char uplo, char diag, lapack_int n, const lapack_complex_float *in, lapack_int ldin, lapack_complex_float *out, lapack_int ldout );
void LAPACKE_ztr_trans( int matrix_layout, char uplo, char diag, lapack_int n, const lapack_complex_double *in, lapack_int ldin, lapack_complex_double *out, lapack_int ldout );
void LAPACKE_spo_trans( int matrix_layout, char uplo, lapack_int n, const float *in, lapack_int ldin, float *out, lapack_int ldout );
void LAPACKE_dpo_trans( int matrix_layout, char uplo, lapack_int n, const double *in, lapack_int ldin, double *out, lapack_int ldout );
void LAPACKE_cpo_trans( int matrix_layout, char uplo, lapack_int n, const lapack_complex_float *in, lapack_int ldin, lapack_complex_float *out, lapack_int ldout );
void LAPACKE_zpo_trans( int matrix_layout, char uplo, lapack_int n, const lapack_complex_double *in, lapack_int ldin, lapack_complex_double *out, lapack_int ldout );
void LAPACKE_ssy_trans( int matrix_layout, char uplo, lapack_int n, const float *in, lapack_int ldin, float *out, lapack_int ldout );
void LAPACKE_dsy_trans( int matrix_layout, char uplo, lapack_int n, const double *in, lapack_int ldin, double *out, lapack_int ldout );
void LAPACKE_che_trans( int matrix_layout, char uplo, lapack_int n, const lapack_complex_float *in, lapack_int ldin, lapack_complex_float *out, lapack_int ldout );
void LAPACKE_zhe_trans( int matrix_layout, char uplo, lapack_int n, const lapack_complex_double *in, lapack_int ldin, lapack_complex_double *out, lapack_int ldout );


void LAPACKE_sge_trans( int matrix_layout, lapack_int m, lapack_int n,
                        const float* in, lapack_int ldin,
                        float* out, lapack_int ldout )
{
    lapack_int i, j, x, y;

    if( in == NULL || out == NULL ) return;

    if( matrix_layout == LAPACK_COL_MAJOR ) {
        x = n;
        y = m;
    } else if ( matrix_layout == LAPACK_ROW_MAJOR ) {
        x = m;
        y = n;
    } else {
        
        return;
    }

    
    for( i = 0; i < MIN( y, ldin ); i++ ) {
        for( j = 0; j < MIN( x, ldout ); j++ ) {
            out[ (size_t)i*ldout + j ] = in[ (size_t)j*ldin + i ];
        }
    }
}

void LAPACKE_dge_trans( int matrix_layout, lapack_int m, lapack_int n,
                        const double* in, lapack_int ldin,
                        double* out, lapack_int ldout )
{
    lapack_int i, j, x, y;

    if( in == NULL || out == NULL ) return;

    if( matrix_layout == LAPACK_COL_MAJOR ) {
        x = n;
        y = m;
    } else if ( matrix_layout == LAPACK_ROW_MAJOR ) {
        x = m;
        y = n;
    } else {
        
        return;
    }

    
    for( i = 0; i < MIN( y, ldin ); i++ ) {
        for( j = 0; j < MIN( x, ldout ); j++ ) {
            out[ (size_t)i*ldout + j ] = in[ (size_t)j*ldin + i ];
        }
    }
}

void LAPACKE_cge_trans( int matrix_layout, lapack_int m, lapack_int n,
                        const lapack_complex_float* in, lapack_int ldin,
                        lapack_complex_float* out, lapack_int ldout )
{
    lapack_int i, j, x, y;

    if( in == NULL || out == NULL ) return;

    if( matrix_layout == LAPACK_COL_MAJOR ) {
        x = n;
        y = m;
    } else if ( matrix_layout == LAPACK_ROW_MAJOR ) {
        x = m;
        y = n;
    } else {
        
        return;
    }

    
    for( i = 0; i < MIN( y, ldin ); i++ ) {
        for( j = 0; j < MIN( x, ldout ); j++ ) {
            out[ (size_t)i*ldout + j ] = in[ (size_t)j*ldin + i ];
        }
    }
}

void LAPACKE_zge_trans( int matrix_layout, lapack_int m, lapack_int n,
                        const lapack_complex_double* in, lapack_int ldin,
                        lapack_complex_double* out, lapack_int ldout )
{
    lapack_int i, j, x, y;

    if( in == NULL || out == NULL ) return;

    if( matrix_layout == LAPACK_COL_MAJOR ) {
        x = n;
        y = m;
    } else if ( matrix_layout == LAPACK_ROW_MAJOR ) {
        x = m;
        y = n;
    } else {
        
        return;
    }

    
    for( i = 0; i < MIN( y, ldin ); i++ ) {
        for( j = 0; j < MIN( x, ldout ); j++ ) {
            out[ (size_t)i*ldout + j ] = in[ (size_t)j*ldin + i ];
        }
    }
}

void LAPACKE_str_trans( int matrix_layout, char uplo, char diag, lapack_int n,
                        const float *in, lapack_int ldin,
                        float *out, lapack_int ldout )
{
    lapack_int i, j, st;
    lapack_logical colmaj, lower, unit;

    if( in == NULL || out == NULL ) return ;

    colmaj = ( matrix_layout == LAPACK_COL_MAJOR );
    lower  = LAPACKE_lsame( uplo, 'l' );
    unit   = LAPACKE_lsame( diag, 'u' );

    if( ( !colmaj && ( matrix_layout != LAPACK_ROW_MAJOR ) ) ||
        ( !lower  && !LAPACKE_lsame( uplo, 'u' ) ) ||
        ( !unit   && !LAPACKE_lsame( diag, 'n' ) ) ) {
        
        return;
    }
    if( unit ) {
        
        st = 1;
    } else  {
        
        st = 0;
    }

    
    if( ( colmaj || lower ) && !( colmaj && lower ) ) {
        for( j = st; j < MIN( n, ldout ); j++ ) {
            for( i = 0; i < MIN( j+1-st, ldin ); i++ ) {
                out[ j+i*ldout ] = in[ i+j*ldin ];
            }
        }
    } else {
        for( j = 0; j < MIN( n-st, ldout ); j++ ) {
            for( i = j+st; i < MIN( n, ldin ); i++ ) {
                out[ j+i*ldout ] = in[ i+j*ldin ];
            }
        }
    }
}

void LAPACKE_dtr_trans( int matrix_layout, char uplo, char diag, lapack_int n,
                        const double *in, lapack_int ldin,
                        double *out, lapack_int ldout )
{
    lapack_int i, j, st;
    lapack_logical colmaj, lower, unit;

    if( in == NULL || out == NULL ) return ;

    colmaj = ( matrix_layout == LAPACK_COL_MAJOR );
    lower  = LAPACKE_lsame( uplo, 'l' );
    unit   = LAPACKE_lsame( diag, 'u' );

    if( ( !colmaj && ( matrix_layout != LAPACK_ROW_MAJOR ) ) ||
        ( !lower  && !LAPACKE_lsame( uplo, 'u' ) ) ||
        ( !unit   && !LAPACKE_lsame( diag, 'n' ) ) ) {
        
        return;
    }
    if( unit ) {
        
        st = 1;
    } else  {
        
        st = 0;
    }

    
    if( ( colmaj || lower ) && !( colmaj && lower ) ) {
        for( j = st; j < MIN( n, ldout ); j++ ) {
            for( i = 0; i < MIN( j+1-st, ldin ); i++ ) {
                out[ j+i*ldout ] = in[ i+j*ldin ];
            }
        }
    } else {
        for( j = 0; j < MIN( n-st, ldout ); j++ ) {
            for( i = j+st; i < MIN( n, ldin ); i++ ) {
                out[ j+i*ldout ] = in[ i+j*ldin ];
            }
        }
    }
}

void LAPACKE_ctr_trans( int matrix_layout, char uplo, char diag, lapack_int n,
                        const lapack_complex_float *in, lapack_int ldin,
                        lapack_complex_float *out, lapack_int ldout )
{
    lapack_int i, j, st;
    lapack_logical colmaj, lower, unit;

    if( in == NULL || out == NULL ) return ;

    colmaj = ( matrix_layout == LAPACK_COL_MAJOR );
    lower  = LAPACKE_lsame( uplo, 'l' );
    unit   = LAPACKE_lsame( diag, 'u' );

    if( ( !colmaj && ( matrix_layout != LAPACK_ROW_MAJOR ) ) ||
        ( !lower  && !LAPACKE_lsame( uplo, 'u' ) ) ||
        ( !unit   && !LAPACKE_lsame( diag, 'n' ) ) ) {
        
        return;
    }
    if( unit ) {
        
        st = 1;
    } else  {
        
        st = 0;
    }

    
    if( ( colmaj || lower ) && !( colmaj && lower ) ) {
        for( j = st; j < MIN( n, ldout ); j++ ) {
            for( i = 0; i < MIN( j+1-st, ldin ); i++ ) {
                out[ j+i*ldout ] = in[ i+j*ldin ];
            }
        }
    } else {
        for( j = 0; j < MIN( n-st, ldout ); j++ ) {
            for( i = j+st; i < MIN( n, ldin ); i++ ) {
                out[ j+i*ldout ] = in[ i+j*ldin ];
            }
        }
    }
}

void LAPACKE_ztr_trans( int matrix_layout, char uplo, char diag, lapack_int n,
                        const lapack_complex_double *in, lapack_int ldin,
                        lapack_complex_double *out, lapack_int ldout )
{
    lapack_int i, j, st;
    lapack_logical colmaj, lower, unit;

    if( in == NULL || out == NULL ) return ;

    colmaj = ( matrix_layout == LAPACK_COL_MAJOR );
    lower  = LAPACKE_lsame( uplo, 'l' );
    unit   = LAPACKE_lsame( diag, 'u' );

    if( ( !colmaj && ( matrix_layout != LAPACK_ROW_MAJOR ) ) ||
        ( !lower  && !LAPACKE_lsame( uplo, 'u' ) ) ||
        ( !unit   && !LAPACKE_lsame( diag, 'n' ) ) ) {
        
        return;
    }
    if( unit ) {
        
        st = 1;
    } else  {
        
        st = 0;
    }

    
    if( ( colmaj || lower ) && !( colmaj && lower ) ) {
        for( j = st; j < MIN( n, ldout ); j++ ) {
            for( i = 0; i < MIN( j+1-st, ldin ); i++ ) {
                out[ j+i*ldout ] = in[ i+j*ldin ];
            }
        }
    } else {
        for( j = 0; j < MIN( n-st, ldout ); j++ ) {
            for( i = j+st; i < MIN( n, ldin ); i++ ) {
                out[ j+i*ldout ] = in[ i+j*ldin ];
            }
        }
    }
}

void LAPACKE_spo_trans( int matrix_layout, char uplo, lapack_int n,
                        const float *in, lapack_int ldin,
                        float *out, lapack_int ldout )
{
    LAPACKE_str_trans( matrix_layout, uplo, 'n', n, in, ldin, out, ldout );
}

void LAPACKE_dpo_trans( int matrix_layout, char uplo, lapack_int n,
                        const double *in, lapack_int ldin,
                        double *out, lapack_int ldout )
{
    LAPACKE_dtr_trans( matrix_layout, uplo, 'n', n, in, ldin, out, ldout );
}

void LAPACKE_cpo_trans( int matrix_layout, char uplo, lapack_int n,
                        const lapack_complex_float *in, lapack_int ldin,
                        lapack_complex_float *out, lapack_int ldout )
{
    LAPACKE_ctr_trans( matrix_layout, uplo, 'n', n, in, ldin, out, ldout );
}

void LAPACKE_zpo_trans( int matrix_layout, char uplo, lapack_int n,
                        const lapack_complex_double *in, lapack_int ldin,
                        lapack_complex_double *out, lapack_int ldout )
{
    LAPACKE_ztr_trans( matrix_layout, uplo, 'n', n, in, ldin, out, ldout );
}

void LAPACKE_ssy_trans( int matrix_layout, char uplo, lapack_int n,
                        const float *in, lapack_int ldin,
                        float *out, lapack_int ldout )
{
    LAPACKE_str_trans( matrix_layout, uplo, 'n', n, in, ldin, out, ldout );
}

void LAPACKE_dsy_trans( int matrix_layout, char uplo, lapack_int n,
                        const double *in, lapack_int ldin,
                        double *out, lapack_int ldout )
{
    LAPACKE_dtr_trans( matrix_layout, uplo, 'n', n, in, ldin, out, ldout );
}

void LAPACKE_che_trans( int matrix_layout, char uplo, lapack_int n,
                        const lapack_complex_float *in, lapack_int ldin,
                        lapack_complex_float *out, lapack_int ldout )
{
    LAPACKE_ctr_trans( matrix_layout, uplo, 'n', n, in, ldin, out, ldout );
}

void LAPACKE_zhe_trans( int matrix_layout, char uplo, lapack_int n,
                        const lapack_complex_double *in, lapack_int ldin,
                        lapack_complex_double *out, lapack_int ldout )
{
    LAPACKE_ztr_trans( matrix_layout, uplo, 'n', n, in, ldin, out, ldout );
}

lapack_int LAPACKE_sgetrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                float* a, lapack_int lda, lapack_int* ipiv );

lapack_int LAPACKE_sgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* ipiv );

lapack_int LAPACKE_dgetrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                double* a, lapack_int lda, lapack_int* ipiv );

lapack_int LAPACKE_dgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv );

lapack_int LAPACKE_cgetrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_complex_float* a, lapack_int lda,
                                lapack_int* ipiv );

lapack_int LAPACKE_cgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );

lapack_int LAPACKE_zgetrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_complex_double* a, lapack_int lda,
                                lapack_int* ipiv );

lapack_int LAPACKE_zgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );

lapack_int LAPACKE_sgetri_work( int matrix_layout, lapack_int n, float* a,
                                lapack_int lda, const lapack_int* ipiv,
                                float* work, lapack_int lwork );

lapack_int LAPACKE_sgetri( int matrix_layout, lapack_int n, float* a,
                           lapack_int lda, const lapack_int* ipiv );

lapack_int LAPACKE_dgetri_work( int matrix_layout, lapack_int n, double* a,
                                lapack_int lda, const lapack_int* ipiv,
                                double* work, lapack_int lwork );

lapack_int LAPACKE_dgetri( int matrix_layout, lapack_int n, double* a,
                           lapack_int lda, const lapack_int* ipiv );

lapack_int LAPACKE_cgetri_work( int matrix_layout, lapack_int n,
                                lapack_complex_float* a, lapack_int lda,
                                const lapack_int* ipiv,
                                lapack_complex_float* work, lapack_int lwork );

lapack_int LAPACKE_cgetri( int matrix_layout, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv );

lapack_int LAPACKE_zgetri_work( int matrix_layout, lapack_int n,
                                lapack_complex_double* a, lapack_int lda,
                                const lapack_int* ipiv,
                                lapack_complex_double* work, lapack_int lwork );

lapack_int LAPACKE_zgetri( int matrix_layout, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv );

lapack_int LAPACKE_sgesv_work( int matrix_layout, lapack_int n, lapack_int nrhs,
                               float* a, lapack_int lda, lapack_int* ipiv,
                               float* b, lapack_int ldb );

lapack_int LAPACKE_sgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          float* a, lapack_int lda, lapack_int* ipiv, float* b,
                          lapack_int ldb );

lapack_int LAPACKE_dgesv_work( int matrix_layout, lapack_int n, lapack_int nrhs,
                               double* a, lapack_int lda, lapack_int* ipiv,
                               double* b, lapack_int ldb );

lapack_int LAPACKE_dgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          double* a, lapack_int lda, lapack_int* ipiv,
                          double* b, lapack_int ldb );

lapack_int LAPACKE_cgesv_work( int matrix_layout, lapack_int n, lapack_int nrhs,
                               lapack_complex_float* a, lapack_int lda,
                               lapack_int* ipiv, lapack_complex_float* b,
                               lapack_int ldb );

lapack_int LAPACKE_cgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );

lapack_int LAPACKE_zgesv_work( int matrix_layout, lapack_int n, lapack_int nrhs,
                               lapack_complex_double* a, lapack_int lda,
                               lapack_int* ipiv, lapack_complex_double* b,
                               lapack_int ldb );

lapack_int LAPACKE_zgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );

lapack_int LAPACKE_sgeev_work( int matrix_layout, char jobvl, char jobvr,
                               lapack_int n, float* a, lapack_int lda,
                               float* wr, float* wi, float* vl, lapack_int ldvl,
                               float* vr, lapack_int ldvr, float* work,
                               lapack_int lwork );

lapack_int LAPACKE_sgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, float* a, lapack_int lda, float* wr,
                          float* wi, float* vl, lapack_int ldvl, float* vr,
                          lapack_int ldvr );

lapack_int LAPACKE_dgeev_work( int matrix_layout, char jobvl, char jobvr,
                               lapack_int n, double* a, lapack_int lda,
                               double* wr, double* wi, double* vl,
                               lapack_int ldvl, double* vr, lapack_int ldvr,
                               double* work, lapack_int lwork );

lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, double* a, lapack_int lda, double* wr,
                          double* wi, double* vl, lapack_int ldvl, double* vr,
                          lapack_int ldvr );

lapack_int LAPACKE_cgeev_work( int matrix_layout, char jobvl, char jobvr,
                               lapack_int n, lapack_complex_float* a,
                               lapack_int lda, lapack_complex_float* w,
                               lapack_complex_float* vl, lapack_int ldvl,
                               lapack_complex_float* vr, lapack_int ldvr,
                               lapack_complex_float* work, lapack_int lwork,
                               float* rwork );

lapack_int LAPACKE_cgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* w, lapack_complex_float* vl,
                          lapack_int ldvl, lapack_complex_float* vr,
                          lapack_int ldvr );

lapack_int LAPACKE_zgeev_work( int matrix_layout, char jobvl, char jobvr,
                               lapack_int n, lapack_complex_double* a,
                               lapack_int lda, lapack_complex_double* w,
                               lapack_complex_double* vl, lapack_int ldvl,
                               lapack_complex_double* vr, lapack_int ldvr,
                               lapack_complex_double* work, lapack_int lwork,
                               double* rwork );

lapack_int LAPACKE_zgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* w,
                          lapack_complex_double* vl, lapack_int ldvl,
                          lapack_complex_double* vr, lapack_int ldvr );

lapack_int LAPACKE_spotrf_work( int matrix_layout, char uplo, lapack_int n,
                                float* a, lapack_int lda );

lapack_int LAPACKE_spotrf( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda );

lapack_int LAPACKE_dpotrf_work( int matrix_layout, char uplo, lapack_int n,
                                double* a, lapack_int lda );

lapack_int LAPACKE_dpotrf( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda );

lapack_int LAPACKE_cpotrf_work( int matrix_layout, char uplo, lapack_int n,
                                lapack_complex_float* a, lapack_int lda );

lapack_int LAPACKE_cpotrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );

lapack_int LAPACKE_zpotrf_work( int matrix_layout, char uplo, lapack_int n,
                                lapack_complex_double* a, lapack_int lda );

lapack_int LAPACKE_zpotrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );

lapack_int LAPACKE_sgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                float* a, lapack_int lda, float* tau,
                                float* work, lapack_int lwork );

lapack_int LAPACKE_sgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );

lapack_int LAPACKE_dgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                double* a, lapack_int lda, double* tau,
                                double* work, lapack_int lwork );

lapack_int LAPACKE_dgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );

lapack_int LAPACKE_cgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_complex_float* a, lapack_int lda,
                                lapack_complex_float* tau,
                                lapack_complex_float* work, lapack_int lwork );

lapack_int LAPACKE_cgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );

lapack_int LAPACKE_zgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_complex_double* a, lapack_int lda,
                                lapack_complex_double* tau,
                                lapack_complex_double* work, lapack_int lwork );

lapack_int LAPACKE_zgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );

lapack_int LAPACKE_sorgqr_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int k, float* a, lapack_int lda,
                                const float* tau, float* work,
                                lapack_int lwork );

lapack_int LAPACKE_sorgqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );

lapack_int LAPACKE_dorgqr_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int k, double* a, lapack_int lda,
                                const double* tau, double* work,
                                lapack_int lwork );

lapack_int LAPACKE_dorgqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

lapack_int LAPACKE_cungqr_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int k, lapack_complex_float* a,
                                lapack_int lda, const lapack_complex_float* tau,
                                lapack_complex_float* work, lapack_int lwork );

lapack_int LAPACKE_cungqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );

lapack_int LAPACKE_zungqr_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int k, lapack_complex_double* a,
                                lapack_int lda,
                                const lapack_complex_double* tau,
                                lapack_complex_double* work, lapack_int lwork );

lapack_int LAPACKE_zungqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );

lapack_int LAPACKE_ssyev_work( int matrix_layout, char jobz, char uplo,
                               lapack_int n, float* a, lapack_int lda, float* w,
                               float* work, lapack_int lwork );

lapack_int LAPACKE_ssyev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          float* a, lapack_int lda, float* w );

lapack_int LAPACKE_dsyev_work( int matrix_layout, char jobz, char uplo,
                               lapack_int n, double* a, lapack_int lda,
                               double* w, double* work, lapack_int lwork );

lapack_int LAPACKE_dsyev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          double* a, lapack_int lda, double* w );

lapack_int LAPACKE_cheev_work( int matrix_layout, char jobz, char uplo,
                               lapack_int n, lapack_complex_float* a,
                               lapack_int lda, float* w,
                               lapack_complex_float* work, lapack_int lwork,
                               float* rwork );

lapack_int LAPACKE_cheev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_complex_float* a, lapack_int lda, float* w );

lapack_int LAPACKE_zheev_work( int matrix_layout, char jobz, char uplo,
                               lapack_int n, lapack_complex_double* a,
                               lapack_int lda, double* w,
                               lapack_complex_double* work, lapack_int lwork,
                               double* rwork );

lapack_int LAPACKE_zheev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_complex_double* a, lapack_int lda, double* w );

lapack_int LAPACKE_ssyevd_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, float* a, lapack_int lda,
                                float* w, float* work, lapack_int lwork,
                                lapack_int* iwork, lapack_int liwork );

lapack_int LAPACKE_ssyevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           float* a, lapack_int lda, float* w );

lapack_int LAPACKE_dsyevd_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, double* a, lapack_int lda,
                                double* w, double* work, lapack_int lwork,
                                lapack_int* iwork, lapack_int liwork );

lapack_int LAPACKE_dsyevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           double* a, lapack_int lda, double* w );

lapack_int LAPACKE_cheevd_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, lapack_complex_float* a,
                                lapack_int lda, float* w,
                                lapack_complex_float* work, lapack_int lwork,
                                float* rwork, lapack_int lrwork,
                                lapack_int* iwork, lapack_int liwork );

lapack_int LAPACKE_cheevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda, float* w );

lapack_int LAPACKE_zheevd_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, lapack_complex_double* a,
                                lapack_int lda, double* w,
                                lapack_complex_double* work, lapack_int lwork,
                                double* rwork, lapack_int lrwork,
                                lapack_int* iwork, lapack_int liwork );

lapack_int LAPACKE_zheevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           double* w );

lapack_int LAPACKE_sgesvd_work( int matrix_layout, char jobu, char jobvt,
                                lapack_int m, lapack_int n, float* a,
                                lapack_int lda, float* s, float* u,
                                lapack_int ldu, float* vt, lapack_int ldvt,
                                float* work, lapack_int lwork );

lapack_int LAPACKE_sgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, float* a, lapack_int lda,
                           float* s, float* u, lapack_int ldu, float* vt,
                           lapack_int ldvt, float* superb );

lapack_int LAPACKE_dgesvd_work( int matrix_layout, char jobu, char jobvt,
                                lapack_int m, lapack_int n, double* a,
                                lapack_int lda, double* s, double* u,
                                lapack_int ldu, double* vt, lapack_int ldvt,
                                double* work, lapack_int lwork );

lapack_int LAPACKE_dgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double* s, double* u, lapack_int ldu,
                           double* vt, lapack_int ldvt, double* superb );

lapack_int LAPACKE_cgesvd_work( int matrix_layout, char jobu, char jobvt,
                                lapack_int m, lapack_int n,
                                lapack_complex_float* a, lapack_int lda,
                                float* s, lapack_complex_float* u,
                                lapack_int ldu, lapack_complex_float* vt,
                                lapack_int ldvt, lapack_complex_float* work,
                                lapack_int lwork, float* rwork );

lapack_int LAPACKE_cgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float* s, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* vt,
                           lapack_int ldvt, float* superb );

lapack_int LAPACKE_zgesvd_work( int matrix_layout, char jobu, char jobvt,
                                lapack_int m, lapack_int n,
                                lapack_complex_double* a, lapack_int lda,
                                double* s, lapack_complex_double* u,
                                lapack_int ldu, lapack_complex_double* vt,
                                lapack_int ldvt, lapack_complex_double* work,
                                lapack_int lwork, double* rwork );

lapack_int LAPACKE_zgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double* s, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* vt,
                           lapack_int ldvt, double* superb );

lapack_int LAPACKE_sgels_work( int matrix_layout, char trans, lapack_int m,
                               lapack_int n, lapack_int nrhs, float* a,
                               lapack_int lda, float* b, lapack_int ldb,
                               float* work, lapack_int lwork );

lapack_int LAPACKE_sgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs, float* a,
                          lapack_int lda, float* b, lapack_int ldb );

lapack_int LAPACKE_dgels_work( int matrix_layout, char trans, lapack_int m,
                               lapack_int n, lapack_int nrhs, double* a,
                               lapack_int lda, double* b, lapack_int ldb,
                               double* work, lapack_int lwork );

lapack_int LAPACKE_dgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs, double* a,
                          lapack_int lda, double* b, lapack_int ldb );

lapack_int LAPACKE_cgels_work( int matrix_layout, char trans, lapack_int m,
                               lapack_int n, lapack_int nrhs,
                               lapack_complex_float* a, lapack_int lda,
                               lapack_complex_float* b, lapack_int ldb,
                               lapack_complex_float* work, lapack_int lwork );

lapack_int LAPACKE_cgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* b, lapack_int ldb );

lapack_int LAPACKE_zgels_work( int matrix_layout, char trans, lapack_int m,
                               lapack_int n, lapack_int nrhs,
                               lapack_complex_double* a, lapack_int lda,
                               lapack_complex_double* b, lapack_int ldb,
                               lapack_complex_double* work, lapack_int lwork );

lapack_int LAPACKE_zgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_sgelsy_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, float* a, lapack_int lda,
                                float* b, lapack_int ldb, lapack_int* jpvt,
                                float rcond, lapack_int* rank, float* work,
                                lapack_int lwork );

lapack_int LAPACKE_sgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, lapack_int* jpvt, float rcond,
                           lapack_int* rank );

lapack_int LAPACKE_dgelsy_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, double* a, lapack_int lda,
                                double* b, lapack_int ldb, lapack_int* jpvt,
                                double rcond, lapack_int* rank, double* work,
                                lapack_int lwork );

lapack_int LAPACKE_dgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, lapack_int* jpvt,
                           double rcond, lapack_int* rank );

lapack_int LAPACKE_cgelsy_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, lapack_complex_float* a,
                                lapack_int lda, lapack_complex_float* b,
                                lapack_int ldb, lapack_int* jpvt, float rcond,
                                lapack_int* rank, lapack_complex_float* work,
                                lapack_int lwork, float* rwork );

lapack_int LAPACKE_cgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_int* jpvt, float rcond,
                           lapack_int* rank );

lapack_int LAPACKE_zgelsy_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, lapack_complex_double* a,
                                lapack_int lda, lapack_complex_double* b,
                                lapack_int ldb, lapack_int* jpvt, double rcond,
                                lapack_int* rank, lapack_complex_double* work,
                                lapack_int lwork, double* rwork );

lapack_int LAPACKE_zgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_int* jpvt, double rcond,
                           lapack_int* rank );

lapack_int LAPACKE_sgelsd_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, float* a, lapack_int lda,
                                float* b, lapack_int ldb, float* s, float rcond,
                                lapack_int* rank, float* work, lapack_int lwork,
                                lapack_int* iwork );

lapack_int LAPACKE_sgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );

lapack_int LAPACKE_dgelsd_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, double* a, lapack_int lda,
                                double* b, lapack_int ldb, double* s,
                                double rcond, lapack_int* rank, double* work,
                                lapack_int lwork, lapack_int* iwork );

lapack_int LAPACKE_dgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );

lapack_int LAPACKE_cgelsd_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, lapack_complex_float* a,
                                lapack_int lda, lapack_complex_float* b,
                                lapack_int ldb, float* s, float rcond,
                                lapack_int* rank, lapack_complex_float* work,
                                lapack_int lwork, float* rwork,
                                lapack_int* iwork );

lapack_int LAPACKE_cgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );

lapack_int LAPACKE_zgelsd_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, lapack_complex_double* a,
                                lapack_int lda, lapack_complex_double* b,
                                lapack_int ldb, double* s, double rcond,
                                lapack_int* rank, lapack_complex_double* work,
                                lapack_int lwork, double* rwork,
                                lapack_int* iwork );

lapack_int LAPACKE_zgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );

lapack_int LAPACKE_sgees_work( int matrix_layout, char jobvs, char sort,
                               LAPACK_S_SELECT2 select, lapack_int n, float* a,
                               lapack_int lda, lapack_int* sdim, float* wr,
                               float* wi, float* vs, lapack_int ldvs,
                               float* work, lapack_int lwork,
                               lapack_logical* bwork );

lapack_int LAPACKE_sgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_S_SELECT2 select, lapack_int n, float* a,
                          lapack_int lda, lapack_int* sdim, float* wr,
                          float* wi, float* vs, lapack_int ldvs );

lapack_int LAPACKE_dgees_work( int matrix_layout, char jobvs, char sort,
                               LAPACK_D_SELECT2 select, lapack_int n, double* a,
                               lapack_int lda, lapack_int* sdim, double* wr,
                               double* wi, double* vs, lapack_int ldvs,
                               double* work, lapack_int lwork,
                               lapack_logical* bwork );

lapack_int LAPACKE_dgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_D_SELECT2 select, lapack_int n, double* a,
                          lapack_int lda, lapack_int* sdim, double* wr,
                          double* wi, double* vs, lapack_int ldvs );

lapack_int LAPACKE_cgees_work( int matrix_layout, char jobvs, char sort,
                               LAPACK_C_SELECT1 select, lapack_int n,
                               lapack_complex_float* a, lapack_int lda,
                               lapack_int* sdim, lapack_complex_float* w,
                               lapack_complex_float* vs, lapack_int ldvs,
                               lapack_complex_float* work, lapack_int lwork,
                               float* rwork, lapack_logical* bwork );

lapack_int LAPACKE_cgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_C_SELECT1 select, lapack_int n,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_int* sdim, lapack_complex_float* w,
                          lapack_complex_float* vs, lapack_int ldvs );

lapack_int LAPACKE_zgees_work( int matrix_layout, char jobvs, char sort,
                               LAPACK_Z_SELECT1 select, lapack_int n,
                               lapack_complex_double* a, lapack_int lda,
                               lapack_int* sdim, lapack_complex_double* w,
                               lapack_complex_double* vs, lapack_int ldvs,
                               lapack_complex_double* work, lapack_int lwork,
                               double* rwork, lapack_logical* bwork );

lapack_int LAPACKE_zgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_Z_SELECT1 select, lapack_int n,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_int* sdim, lapack_complex_double* w,
                          lapack_complex_double* vs, lapack_int ldvs );

lapack_int LAPACKE_sgehrd_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, float* a, lapack_int lda,
                                float* tau, float* work, lapack_int lwork );

lapack_int LAPACKE_sgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, float* a, lapack_int lda,
                           float* tau );

lapack_int LAPACKE_dgehrd_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, double* a, lapack_int lda,
                                double* tau, double* work, lapack_int lwork );

lapack_int LAPACKE_dgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, double* a, lapack_int lda,
                           double* tau );

lapack_int LAPACKE_cgehrd_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, lapack_complex_float* a,
                                lapack_int lda, lapack_complex_float* tau,
                                lapack_complex_float* work, lapack_int lwork );

lapack_int LAPACKE_cgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* tau );

lapack_int LAPACKE_zgehrd_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, lapack_complex_double* a,
                                lapack_int lda, lapack_complex_double* tau,
                                lapack_complex_double* work, lapack_int lwork );

lapack_int LAPACKE_zgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* tau );

lapack_int LAPACKE_sorghr_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, float* a, lapack_int lda,
                                const float* tau, float* work,
                                lapack_int lwork );

lapack_int LAPACKE_sorghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, float* a, lapack_int lda,
                           const float* tau );

lapack_int LAPACKE_dorghr_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, double* a, lapack_int lda,
                                const double* tau, double* work,
                                lapack_int lwork );

lapack_int LAPACKE_dorghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, double* a, lapack_int lda,
                           const double* tau );

lapack_int LAPACKE_cunghr_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, lapack_complex_float* a,
                                lapack_int lda, const lapack_complex_float* tau,
                                lapack_complex_float* work, lapack_int lwork );

lapack_int LAPACKE_cunghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );

lapack_int LAPACKE_zunghr_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, lapack_complex_double* a,
                                lapack_int lda,
                                const lapack_complex_double* tau,
                                lapack_complex_double* work, lapack_int lwork );

lapack_int LAPACKE_zunghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );

lapack_int LAPACKE_sgetrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                float* a, lapack_int lda, lapack_int* ipiv )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgetrf( &m, &n, a, &lda, ipiv, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_sgetrf_work", info );
            return info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_sge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_sgetrf( &m, &n, a_t, &lda_t, ipiv, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgetrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgetrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* ipiv )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgetrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    return LAPACKE_sgetrf_work( matrix_layout, m, n, a, lda, ipiv );
}

lapack_int LAPACKE_dgetrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                double* a, lapack_int lda, lapack_int* ipiv )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgetrf( &m, &n, a, &lda, ipiv, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_dgetrf_work", info );
            return info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_dgetrf( &m, &n, a_t, &lda_t, ipiv, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgetrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgetrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgetrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    return LAPACKE_dgetrf_work( matrix_layout, m, n, a, lda, ipiv );
}

lapack_int LAPACKE_cgetrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_complex_float* a, lapack_int lda,
                                lapack_int* ipiv )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgetrf( &m, &n, a, &lda, ipiv, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_cgetrf_work", info );
            return info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_cge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_cgetrf( &m, &n, a_t, &lda_t, ipiv, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgetrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgetrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgetrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    return LAPACKE_cgetrf_work( matrix_layout, m, n, a, lda, ipiv );
}

lapack_int LAPACKE_zgetrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_complex_double* a, lapack_int lda,
                                lapack_int* ipiv )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgetrf( &m, &n, a, &lda, ipiv, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_zgetrf_work", info );
            return info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_zgetrf( &m, &n, a_t, &lda_t, ipiv, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgetrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgetrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgetrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    return LAPACKE_zgetrf_work( matrix_layout, m, n, a, lda, ipiv );
}

lapack_int LAPACKE_sgetri_work( int matrix_layout, lapack_int n, float* a,
                                lapack_int lda, const lapack_int* ipiv,
                                float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgetri( &n, a, &lda, ipiv, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -4;
            LAPACKE_xerbla( "LAPACKE_sgetri_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sgetri( &n, a, &lda_t, ipiv, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_sge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_sgetri( &n, a_t, &lda_t, ipiv, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgetri_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgetri_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgetri( int matrix_layout, lapack_int n, float* a,
                           lapack_int lda, const lapack_int* ipiv )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgetri", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -3;
        }
    }
#endif
    
    info = LAPACKE_sgetri_work( matrix_layout, n, a, lda, ipiv, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sgetri_work( matrix_layout, n, a, lda, ipiv, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgetri", info );
    }
    return info;
}

lapack_int LAPACKE_dgetri_work( int matrix_layout, lapack_int n, double* a,
                                lapack_int lda, const lapack_int* ipiv,
                                double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgetri( &n, a, &lda, ipiv, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -4;
            LAPACKE_xerbla( "LAPACKE_dgetri_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dgetri( &n, a, &lda_t, ipiv, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_dgetri( &n, a_t, &lda_t, ipiv, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgetri_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgetri_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgetri( int matrix_layout, lapack_int n, double* a,
                           lapack_int lda, const lapack_int* ipiv )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgetri", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -3;
        }
    }
#endif
    
    info = LAPACKE_dgetri_work( matrix_layout, n, a, lda, ipiv, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dgetri_work( matrix_layout, n, a, lda, ipiv, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgetri", info );
    }
    return info;
}

lapack_int LAPACKE_cgetri_work( int matrix_layout, lapack_int n,
                                lapack_complex_float* a, lapack_int lda,
                                const lapack_int* ipiv,
                                lapack_complex_float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgetri( &n, a, &lda, ipiv, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -4;
            LAPACKE_xerbla( "LAPACKE_cgetri_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cgetri( &n, a, &lda_t, ipiv, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_cge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_cgetri( &n, a_t, &lda_t, ipiv, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgetri_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgetri_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgetri( int matrix_layout, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgetri", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -3;
        }
    }
#endif
    
    info = LAPACKE_cgetri_work( matrix_layout, n, a, lda, ipiv, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cgetri_work( matrix_layout, n, a, lda, ipiv, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgetri", info );
    }
    return info;
}

lapack_int LAPACKE_zgetri_work( int matrix_layout, lapack_int n,
                                lapack_complex_double* a, lapack_int lda,
                                const lapack_int* ipiv,
                                lapack_complex_double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgetri( &n, a, &lda, ipiv, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -4;
            LAPACKE_xerbla( "LAPACKE_zgetri_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zgetri( &n, a, &lda_t, ipiv, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_zgetri( &n, a_t, &lda_t, ipiv, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgetri_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgetri_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgetri( int matrix_layout, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgetri", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -3;
        }
    }
#endif
    
    info = LAPACKE_zgetri_work( matrix_layout, n, a, lda, ipiv, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zgetri_work( matrix_layout, n, a, lda, ipiv, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgetri", info );
    }
    return info;
}

lapack_int LAPACKE_sgesv_work( int matrix_layout, lapack_int n, lapack_int nrhs,
                               float* a, lapack_int lda, lapack_int* ipiv,
                               float* b, lapack_int ldb )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgesv( &n, &nrhs, a, &lda, ipiv, b, &ldb, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldb_t = MAX(1,n);
        float* a_t = NULL;
        float* b_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_sgesv_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_sgesv_work", info );
            return info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (float*)LAPACKE_malloc( sizeof(float) * ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_sge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        LAPACKE_sge_trans( matrix_layout, n, nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_sgesv( &n, &nrhs, a_t, &lda_t, ipiv, b_t, &ldb_t, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, nrhs, b_t, ldb_t, b, ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgesv_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgesv_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          float* a, lapack_int lda, lapack_int* ipiv, float* b,
                          lapack_int ldb )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgesv", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -4;
        }
        if( LAPACKE_sge_nancheck( matrix_layout, n, nrhs, b, ldb ) ) {
            return -7;
        }
    }
#endif
    return LAPACKE_sgesv_work( matrix_layout, n, nrhs, a, lda, ipiv, b, ldb );
}

lapack_int LAPACKE_dgesv_work( int matrix_layout, lapack_int n, lapack_int nrhs,
                               double* a, lapack_int lda, lapack_int* ipiv,
                               double* b, lapack_int ldb )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgesv( &n, &nrhs, a, &lda, ipiv, b, &ldb, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldb_t = MAX(1,n);
        double* a_t = NULL;
        double* b_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_dgesv_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_dgesv_work", info );
            return info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (double*)LAPACKE_malloc( sizeof(double) * ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_dge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        LAPACKE_dge_trans( matrix_layout, n, nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_dgesv( &n, &nrhs, a_t, &lda_t, ipiv, b_t, &ldb_t, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, nrhs, b_t, ldb_t, b, ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgesv_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgesv_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          double* a, lapack_int lda, lapack_int* ipiv,
                          double* b, lapack_int ldb )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgesv", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -4;
        }
        if( LAPACKE_dge_nancheck( matrix_layout, n, nrhs, b, ldb ) ) {
            return -7;
        }
    }
#endif
    return LAPACKE_dgesv_work( matrix_layout, n, nrhs, a, lda, ipiv, b, ldb );
}

lapack_int LAPACKE_cgesv_work( int matrix_layout, lapack_int n, lapack_int nrhs,
                               lapack_complex_float* a, lapack_int lda,
                               lapack_int* ipiv, lapack_complex_float* b,
                               lapack_int ldb )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgesv( &n, &nrhs, a, &lda, ipiv, b, &ldb, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldb_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        lapack_complex_float* b_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_cgesv_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_cgesv_work", info );
            return info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) *
                            ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_cge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        LAPACKE_cge_trans( matrix_layout, n, nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_cgesv( &n, &nrhs, a_t, &lda_t, ipiv, b_t, &ldb_t, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, nrhs, b_t, ldb_t, b, ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgesv_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgesv_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgesv", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -4;
        }
        if( LAPACKE_cge_nancheck( matrix_layout, n, nrhs, b, ldb ) ) {
            return -7;
        }
    }
#endif
    return LAPACKE_cgesv_work( matrix_layout, n, nrhs, a, lda, ipiv, b, ldb );
}

lapack_int LAPACKE_zgesv_work( int matrix_layout, lapack_int n, lapack_int nrhs,
                               lapack_complex_double* a, lapack_int lda,
                               lapack_int* ipiv, lapack_complex_double* b,
                               lapack_int ldb )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgesv( &n, &nrhs, a, &lda, ipiv, b, &ldb, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldb_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        lapack_complex_double* b_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_zgesv_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_zgesv_work", info );
            return info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) *
                            ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_zge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        LAPACKE_zge_trans( matrix_layout, n, nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_zgesv( &n, &nrhs, a_t, &lda_t, ipiv, b_t, &ldb_t, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, nrhs, b_t, ldb_t, b, ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgesv_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgesv_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgesv", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -4;
        }
        if( LAPACKE_zge_nancheck( matrix_layout, n, nrhs, b, ldb ) ) {
            return -7;
        }
    }
#endif
    return LAPACKE_zgesv_work( matrix_layout, n, nrhs, a, lda, ipiv, b, ldb );
}

lapack_int LAPACKE_sgeev_work( int matrix_layout, char jobvl, char jobvr,
                               lapack_int n, float* a, lapack_int lda,
                               float* wr, float* wi, float* vl, lapack_int ldvl,
                               float* vr, lapack_int ldvr, float* work,
                               lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgeev( &jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr,
                      work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldvl_t = MAX(1,n);
        lapack_int ldvr_t = MAX(1,n);
        float* a_t = NULL;
        float* vl_t = NULL;
        float* vr_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_sgeev_work", info );
            return info;
        }
        if( ldvl < 1 || ( LAPACKE_lsame( jobvl, 'v' ) && ldvl < n ) ) {
            info = -10;
            LAPACKE_xerbla( "LAPACKE_sgeev_work", info );
            return info;
        }
        if( ldvr < 1 || ( LAPACKE_lsame( jobvr, 'v' ) && ldvr < n ) ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_sgeev_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sgeev( &jobvl, &jobvr, &n, a, &lda_t, wr, wi, vl, &ldvl_t,
                          vr, &ldvr_t, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            vl_t = (float*)LAPACKE_malloc( sizeof(float) * ldvl_t * MAX(1,n) );
            if( vl_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            vr_t = (float*)LAPACKE_malloc( sizeof(float) * ldvr_t * MAX(1,n) );
            if( vr_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        
        LAPACKE_sge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_sgeev( &jobvl, &jobvr, &n, a_t, &lda_t, wr, wi, vl_t, &ldvl_t,
                      vr_t, &ldvr_t, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, vl_t, ldvl_t, vl, ldvl );
        }
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, vr_t, ldvr_t, vr, ldvr );
        }
        
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            LAPACKE_free( vr_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            LAPACKE_free( vl_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgeev_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgeev_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, float* a, lapack_int lda, float* wr,
                          float* wi, float* vl, lapack_int ldvl, float* vr,
                          lapack_int ldvr )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgeev", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_sgeev_work( matrix_layout, jobvl, jobvr, n, a, lda, wr, wi,
                               vl, ldvl, vr, ldvr, &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sgeev_work( matrix_layout, jobvl, jobvr, n, a, lda, wr, wi,
                               vl, ldvl, vr, ldvr, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgeev", info );
    }
    return info;
}

lapack_int LAPACKE_dgeev_work( int matrix_layout, char jobvl, char jobvr,
                               lapack_int n, double* a, lapack_int lda,
                               double* wr, double* wi, double* vl,
                               lapack_int ldvl, double* vr, lapack_int ldvr,
                               double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgeev( &jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr,
                      work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldvl_t = MAX(1,n);
        lapack_int ldvr_t = MAX(1,n);
        double* a_t = NULL;
        double* vl_t = NULL;
        double* vr_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_dgeev_work", info );
            return info;
        }
        if( ldvl < 1 || ( LAPACKE_lsame( jobvl, 'v' ) && ldvl < n ) ) {
            info = -10;
            LAPACKE_xerbla( "LAPACKE_dgeev_work", info );
            return info;
        }
        if( ldvr < 1 || ( LAPACKE_lsame( jobvr, 'v' ) && ldvr < n ) ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_dgeev_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dgeev( &jobvl, &jobvr, &n, a, &lda_t, wr, wi, vl, &ldvl_t,
                          vr, &ldvr_t, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            vl_t = (double*)
                LAPACKE_malloc( sizeof(double) * ldvl_t * MAX(1,n) );
            if( vl_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            vr_t = (double*)
                LAPACKE_malloc( sizeof(double) * ldvr_t * MAX(1,n) );
            if( vr_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        
        LAPACKE_dge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_dgeev( &jobvl, &jobvr, &n, a_t, &lda_t, wr, wi, vl_t, &ldvl_t,
                      vr_t, &ldvr_t, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, vl_t, ldvl_t, vl, ldvl );
        }
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, vr_t, ldvr_t, vr, ldvr );
        }
        
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            LAPACKE_free( vr_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            LAPACKE_free( vl_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgeev_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgeev_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, double* a, lapack_int lda, double* wr,
                          double* wi, double* vl, lapack_int ldvl, double* vr,
                          lapack_int ldvr )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgeev", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_dgeev_work( matrix_layout, jobvl, jobvr, n, a, lda, wr, wi,
                               vl, ldvl, vr, ldvr, &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dgeev_work( matrix_layout, jobvl, jobvr, n, a, lda, wr, wi,
                               vl, ldvl, vr, ldvr, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgeev", info );
    }
    return info;
}

lapack_int LAPACKE_cgeev_work( int matrix_layout, char jobvl, char jobvr,
                               lapack_int n, lapack_complex_float* a,
                               lapack_int lda, lapack_complex_float* w,
                               lapack_complex_float* vl, lapack_int ldvl,
                               lapack_complex_float* vr, lapack_int ldvr,
                               lapack_complex_float* work, lapack_int lwork,
                               float* rwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgeev( &jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr,
                      work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldvl_t = MAX(1,n);
        lapack_int ldvr_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        lapack_complex_float* vl_t = NULL;
        lapack_complex_float* vr_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_cgeev_work", info );
            return info;
        }
        if( ldvl < 1 || ( LAPACKE_lsame( jobvl, 'v' ) && ldvl < n ) ) {
            info = -9;
            LAPACKE_xerbla( "LAPACKE_cgeev_work", info );
            return info;
        }
        if( ldvr < 1 || ( LAPACKE_lsame( jobvr, 'v' ) && ldvr < n ) ) {
            info = -11;
            LAPACKE_xerbla( "LAPACKE_cgeev_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cgeev( &jobvl, &jobvr, &n, a, &lda_t, w, vl, &ldvl_t, vr,
                          &ldvr_t, work, &lwork, rwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            vl_t = (lapack_complex_float*)
                LAPACKE_malloc( sizeof(lapack_complex_float) *
                                ldvl_t * MAX(1,n) );
            if( vl_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            vr_t = (lapack_complex_float*)
                LAPACKE_malloc( sizeof(lapack_complex_float) *
                                ldvr_t * MAX(1,n) );
            if( vr_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        
        LAPACKE_cge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_cgeev( &jobvl, &jobvr, &n, a_t, &lda_t, w, vl_t, &ldvl_t, vr_t,
                      &ldvr_t, work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, vl_t, ldvl_t, vl, ldvl );
        }
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, vr_t, ldvr_t, vr, ldvr );
        }
        
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            LAPACKE_free( vr_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            LAPACKE_free( vl_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgeev_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgeev_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* w, lapack_complex_float* vl,
                          lapack_int ldvl, lapack_complex_float* vr,
                          lapack_int ldvr )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* rwork = NULL;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgeev", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    rwork = (float*)LAPACKE_malloc( sizeof(float) * MAX(1,2*n) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cgeev_work( matrix_layout, jobvl, jobvr, n, a, lda, w, vl,
                               ldvl, vr, ldvr, &work_query, lwork, rwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_cgeev_work( matrix_layout, jobvl, jobvr, n, a, lda, w, vl,
                               ldvl, vr, ldvr, work, lwork, rwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( rwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgeev", info );
    }
    return info;
}

lapack_int LAPACKE_zgeev_work( int matrix_layout, char jobvl, char jobvr,
                               lapack_int n, lapack_complex_double* a,
                               lapack_int lda, lapack_complex_double* w,
                               lapack_complex_double* vl, lapack_int ldvl,
                               lapack_complex_double* vr, lapack_int ldvr,
                               lapack_complex_double* work, lapack_int lwork,
                               double* rwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgeev( &jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr,
                      work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldvl_t = MAX(1,n);
        lapack_int ldvr_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        lapack_complex_double* vl_t = NULL;
        lapack_complex_double* vr_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_zgeev_work", info );
            return info;
        }
        if( ldvl < 1 || ( LAPACKE_lsame( jobvl, 'v' ) && ldvl < n ) ) {
            info = -9;
            LAPACKE_xerbla( "LAPACKE_zgeev_work", info );
            return info;
        }
        if( ldvr < 1 || ( LAPACKE_lsame( jobvr, 'v' ) && ldvr < n ) ) {
            info = -11;
            LAPACKE_xerbla( "LAPACKE_zgeev_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zgeev( &jobvl, &jobvr, &n, a, &lda_t, w, vl, &ldvl_t, vr,
                          &ldvr_t, work, &lwork, rwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            vl_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldvl_t * MAX(1,n) );
            if( vl_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            vr_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldvr_t * MAX(1,n) );
            if( vr_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        
        LAPACKE_zge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_zgeev( &jobvl, &jobvr, &n, a_t, &lda_t, w, vl_t, &ldvl_t, vr_t,
                      &ldvr_t, work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, vl_t, ldvl_t, vl, ldvl );
        }
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, vr_t, ldvr_t, vr, ldvr );
        }
        
        if( LAPACKE_lsame( jobvr, 'v' ) ) {
            LAPACKE_free( vr_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobvl, 'v' ) ) {
            LAPACKE_free( vl_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgeev_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgeev_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* w,
                          lapack_complex_double* vl, lapack_int ldvl,
                          lapack_complex_double* vr, lapack_int ldvr )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* rwork = NULL;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgeev", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    rwork = (double*)LAPACKE_malloc( sizeof(double) * MAX(1,2*n) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zgeev_work( matrix_layout, jobvl, jobvr, n, a, lda, w, vl,
                               ldvl, vr, ldvr, &work_query, lwork, rwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_zgeev_work( matrix_layout, jobvl, jobvr, n, a, lda, w, vl,
                               ldvl, vr, ldvr, work, lwork, rwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( rwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgeev", info );
    }
    return info;
}

lapack_int LAPACKE_spotrf_work( int matrix_layout, char uplo, lapack_int n,
                                float* a, lapack_int lda )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_spotrf( &uplo, &n, a, &lda, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_spotrf_work", info );
            return info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_spo_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_spotrf( &uplo, &n, a_t, &lda_t, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_spo_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_spotrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_spotrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_spotrf( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_spotrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_spo_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    return LAPACKE_spotrf_work( matrix_layout, uplo, n, a, lda );
}

lapack_int LAPACKE_dpotrf_work( int matrix_layout, char uplo, lapack_int n,
                                double* a, lapack_int lda )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dpotrf( &uplo, &n, a, &lda, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_dpotrf_work", info );
            return info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dpo_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_dpotrf( &uplo, &n, a_t, &lda_t, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dpo_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dpotrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dpotrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_dpotrf( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dpotrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dpo_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    return LAPACKE_dpotrf_work( matrix_layout, uplo, n, a, lda );
}

lapack_int LAPACKE_cpotrf_work( int matrix_layout, char uplo, lapack_int n,
                                lapack_complex_float* a, lapack_int lda )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cpotrf( &uplo, &n, a, &lda, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_cpotrf_work", info );
            return info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_cpo_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_cpotrf( &uplo, &n, a_t, &lda_t, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cpo_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cpotrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cpotrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_cpotrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cpotrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cpo_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    return LAPACKE_cpotrf_work( matrix_layout, uplo, n, a, lda );
}

lapack_int LAPACKE_zpotrf_work( int matrix_layout, char uplo, lapack_int n,
                                lapack_complex_double* a, lapack_int lda )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zpotrf( &uplo, &n, a, &lda, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_zpotrf_work", info );
            return info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zpo_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_zpotrf( &uplo, &n, a_t, &lda_t, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zpo_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zpotrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zpotrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_zpotrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda )
{
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zpotrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zpo_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    return LAPACKE_zpotrf_work( matrix_layout, uplo, n, a, lda );
}

lapack_int LAPACKE_sgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                float* a, lapack_int lda, float* tau,
                                float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgeqrf( &m, &n, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_sgeqrf_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sgeqrf( &m, &n, a, &lda_t, tau, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_sge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_sgeqrf( &m, &n, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgeqrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgeqrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgeqrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    
    info = LAPACKE_sgeqrf_work( matrix_layout, m, n, a, lda, tau, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sgeqrf_work( matrix_layout, m, n, a, lda, tau, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgeqrf", info );
    }
    return info;
}

lapack_int LAPACKE_dgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                double* a, lapack_int lda, double* tau,
                                double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgeqrf( &m, &n, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_dgeqrf_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dgeqrf( &m, &n, a, &lda_t, tau, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_dgeqrf( &m, &n, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgeqrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgeqrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgeqrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    
    info = LAPACKE_dgeqrf_work( matrix_layout, m, n, a, lda, tau, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dgeqrf_work( matrix_layout, m, n, a, lda, tau, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgeqrf", info );
    }
    return info;
}

lapack_int LAPACKE_cgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_complex_float* a, lapack_int lda,
                                lapack_complex_float* tau,
                                lapack_complex_float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgeqrf( &m, &n, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_cgeqrf_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cgeqrf( &m, &n, a, &lda_t, tau, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_cge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_cgeqrf( &m, &n, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgeqrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgeqrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgeqrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    
    info = LAPACKE_cgeqrf_work( matrix_layout, m, n, a, lda, tau, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cgeqrf_work( matrix_layout, m, n, a, lda, tau, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgeqrf", info );
    }
    return info;
}

lapack_int LAPACKE_zgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_complex_double* a, lapack_int lda,
                                lapack_complex_double* tau,
                                lapack_complex_double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgeqrf( &m, &n, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -5;
            LAPACKE_xerbla( "LAPACKE_zgeqrf_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zgeqrf( &m, &n, a, &lda_t, tau, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_zgeqrf( &m, &n, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgeqrf_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgeqrf_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgeqrf", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -4;
        }
    }
#endif
    
    info = LAPACKE_zgeqrf_work( matrix_layout, m, n, a, lda, tau, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zgeqrf_work( matrix_layout, m, n, a, lda, tau, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgeqrf", info );
    }
    return info;
}

lapack_int LAPACKE_sorgqr_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int k, float* a, lapack_int lda,
                                const float* tau, float* work,
                                lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sorgqr( &m, &n, &k, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_sorgqr_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sorgqr( &m, &n, &k, a, &lda_t, tau, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_sge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_sorgqr( &m, &n, &k, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sorgqr_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sorgqr_work", info );
    }
    return info;
}

lapack_int LAPACKE_sorgqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sorgqr", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_s_nancheck( k, tau, 1 ) ) {
            return -7;
        }
    }
#endif
    
    info = LAPACKE_sorgqr_work( matrix_layout, m, n, k, a, lda, tau, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sorgqr_work( matrix_layout, m, n, k, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sorgqr", info );
    }
    return info;
}

lapack_int LAPACKE_dorgqr_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int k, double* a, lapack_int lda,
                                const double* tau, double* work,
                                lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dorgqr( &m, &n, &k, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_dorgqr_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dorgqr( &m, &n, &k, a, &lda_t, tau, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_dorgqr( &m, &n, &k, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dorgqr_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dorgqr_work", info );
    }
    return info;
}

lapack_int LAPACKE_dorgqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dorgqr", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_d_nancheck( k, tau, 1 ) ) {
            return -7;
        }
    }
#endif
    
    info = LAPACKE_dorgqr_work( matrix_layout, m, n, k, a, lda, tau, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dorgqr_work( matrix_layout, m, n, k, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dorgqr", info );
    }
    return info;
}

lapack_int LAPACKE_cungqr_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int k, lapack_complex_float* a,
                                lapack_int lda, const lapack_complex_float* tau,
                                lapack_complex_float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cungqr( &m, &n, &k, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_cungqr_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cungqr( &m, &n, &k, a, &lda_t, tau, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_cge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_cungqr( &m, &n, &k, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cungqr_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cungqr_work", info );
    }
    return info;
}

lapack_int LAPACKE_cungqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cungqr", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_c_nancheck( k, tau, 1 ) ) {
            return -7;
        }
    }
#endif
    
    info = LAPACKE_cungqr_work( matrix_layout, m, n, k, a, lda, tau, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cungqr_work( matrix_layout, m, n, k, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cungqr", info );
    }
    return info;
}

lapack_int LAPACKE_zungqr_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int k, lapack_complex_double* a,
                                lapack_int lda,
                                const lapack_complex_double* tau,
                                lapack_complex_double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zungqr( &m, &n, &k, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_zungqr_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zungqr( &m, &n, &k, a, &lda_t, tau, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_zungqr( &m, &n, &k, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zungqr_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zungqr_work", info );
    }
    return info;
}

lapack_int LAPACKE_zungqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zungqr", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_z_nancheck( k, tau, 1 ) ) {
            return -7;
        }
    }
#endif
    
    info = LAPACKE_zungqr_work( matrix_layout, m, n, k, a, lda, tau, &work_query,
                                lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zungqr_work( matrix_layout, m, n, k, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zungqr", info );
    }
    return info;
}

lapack_int LAPACKE_ssyev_work( int matrix_layout, char jobz, char uplo,
                               lapack_int n, float* a, lapack_int lda, float* w,
                               float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_ssyev( &jobz, &uplo, &n, a, &lda, w, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_ssyev_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_ssyev( &jobz, &uplo, &n, a, &lda_t, w, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_ssy_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_ssyev( &jobz, &uplo, &n, a_t, &lda_t, w, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        if ( jobz == 'V' || jobz == 'v' ) {
            LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        } else {
            LAPACKE_ssy_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        }
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_ssyev_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_ssyev_work", info );
    }
    return info;
}

lapack_int LAPACKE_ssyev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          float* a, lapack_int lda, float* w )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_ssyev", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_ssy_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_ssyev_work( matrix_layout, jobz, uplo, n, a, lda, w,
                               &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_ssyev_work( matrix_layout, jobz, uplo, n, a, lda, w, work,
                               lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_ssyev", info );
    }
    return info;
}

lapack_int LAPACKE_dsyev_work( int matrix_layout, char jobz, char uplo,
                               lapack_int n, double* a, lapack_int lda,
                               double* w, double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dsyev( &jobz, &uplo, &n, a, &lda, w, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_dsyev_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dsyev( &jobz, &uplo, &n, a, &lda_t, w, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dsy_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_dsyev( &jobz, &uplo, &n, a_t, &lda_t, w, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        if ( jobz == 'V' || jobz == 'v' ) {
            LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        } else {
            LAPACKE_dsy_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        }
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dsyev_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dsyev_work", info );
    }
    return info;
}

lapack_int LAPACKE_dsyev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          double* a, lapack_int lda, double* w )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dsyev", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dsy_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_dsyev_work( matrix_layout, jobz, uplo, n, a, lda, w,
                               &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dsyev_work( matrix_layout, jobz, uplo, n, a, lda, w, work,
                               lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dsyev", info );
    }
    return info;
}

lapack_int LAPACKE_cheev_work( int matrix_layout, char jobz, char uplo,
                               lapack_int n, lapack_complex_float* a,
                               lapack_int lda, float* w,
                               lapack_complex_float* work, lapack_int lwork,
                               float* rwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cheev( &jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork,
                      &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_cheev_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cheev( &jobz, &uplo, &n, a, &lda_t, w, work, &lwork, rwork,
                          &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_che_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_cheev( &jobz, &uplo, &n, a_t, &lda_t, w, work, &lwork, rwork,
                      &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        if ( jobz == 'V' || jobz == 'v' ) {
            LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        } else {
            LAPACKE_che_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        }
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cheev_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cheev_work", info );
    }
    return info;
}

lapack_int LAPACKE_cheev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_complex_float* a, lapack_int lda, float* w )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* rwork = NULL;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cheev", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_che_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    rwork = (float*)LAPACKE_malloc( sizeof(float) * MAX(1,3*n-2) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cheev_work( matrix_layout, jobz, uplo, n, a, lda, w,
                               &work_query, lwork, rwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_cheev_work( matrix_layout, jobz, uplo, n, a, lda, w, work,
                               lwork, rwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( rwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cheev", info );
    }
    return info;
}

lapack_int LAPACKE_zheev_work( int matrix_layout, char jobz, char uplo,
                               lapack_int n, lapack_complex_double* a,
                               lapack_int lda, double* w,
                               lapack_complex_double* work, lapack_int lwork,
                               double* rwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zheev( &jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork,
                      &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_zheev_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zheev( &jobz, &uplo, &n, a, &lda_t, w, work, &lwork, rwork,
                          &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zhe_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_zheev( &jobz, &uplo, &n, a_t, &lda_t, w, work, &lwork, rwork,
                      &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        if ( jobz == 'V' || jobz == 'v' ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        } else {
            LAPACKE_zhe_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        }
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zheev_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zheev_work", info );
    }
    return info;
}

lapack_int LAPACKE_zheev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_complex_double* a, lapack_int lda, double* w )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* rwork = NULL;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zheev", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zhe_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    rwork = (double*)LAPACKE_malloc( sizeof(double) * MAX(1,3*n-2) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zheev_work( matrix_layout, jobz, uplo, n, a, lda, w,
                               &work_query, lwork, rwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_zheev_work( matrix_layout, jobz, uplo, n, a, lda, w, work,
                               lwork, rwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( rwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zheev", info );
    }
    return info;
}

lapack_int LAPACKE_ssyevd_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, float* a, lapack_int lda,
                                float* w, float* work, lapack_int lwork,
                                lapack_int* iwork, lapack_int liwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_ssyevd( &jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork,
                       &liwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_ssyevd_work", info );
            return info;
        }
        
        if( liwork == -1 || lwork == -1 ) {
            LAPACK_ssyevd( &jobz, &uplo, &n, a, &lda_t, w, work, &lwork, iwork,
                           &liwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_ssy_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_ssyevd( &jobz, &uplo, &n, a_t, &lda_t, w, work, &lwork, iwork,
                       &liwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        if ( jobz == 'V' || jobz == 'v' ) {
            LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        } else {
            LAPACKE_ssy_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        }
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_ssyevd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_ssyevd_work", info );
    }
    return info;
}

lapack_int LAPACKE_ssyevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           float* a, lapack_int lda, float* w )
{
    lapack_int info = 0;
    lapack_int liwork = -1;
    lapack_int lwork = -1;
    lapack_int* iwork = NULL;
    float* work = NULL;
    lapack_int iwork_query;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_ssyevd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_ssy_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_ssyevd_work( matrix_layout, jobz, uplo, n, a, lda, w,
                                &work_query, lwork, &iwork_query, liwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    liwork = iwork_query;
    lwork = (lapack_int)work_query;
    
    iwork = (lapack_int*)LAPACKE_malloc( sizeof(lapack_int) * liwork );
    if( iwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_ssyevd_work( matrix_layout, jobz, uplo, n, a, lda, w, work,
                                lwork, iwork, liwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( iwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_ssyevd", info );
    }
    return info;
}

lapack_int LAPACKE_dsyevd_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, double* a, lapack_int lda,
                                double* w, double* work, lapack_int lwork,
                                lapack_int* iwork, lapack_int liwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dsyevd( &jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork,
                       &liwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_dsyevd_work", info );
            return info;
        }
        
        if( liwork == -1 || lwork == -1 ) {
            LAPACK_dsyevd( &jobz, &uplo, &n, a, &lda_t, w, work, &lwork, iwork,
                           &liwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dsy_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_dsyevd( &jobz, &uplo, &n, a_t, &lda_t, w, work, &lwork, iwork,
                       &liwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        if ( jobz == 'V' || jobz == 'v' ) {
            LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        } else {
            LAPACKE_dsy_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        }
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dsyevd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dsyevd_work", info );
    }
    return info;
}

lapack_int LAPACKE_dsyevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           double* a, lapack_int lda, double* w )
{
    lapack_int info = 0;
    lapack_int liwork = -1;
    lapack_int lwork = -1;
    lapack_int* iwork = NULL;
    double* work = NULL;
    lapack_int iwork_query;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dsyevd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dsy_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_dsyevd_work( matrix_layout, jobz, uplo, n, a, lda, w,
                                &work_query, lwork, &iwork_query, liwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    liwork = iwork_query;
    lwork = (lapack_int)work_query;
    
    iwork = (lapack_int*)LAPACKE_malloc( sizeof(lapack_int) * liwork );
    if( iwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_dsyevd_work( matrix_layout, jobz, uplo, n, a, lda, w, work,
                                lwork, iwork, liwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( iwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dsyevd", info );
    }
    return info;
}

lapack_int LAPACKE_cheevd_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, lapack_complex_float* a,
                                lapack_int lda, float* w,
                                lapack_complex_float* work, lapack_int lwork,
                                float* rwork, lapack_int lrwork,
                                lapack_int* iwork, lapack_int liwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cheevd( &jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork,
                       &lrwork, iwork, &liwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_cheevd_work", info );
            return info;
        }
        
        if( liwork == -1 || lrwork == -1 || lwork == -1 ) {
            LAPACK_cheevd( &jobz, &uplo, &n, a, &lda_t, w, work, &lwork, rwork,
                           &lrwork, iwork, &liwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_che_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_cheevd( &jobz, &uplo, &n, a_t, &lda_t, w, work, &lwork, rwork,
                       &lrwork, iwork, &liwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        if ( jobz == 'V' || jobz == 'v' ) {
            LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        } else { 
            LAPACKE_che_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        }
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cheevd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cheevd_work", info );
    }
    return info;
}

lapack_int LAPACKE_cheevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda, float* w )
{
    lapack_int info = 0;
    lapack_int liwork = -1;
    lapack_int lrwork = -1;
    lapack_int lwork = -1;
    lapack_int* iwork = NULL;
    float* rwork = NULL;
    lapack_complex_float* work = NULL;
    lapack_int iwork_query;
    float rwork_query;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cheevd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_che_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_cheevd_work( matrix_layout, jobz, uplo, n, a, lda, w,
                                &work_query, lwork, &rwork_query, lrwork,
                                &iwork_query, liwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    liwork = iwork_query;
    lrwork = (lapack_int)rwork_query;
    lwork = LAPACK_C2INT( work_query );
    
    iwork = (lapack_int*)LAPACKE_malloc( sizeof(lapack_int) * liwork );
    if( iwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    rwork = (float*)LAPACKE_malloc( sizeof(float) * lrwork );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_2;
    }
    
    info = LAPACKE_cheevd_work( matrix_layout, jobz, uplo, n, a, lda, w, work,
                                lwork, rwork, lrwork, iwork, liwork );
    
    LAPACKE_free( work );
exit_level_2:
    LAPACKE_free( rwork );
exit_level_1:
    LAPACKE_free( iwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cheevd", info );
    }
    return info;
}

lapack_int LAPACKE_zheevd_work( int matrix_layout, char jobz, char uplo,
                                lapack_int n, lapack_complex_double* a,
                                lapack_int lda, double* w,
                                lapack_complex_double* work, lapack_int lwork,
                                double* rwork, lapack_int lrwork,
                                lapack_int* iwork, lapack_int liwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zheevd( &jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork,
                       &lrwork, iwork, &liwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_zheevd_work", info );
            return info;
        }
        
        if( liwork == -1 || lrwork == -1 || lwork == -1 ) {
            LAPACK_zheevd( &jobz, &uplo, &n, a, &lda_t, w, work, &lwork, rwork,
                           &lrwork, iwork, &liwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zhe_trans( matrix_layout, uplo, n, a, lda, a_t, lda_t );
        
        LAPACK_zheevd( &jobz, &uplo, &n, a_t, &lda_t, w, work, &lwork, rwork,
                       &lrwork, iwork, &liwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        if ( jobz == 'V' || jobz == 'v' ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        } else { 
            LAPACKE_zhe_trans( LAPACK_COL_MAJOR, uplo, n, a_t, lda_t, a, lda );
        }
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zheevd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zheevd_work", info );
    }
    return info;
}

lapack_int LAPACKE_zheevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda, double* w )
{
    lapack_int info = 0;
    lapack_int liwork = -1;
    lapack_int lrwork = -1;
    lapack_int lwork = -1;
    lapack_int* iwork = NULL;
    double* rwork = NULL;
    lapack_complex_double* work = NULL;
    lapack_int iwork_query;
    double rwork_query;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zheevd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zhe_nancheck( matrix_layout, uplo, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_zheevd_work( matrix_layout, jobz, uplo, n, a, lda, w,
                                &work_query, lwork, &rwork_query, lrwork,
                                &iwork_query, liwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    liwork = iwork_query;
    lrwork = (lapack_int)rwork_query;
    lwork = LAPACK_Z2INT( work_query );
    
    iwork = (lapack_int*)LAPACKE_malloc( sizeof(lapack_int) * liwork );
    if( iwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    rwork = (double*)LAPACKE_malloc( sizeof(double) * lrwork );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_2;
    }
    
    info = LAPACKE_zheevd_work( matrix_layout, jobz, uplo, n, a, lda, w, work,
                                lwork, rwork, lrwork, iwork, liwork );
    
    LAPACKE_free( work );
exit_level_2:
    LAPACKE_free( rwork );
exit_level_1:
    LAPACKE_free( iwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zheevd", info );
    }
    return info;
}

lapack_int LAPACKE_sgesvd_work( int matrix_layout, char jobu, char jobvt,
                                lapack_int m, lapack_int n, float* a,
                                lapack_int lda, float* s, float* u,
                                lapack_int ldu, float* vt, lapack_int ldvt,
                                float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgesvd( &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                       work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int nrows_u = ( LAPACKE_lsame( jobu, 'a' ) ||
                             LAPACKE_lsame( jobu, 's' ) ) ? m : 1;
        lapack_int ncols_u = LAPACKE_lsame( jobu, 'a' ) ? m :
                             ( LAPACKE_lsame( jobu, 's' ) ? MIN(m,n) : 1);
        lapack_int nrows_vt = LAPACKE_lsame( jobvt, 'a' ) ? n :
                              ( LAPACKE_lsame( jobvt, 's' ) ? MIN(m,n) : 1);
        lapack_int ncols_vt = ( LAPACKE_lsame( jobvt, 'a' ) ||
                               LAPACKE_lsame( jobvt, 's' ) ) ? n : 1;
        lapack_int lda_t = MAX(1,m);
        lapack_int ldu_t = MAX(1,nrows_u);
        lapack_int ldvt_t = MAX(1,nrows_vt);
        float* a_t = NULL;
        float* u_t = NULL;
        float* vt_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_sgesvd_work", info );
            return info;
        }
        if( ldu < ncols_u ) {
            info = -10;
            LAPACKE_xerbla( "LAPACKE_sgesvd_work", info );
            return info;
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
	if( ldvt < ncols_vt ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_sgesvd_work", info );
            return info;
        }
        }
        
        if( lwork == -1 ) {
            LAPACK_sgesvd( &jobu, &jobvt, &m, &n, a, &lda_t, s, u, &ldu_t, vt,
                           &ldvt_t, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            u_t = (float*)
                LAPACKE_malloc( sizeof(float) * ldu_t * MAX(1,ncols_u) );
            if( u_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            vt_t = (float*)LAPACKE_malloc( sizeof(float) * ldvt_t * MAX(1,n) );
            if( vt_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        
        LAPACKE_sge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_sgesvd( &jobu, &jobvt, &m, &n, a_t, &lda_t, s, u_t, &ldu_t, vt_t,
                       &ldvt_t, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            LAPACKE_sge_trans( LAPACK_COL_MAJOR, nrows_u, ncols_u, u_t, ldu_t,
                               u, ldu );
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            LAPACKE_sge_trans( LAPACK_COL_MAJOR, nrows_vt, n, vt_t, ldvt_t, vt,
                               ldvt );
        }
        
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            LAPACKE_free( vt_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            LAPACKE_free( u_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgesvd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgesvd_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, float* a, lapack_int lda,
                           float* s, float* u, lapack_int ldu, float* vt,
                           lapack_int ldvt, float* superb )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    lapack_int i;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgesvd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -6;
        }
    }
#endif
    
    info = LAPACKE_sgesvd_work( matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                                ldu, vt, ldvt, &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sgesvd_work( matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                                ldu, vt, ldvt, work, lwork );
    
    for( i=0; i<MIN(m,n)-1; i++ ) {
        superb[i] = work[i+1];
    }
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgesvd", info );
    }
    return info;
}

lapack_int LAPACKE_dgesvd_work( int matrix_layout, char jobu, char jobvt,
                                lapack_int m, lapack_int n, double* a,
                                lapack_int lda, double* s, double* u,
                                lapack_int ldu, double* vt, lapack_int ldvt,
                                double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgesvd( &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                       work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int nrows_u = ( LAPACKE_lsame( jobu, 'a' ) ||
                             LAPACKE_lsame( jobu, 's' ) ) ? m : 1;
        lapack_int ncols_u = LAPACKE_lsame( jobu, 'a' ) ? m :
                             ( LAPACKE_lsame( jobu, 's' ) ? MIN(m,n) : 1);
        lapack_int nrows_vt = LAPACKE_lsame( jobvt, 'a' ) ? n :
                              ( LAPACKE_lsame( jobvt, 's' ) ? MIN(m,n) : 1);
        lapack_int ncols_vt = ( LAPACKE_lsame( jobvt, 'a' ) ||
                               LAPACKE_lsame( jobvt, 's' ) ) ? n : 1;
        lapack_int lda_t = MAX(1,m);
        lapack_int ldu_t = MAX(1,nrows_u);
        lapack_int ldvt_t = MAX(1,nrows_vt);
        double* a_t = NULL;
        double* u_t = NULL;
        double* vt_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_dgesvd_work", info );
            return info;
        }
        if( ldu < ncols_u ) {
            info = -10;
            LAPACKE_xerbla( "LAPACKE_dgesvd_work", info );
            return info;
        }
	if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
        if( ldvt < ncols_vt ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_dgesvd_work", info );
            return info;
        }
        }
        
        if( lwork == -1 ) {
            LAPACK_dgesvd( &jobu, &jobvt, &m, &n, a, &lda_t, s, u, &ldu_t, vt,
                           &ldvt_t, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            u_t = (double*)
                LAPACKE_malloc( sizeof(double) * ldu_t * MAX(1,ncols_u) );
            if( u_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            vt_t = (double*)
                LAPACKE_malloc( sizeof(double) * ldvt_t * MAX(1,n) );
            if( vt_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        
        LAPACKE_dge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_dgesvd( &jobu, &jobvt, &m, &n, a_t, &lda_t, s, u_t, &ldu_t, vt_t,
                       &ldvt_t, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            LAPACKE_dge_trans( LAPACK_COL_MAJOR, nrows_u, ncols_u, u_t, ldu_t,
                               u, ldu );
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            LAPACKE_dge_trans( LAPACK_COL_MAJOR, nrows_vt, n, vt_t, ldvt_t, vt,
                               ldvt );
        }
        
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            LAPACKE_free( vt_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            LAPACKE_free( u_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgesvd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgesvd_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double* s, double* u, lapack_int ldu,
                           double* vt, lapack_int ldvt, double* superb )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    lapack_int i;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgesvd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -6;
        }
    }
#endif
    
    info = LAPACKE_dgesvd_work( matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                                ldu, vt, ldvt, &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dgesvd_work( matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                                ldu, vt, ldvt, work, lwork );
    
    for( i=0; i<MIN(m,n)-1; i++ ) {
        superb[i] = work[i+1];
    }
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgesvd", info );
    }
    return info;
}

lapack_int LAPACKE_cgesvd_work( int matrix_layout, char jobu, char jobvt,
                                lapack_int m, lapack_int n,
                                lapack_complex_float* a, lapack_int lda,
                                float* s, lapack_complex_float* u,
                                lapack_int ldu, lapack_complex_float* vt,
                                lapack_int ldvt, lapack_complex_float* work,
                                lapack_int lwork, float* rwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgesvd( &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                       work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int nrows_u = ( LAPACKE_lsame( jobu, 'a' ) ||
                             LAPACKE_lsame( jobu, 's' ) ) ? m : 1;
        lapack_int ncols_u = LAPACKE_lsame( jobu, 'a' ) ? m :
                             ( LAPACKE_lsame( jobu, 's' ) ? MIN(m,n) : 1);
        lapack_int nrows_vt = LAPACKE_lsame( jobvt, 'a' ) ? n :
                              ( LAPACKE_lsame( jobvt, 's' ) ? MIN(m,n) : 1);
        lapack_int ncols_vt = ( LAPACKE_lsame( jobvt, 'a' ) ||
                               LAPACKE_lsame( jobvt, 's' ) ) ? n : 1;
        lapack_int lda_t = MAX(1,m);
        lapack_int ldu_t = MAX(1,nrows_u);
        lapack_int ldvt_t = MAX(1,nrows_vt);
        lapack_complex_float* a_t = NULL;
        lapack_complex_float* u_t = NULL;
        lapack_complex_float* vt_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_cgesvd_work", info );
            return info;
        }
        if( ldu < ncols_u ) {
            info = -10;
            LAPACKE_xerbla( "LAPACKE_cgesvd_work", info );
            return info;
        }
	if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
	if( ldvt < ncols_vt ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_cgesvd_work", info );
            return info;
        }
        }
        
        if( lwork == -1 ) {
            LAPACK_cgesvd( &jobu, &jobvt, &m, &n, a, &lda_t, s, u, &ldu_t, vt,
                           &ldvt_t, work, &lwork, rwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            u_t = (lapack_complex_float*)
                LAPACKE_malloc( sizeof(lapack_complex_float) *
                                ldu_t * MAX(1,ncols_u) );
            if( u_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            vt_t = (lapack_complex_float*)
                LAPACKE_malloc( sizeof(lapack_complex_float) *
                                ldvt_t * MAX(1,n) );
            if( vt_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        
        LAPACKE_cge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_cgesvd( &jobu, &jobvt, &m, &n, a_t, &lda_t, s, u_t, &ldu_t, vt_t,
                       &ldvt_t, work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            LAPACKE_cge_trans( LAPACK_COL_MAJOR, nrows_u, ncols_u, u_t, ldu_t,
                               u, ldu );
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            LAPACKE_cge_trans( LAPACK_COL_MAJOR, nrows_vt, n, vt_t, ldvt_t, vt,
                               ldvt );
        }
        
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            LAPACKE_free( vt_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            LAPACKE_free( u_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgesvd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgesvd_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float* s, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* vt,
                           lapack_int ldvt, float* superb )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* rwork = NULL;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    lapack_int i;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgesvd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -6;
        }
    }
#endif
    
    rwork = (float*)LAPACKE_malloc( sizeof(float) * MAX(1,5*MIN(m,n)) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cgesvd_work( matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                                ldu, vt, ldvt, &work_query, lwork, rwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_cgesvd_work( matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                                ldu, vt, ldvt, work, lwork, rwork );
    
    for( i=0; i<MIN(m,n)-1; i++ ) {
        superb[i] = rwork[i];
    }
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( rwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgesvd", info );
    }
    return info;
}

lapack_int LAPACKE_zgesvd_work( int matrix_layout, char jobu, char jobvt,
                                lapack_int m, lapack_int n,
                                lapack_complex_double* a, lapack_int lda,
                                double* s, lapack_complex_double* u,
                                lapack_int ldu, lapack_complex_double* vt,
                                lapack_int ldvt, lapack_complex_double* work,
                                lapack_int lwork, double* rwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgesvd( &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                       work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int nrows_u = ( LAPACKE_lsame( jobu, 'a' ) ||
                             LAPACKE_lsame( jobu, 's' ) ) ? m : 1;
        lapack_int ncols_u = LAPACKE_lsame( jobu, 'a' ) ? m :
                             ( LAPACKE_lsame( jobu, 's' ) ? MIN(m,n) : 1);
        lapack_int nrows_vt = LAPACKE_lsame( jobvt, 'a' ) ? n :
                              ( LAPACKE_lsame( jobvt, 's' ) ? MIN(m,n) : 1);
        lapack_int ncols_vt = ( LAPACKE_lsame( jobvt, 'a' ) ||
                               LAPACKE_lsame( jobvt, 's' ) ) ? n : 1;
        lapack_int lda_t = MAX(1,m);
        lapack_int ldu_t = MAX(1,nrows_u);
        lapack_int ldvt_t = MAX(1,nrows_vt);
        lapack_complex_double* a_t = NULL;
        lapack_complex_double* u_t = NULL;
        lapack_complex_double* vt_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_zgesvd_work", info );
            return info;
        }
        if( ldu < ncols_u ) {
            info = -10;
            LAPACKE_xerbla( "LAPACKE_zgesvd_work", info );
            return info;
        }
	if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
	if( ldvt < ncols_vt ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_zgesvd_work", info );
            return info;
        }
        }
        
        if( lwork == -1 ) {
            LAPACK_zgesvd( &jobu, &jobvt, &m, &n, a, &lda_t, s, u, &ldu_t, vt,
                           &ldvt_t, work, &lwork, rwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            u_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldu_t * MAX(1,ncols_u) );
            if( u_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            vt_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldvt_t * MAX(1,n) );
            if( vt_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_2;
            }
        }
        
        LAPACKE_zge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        
        LAPACK_zgesvd( &jobu, &jobvt, &m, &n, a_t, &lda_t, s, u_t, &ldu_t, vt_t,
                       &ldvt_t, work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, nrows_u, ncols_u, u_t, ldu_t,
                               u, ldu );
        }
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, nrows_vt, n, vt_t, ldvt_t, vt,
                               ldvt );
        }
        
        if( LAPACKE_lsame( jobvt, 'a' ) || LAPACKE_lsame( jobvt, 's' ) ) {
            LAPACKE_free( vt_t );
        }
exit_level_2:
        if( LAPACKE_lsame( jobu, 'a' ) || LAPACKE_lsame( jobu, 's' ) ) {
            LAPACKE_free( u_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgesvd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgesvd_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double* s, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* vt,
                           lapack_int ldvt, double* superb )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* rwork = NULL;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    lapack_int i;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgesvd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -6;
        }
    }
#endif
    
    rwork = (double*)LAPACKE_malloc( sizeof(double) * MAX(1,5*MIN(m,n)) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zgesvd_work( matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                                ldu, vt, ldvt, &work_query, lwork, rwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_zgesvd_work( matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                                ldu, vt, ldvt, work, lwork, rwork );
    
    for( i=0; i<MIN(m,n)-1; i++ ) {
        superb[i] = rwork[i];
    }
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( rwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgesvd", info );
    }
    return info;
}

lapack_int LAPACKE_sgels_work( int matrix_layout, char trans, lapack_int m,
                               lapack_int n, lapack_int nrhs, float* a,
                               lapack_int lda, float* b, lapack_int ldb,
                               float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgels( &trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
                      &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        float* a_t = NULL;
        float* b_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_sgels_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -9;
            LAPACKE_xerbla( "LAPACKE_sgels_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sgels( &trans, &m, &n, &nrhs, a, &lda_t, b, &ldb_t, work,
                          &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (float*)LAPACKE_malloc( sizeof(float) * ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_sge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_sge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_sgels( &trans, &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, work,
                      &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgels_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgels_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs, float* a,
                          lapack_int lda, float* b, lapack_int ldb )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgels", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -6;
        }
        if( LAPACKE_sge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -8;
        }
    }
#endif
    
    info = LAPACKE_sgels_work( matrix_layout, trans, m, n, nrhs, a, lda, b, ldb,
                               &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sgels_work( matrix_layout, trans, m, n, nrhs, a, lda, b, ldb,
                               work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgels", info );
    }
    return info;
}

lapack_int LAPACKE_dgels_work( int matrix_layout, char trans, lapack_int m,
                               lapack_int n, lapack_int nrhs, double* a,
                               lapack_int lda, double* b, lapack_int ldb,
                               double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgels( &trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
                      &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        double* a_t = NULL;
        double* b_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_dgels_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -9;
            LAPACKE_xerbla( "LAPACKE_dgels_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dgels( &trans, &m, &n, &nrhs, a, &lda_t, b, &ldb_t, work,
                          &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (double*)LAPACKE_malloc( sizeof(double) * ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_dge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_dge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_dgels( &trans, &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, work,
                      &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgels_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgels_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs, double* a,
                          lapack_int lda, double* b, lapack_int ldb )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgels", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -6;
        }
        if( LAPACKE_dge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -8;
        }
    }
#endif
    
    info = LAPACKE_dgels_work( matrix_layout, trans, m, n, nrhs, a, lda, b, ldb,
                               &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dgels_work( matrix_layout, trans, m, n, nrhs, a, lda, b, ldb,
                               work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgels", info );
    }
    return info;
}

lapack_int LAPACKE_cgels_work( int matrix_layout, char trans, lapack_int m,
                               lapack_int n, lapack_int nrhs,
                               lapack_complex_float* a, lapack_int lda,
                               lapack_complex_float* b, lapack_int ldb,
                               lapack_complex_float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgels( &trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
                      &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        lapack_complex_float* a_t = NULL;
        lapack_complex_float* b_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_cgels_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -9;
            LAPACKE_xerbla( "LAPACKE_cgels_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cgels( &trans, &m, &n, &nrhs, a, &lda_t, b, &ldb_t, work,
                          &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) *
                            ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_cge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_cge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_cgels( &trans, &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, work,
                      &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgels_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgels_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* b, lapack_int ldb )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgels", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -6;
        }
        if( LAPACKE_cge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -8;
        }
    }
#endif
    
    info = LAPACKE_cgels_work( matrix_layout, trans, m, n, nrhs, a, lda, b, ldb,
                               &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cgels_work( matrix_layout, trans, m, n, nrhs, a, lda, b, ldb,
                               work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgels", info );
    }
    return info;
}

lapack_int LAPACKE_zgels_work( int matrix_layout, char trans, lapack_int m,
                               lapack_int n, lapack_int nrhs,
                               lapack_complex_double* a, lapack_int lda,
                               lapack_complex_double* b, lapack_int ldb,
                               lapack_complex_double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgels( &trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
                      &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        lapack_complex_double* a_t = NULL;
        lapack_complex_double* b_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_zgels_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -9;
            LAPACKE_xerbla( "LAPACKE_zgels_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zgels( &trans, &m, &n, &nrhs, a, &lda_t, b, &ldb_t, work,
                          &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) *
                            ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_zge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_zge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_zgels( &trans, &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, work,
                      &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgels_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgels_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* b, lapack_int ldb )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgels", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -6;
        }
        if( LAPACKE_zge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -8;
        }
    }
#endif
    
    info = LAPACKE_zgels_work( matrix_layout, trans, m, n, nrhs, a, lda, b, ldb,
                               &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zgels_work( matrix_layout, trans, m, n, nrhs, a, lda, b, ldb,
                               work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgels", info );
    }
    return info;
}

lapack_int LAPACKE_sgelsy_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, float* a, lapack_int lda,
                                float* b, lapack_int ldb, lapack_int* jpvt,
                                float rcond, lapack_int* rank, float* work,
                                lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgelsy( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank,
                       work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        float* a_t = NULL;
        float* b_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_sgelsy_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_sgelsy_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sgelsy( &m, &n, &nrhs, a, &lda_t, b, &ldb_t, jpvt, &rcond,
                           rank, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (float*)LAPACKE_malloc( sizeof(float) * ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_sge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_sge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_sgelsy( &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, jpvt, &rcond,
                       rank, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgelsy_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgelsy_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, lapack_int* jpvt, float rcond,
                           lapack_int* rank )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgelsy", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_sge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -7;
        }
        if( LAPACKE_s_nancheck( 1, &rcond, 1 ) ) {
            return -10;
        }
    }
#endif
    
    info = LAPACKE_sgelsy_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt,
                                rcond, rank, &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sgelsy_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt,
                                rcond, rank, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgelsy", info );
    }
    return info;
}

lapack_int LAPACKE_dgelsy_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, double* a, lapack_int lda,
                                double* b, lapack_int ldb, lapack_int* jpvt,
                                double rcond, lapack_int* rank, double* work,
                                lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgelsy( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank,
                       work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        double* a_t = NULL;
        double* b_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_dgelsy_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_dgelsy_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dgelsy( &m, &n, &nrhs, a, &lda_t, b, &ldb_t, jpvt, &rcond,
                           rank, work, &lwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (double*)LAPACKE_malloc( sizeof(double) * ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_dge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_dge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_dgelsy( &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, jpvt, &rcond,
                       rank, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgelsy_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgelsy_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, lapack_int* jpvt,
                           double rcond, lapack_int* rank )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgelsy", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_dge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -7;
        }
        if( LAPACKE_d_nancheck( 1, &rcond, 1 ) ) {
            return -10;
        }
    }
#endif
    
    info = LAPACKE_dgelsy_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt,
                                rcond, rank, &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dgelsy_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt,
                                rcond, rank, work, lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgelsy", info );
    }
    return info;
}

lapack_int LAPACKE_cgelsy_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, lapack_complex_float* a,
                                lapack_int lda, lapack_complex_float* b,
                                lapack_int ldb, lapack_int* jpvt, float rcond,
                                lapack_int* rank, lapack_complex_float* work,
                                lapack_int lwork, float* rwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgelsy( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank,
                       work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        lapack_complex_float* a_t = NULL;
        lapack_complex_float* b_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_cgelsy_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_cgelsy_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cgelsy( &m, &n, &nrhs, a, &lda_t, b, &ldb_t, jpvt, &rcond,
                           rank, work, &lwork, rwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) *
                            ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_cge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_cge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_cgelsy( &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, jpvt, &rcond,
                       rank, work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgelsy_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgelsy_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_int* jpvt, float rcond,
                           lapack_int* rank )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* rwork = NULL;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgelsy", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_cge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -7;
        }
        if( LAPACKE_s_nancheck( 1, &rcond, 1 ) ) {
            return -10;
        }
    }
#endif
    
    rwork = (float*)LAPACKE_malloc( sizeof(float) * MAX(1,2*n) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cgelsy_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt,
                                rcond, rank, &work_query, lwork, rwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_cgelsy_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt,
                                rcond, rank, work, lwork, rwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( rwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgelsy", info );
    }
    return info;
}

lapack_int LAPACKE_zgelsy_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, lapack_complex_double* a,
                                lapack_int lda, lapack_complex_double* b,
                                lapack_int ldb, lapack_int* jpvt, double rcond,
                                lapack_int* rank, lapack_complex_double* work,
                                lapack_int lwork, double* rwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgelsy( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank,
                       work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        lapack_complex_double* a_t = NULL;
        lapack_complex_double* b_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_zgelsy_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_zgelsy_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zgelsy( &m, &n, &nrhs, a, &lda_t, b, &ldb_t, jpvt, &rcond,
                           rank, work, &lwork, rwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) *
                            ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_zge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_zge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_zgelsy( &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, jpvt, &rcond,
                       rank, work, &lwork, rwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgelsy_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgelsy_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_int* jpvt, double rcond,
                           lapack_int* rank )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* rwork = NULL;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgelsy", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_zge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -7;
        }
        if( LAPACKE_d_nancheck( 1, &rcond, 1 ) ) {
            return -10;
        }
    }
#endif
    
    rwork = (double*)LAPACKE_malloc( sizeof(double) * MAX(1,2*n) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zgelsy_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt,
                                rcond, rank, &work_query, lwork, rwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_zgelsy_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt,
                                rcond, rank, work, lwork, rwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( rwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgelsy", info );
    }
    return info;
}

lapack_int LAPACKE_sgelsd_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, float* a, lapack_int lda,
                                float* b, lapack_int ldb, float* s, float rcond,
                                lapack_int* rank, float* work, lapack_int lwork,
                                lapack_int* iwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgelsd( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work,
                       &lwork, iwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        float* a_t = NULL;
        float* b_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_sgelsd_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_sgelsd_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sgelsd( &m, &n, &nrhs, a, &lda_t, b, &ldb_t, s, &rcond, rank,
                           work, &lwork, iwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (float*)LAPACKE_malloc( sizeof(float) * ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_sge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_sge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_sgelsd( &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, s, &rcond, rank,
                       work, &lwork, iwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgelsd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgelsd_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    
    lapack_int liwork;
    lapack_int* iwork = NULL;
    float* work = NULL;
    lapack_int iwork_query;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgelsd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_sge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -7;
        }
        if( LAPACKE_s_nancheck( 1, &rcond, 1 ) ) {
            return -10;
        }
    }
#endif
    
    info = LAPACKE_sgelsd_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, s,
                                rcond, rank, &work_query, lwork, &iwork_query );
    if( info != 0 ) {
        goto exit_level_0;
    }
    liwork = iwork_query;
    lwork = (lapack_int)work_query;
    
    iwork = (lapack_int*)LAPACKE_malloc( sizeof(lapack_int) * liwork );
    if( iwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_sgelsd_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, s,
                                rcond, rank, work, lwork, iwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( iwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgelsd", info );
    }
    return info;
}

lapack_int LAPACKE_dgelsd_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, double* a, lapack_int lda,
                                double* b, lapack_int ldb, double* s,
                                double rcond, lapack_int* rank, double* work,
                                lapack_int lwork, lapack_int* iwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgelsd( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work,
                       &lwork, iwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        double* a_t = NULL;
        double* b_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_dgelsd_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_dgelsd_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dgelsd( &m, &n, &nrhs, a, &lda_t, b, &ldb_t, s, &rcond, rank,
                           work, &lwork, iwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (double*)LAPACKE_malloc( sizeof(double) * ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_dge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_dge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_dgelsd( &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, s, &rcond, rank,
                       work, &lwork, iwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgelsd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgelsd_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* s, double rcond,
                           lapack_int* rank )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    
    lapack_int liwork;
    lapack_int* iwork = NULL;
    double* work = NULL;
    lapack_int iwork_query;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgelsd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_dge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -7;
        }
        if( LAPACKE_d_nancheck( 1, &rcond, 1 ) ) {
            return -10;
        }
    }
#endif
    
    info = LAPACKE_dgelsd_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, s,
                                rcond, rank, &work_query, lwork, &iwork_query );
    if( info != 0 ) {
        goto exit_level_0;
    }
    liwork = iwork_query;
    lwork = (lapack_int)work_query;
    
    iwork = (lapack_int*)LAPACKE_malloc( sizeof(lapack_int) * liwork );
    if( iwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_dgelsd_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, s,
                                rcond, rank, work, lwork, iwork );
    
    LAPACKE_free( work );
exit_level_1:
    LAPACKE_free( iwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgelsd", info );
    }
    return info;
}

lapack_int LAPACKE_cgelsd_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, lapack_complex_float* a,
                                lapack_int lda, lapack_complex_float* b,
                                lapack_int ldb, float* s, float rcond,
                                lapack_int* rank, lapack_complex_float* work,
                                lapack_int lwork, float* rwork,
                                lapack_int* iwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgelsd( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work,
                       &lwork, rwork, iwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        lapack_complex_float* a_t = NULL;
        lapack_complex_float* b_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_cgelsd_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_cgelsd_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cgelsd( &m, &n, &nrhs, a, &lda_t, b, &ldb_t, s, &rcond, rank,
                           work, &lwork, rwork, iwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) *
                            ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_cge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_cge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_cgelsd( &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, s, &rcond, rank,
                       work, &lwork, rwork, iwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgelsd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgelsd_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    
    lapack_int liwork;
    lapack_int lrwork;
    lapack_int* iwork = NULL;
    float* rwork = NULL;
    lapack_complex_float* work = NULL;
    lapack_int iwork_query;
    float rwork_query;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgelsd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_cge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -7;
        }
        if( LAPACKE_s_nancheck( 1, &rcond, 1 ) ) {
            return -10;
        }
    }
#endif
    
    info = LAPACKE_cgelsd_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, s,
                                rcond, rank, &work_query, lwork, &rwork_query,
                                &iwork_query );
    if( info != 0 ) {
        goto exit_level_0;
    }
    liwork = iwork_query;
    lrwork = (lapack_int)rwork_query;
    lwork = LAPACK_C2INT( work_query );
    
    iwork = (lapack_int*)LAPACKE_malloc( sizeof(lapack_int) * liwork );
    if( iwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    rwork = (float*)LAPACKE_malloc( sizeof(float) * lrwork );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_2;
    }
    
    info = LAPACKE_cgelsd_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, s,
                                rcond, rank, work, lwork, rwork, iwork );
    
    LAPACKE_free( work );
exit_level_2:
    LAPACKE_free( rwork );
exit_level_1:
    LAPACKE_free( iwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgelsd", info );
    }
    return info;
}

lapack_int LAPACKE_zgelsd_work( int matrix_layout, lapack_int m, lapack_int n,
                                lapack_int nrhs, lapack_complex_double* a,
                                lapack_int lda, lapack_complex_double* b,
                                lapack_int ldb, double* s, double rcond,
                                lapack_int* rank, lapack_complex_double* work,
                                lapack_int lwork, double* rwork,
                                lapack_int* iwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgelsd( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work,
                       &lwork, rwork, iwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,m);
        lapack_int ldb_t = MAX(1,MAX(m,n));
        lapack_complex_double* a_t = NULL;
        lapack_complex_double* b_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_zgelsd_work", info );
            return info;
        }
        if( ldb < nrhs ) {
            info = -8;
            LAPACKE_xerbla( "LAPACKE_zgelsd_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zgelsd( &m, &n, &nrhs, a, &lda_t, b, &ldb_t, s, &rcond, rank,
                           work, &lwork, rwork, iwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        b_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) *
                            ldb_t * MAX(1,nrhs) );
        if( b_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_1;
        }
        
        LAPACKE_zge_trans( matrix_layout, m, n, a, lda, a_t, lda_t );
        LAPACKE_zge_trans( matrix_layout, MAX(m,n), nrhs, b, ldb, b_t, ldb_t );
        
        LAPACK_zgelsd( &m, &n, &nrhs, a_t, &lda_t, b_t, &ldb_t, s, &rcond, rank,
                       work, &lwork, rwork, iwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda );
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, MAX(m,n), nrhs, b_t, ldb_t, b,
                           ldb );
        
        LAPACKE_free( b_t );
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgelsd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgelsd_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double* s, double rcond,
                           lapack_int* rank )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    
    lapack_int liwork;
    lapack_int lrwork;
    lapack_int* iwork = NULL;
    double* rwork = NULL;
    lapack_complex_double* work = NULL;
    lapack_int iwork_query;
    double rwork_query;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgelsd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, m, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_zge_nancheck( matrix_layout, MAX(m,n), nrhs, b, ldb ) ) {
            return -7;
        }
        if( LAPACKE_d_nancheck( 1, &rcond, 1 ) ) {
            return -10;
        }
    }
#endif
    
    info = LAPACKE_zgelsd_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, s,
                                rcond, rank, &work_query, lwork, &rwork_query,
                                &iwork_query );
    if( info != 0 ) {
        goto exit_level_0;
    }
    liwork = iwork_query;
    lrwork = (lapack_int)rwork_query;
    lwork = LAPACK_Z2INT( work_query );
    
    iwork = (lapack_int*)LAPACKE_malloc( sizeof(lapack_int) * liwork );
    if( iwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    rwork = (double*)LAPACKE_malloc( sizeof(double) * lrwork );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_2;
    }
    
    info = LAPACKE_zgelsd_work( matrix_layout, m, n, nrhs, a, lda, b, ldb, s,
                                rcond, rank, work, lwork, rwork, iwork );
    
    LAPACKE_free( work );
exit_level_2:
    LAPACKE_free( rwork );
exit_level_1:
    LAPACKE_free( iwork );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgelsd", info );
    }
    return info;
}

lapack_int LAPACKE_sgees_work( int matrix_layout, char jobvs, char sort,
                               LAPACK_S_SELECT2 select, lapack_int n, float* a,
                               lapack_int lda, lapack_int* sdim, float* wr,
                               float* wi, float* vs, lapack_int ldvs,
                               float* work, lapack_int lwork,
                               lapack_logical* bwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgees( &jobvs, &sort, select, &n, a, &lda, sdim, wr, wi, vs,
                      &ldvs, work, &lwork, bwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldvs_t = MAX(1,n);
        float* a_t = NULL;
        float* vs_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_sgees_work", info );
            return info;
        }
        if( ldvs < n ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_sgees_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sgees( &jobvs, &sort, select, &n, a, &lda_t, sdim, wr, wi,
                          vs, &ldvs_t, work, &lwork, bwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            vs_t = (float*)LAPACKE_malloc( sizeof(float) * ldvs_t * MAX(1,n) );
            if( vs_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        
        LAPACKE_sge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_sgees( &jobvs, &sort, select, &n, a_t, &lda_t, sdim, wr, wi,
                      vs_t, &ldvs_t, work, &lwork, bwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, vs_t, ldvs_t, vs, ldvs );
        }
        
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            LAPACKE_free( vs_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgees_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgees_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_S_SELECT2 select, lapack_int n, float* a,
                          lapack_int lda, lapack_int* sdim, float* wr,
                          float* wi, float* vs, lapack_int ldvs )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_logical* bwork = NULL;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgees", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -6;
        }
    }
#endif
    
    if( LAPACKE_lsame( sort, 's' ) ) {
        bwork = (lapack_logical*)
            LAPACKE_malloc( sizeof(lapack_logical) * MAX(1,n) );
        if( bwork == NULL ) {
            info = LAPACK_WORK_MEMORY_ERROR;
            goto exit_level_0;
        }
    }
    
    info = LAPACKE_sgees_work( matrix_layout, jobvs, sort, select, n, a, lda,
                               sdim, wr, wi, vs, ldvs, &work_query, lwork,
                               bwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_sgees_work( matrix_layout, jobvs, sort, select, n, a, lda,
                               sdim, wr, wi, vs, ldvs, work, lwork, bwork );
    
    LAPACKE_free( work );
exit_level_1:
    if( LAPACKE_lsame( sort, 's' ) ) {
        LAPACKE_free( bwork );
    }
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgees", info );
    }
    return info;
}

lapack_int LAPACKE_dgees_work( int matrix_layout, char jobvs, char sort,
                               LAPACK_D_SELECT2 select, lapack_int n, double* a,
                               lapack_int lda, lapack_int* sdim, double* wr,
                               double* wi, double* vs, lapack_int ldvs,
                               double* work, lapack_int lwork,
                               lapack_logical* bwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgees( &jobvs, &sort, select, &n, a, &lda, sdim, wr, wi, vs,
                      &ldvs, work, &lwork, bwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldvs_t = MAX(1,n);
        double* a_t = NULL;
        double* vs_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_dgees_work", info );
            return info;
        }
        if( ldvs < n ) {
            info = -12;
            LAPACKE_xerbla( "LAPACKE_dgees_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dgees( &jobvs, &sort, select, &n, a, &lda_t, sdim, wr, wi,
                          vs, &ldvs_t, work, &lwork, bwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            vs_t = (double*)
                LAPACKE_malloc( sizeof(double) * ldvs_t * MAX(1,n) );
            if( vs_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        
        LAPACKE_dge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_dgees( &jobvs, &sort, select, &n, a_t, &lda_t, sdim, wr, wi,
                      vs_t, &ldvs_t, work, &lwork, bwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, vs_t, ldvs_t, vs, ldvs );
        }
        
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            LAPACKE_free( vs_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgees_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgees_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_D_SELECT2 select, lapack_int n, double* a,
                          lapack_int lda, lapack_int* sdim, double* wr,
                          double* wi, double* vs, lapack_int ldvs )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_logical* bwork = NULL;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgees", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -6;
        }
    }
#endif
    
    if( LAPACKE_lsame( sort, 's' ) ) {
        bwork = (lapack_logical*)
            LAPACKE_malloc( sizeof(lapack_logical) * MAX(1,n) );
        if( bwork == NULL ) {
            info = LAPACK_WORK_MEMORY_ERROR;
            goto exit_level_0;
        }
    }
    
    info = LAPACKE_dgees_work( matrix_layout, jobvs, sort, select, n, a, lda,
                               sdim, wr, wi, vs, ldvs, &work_query, lwork,
                               bwork );
    if( info != 0 ) {
        goto exit_level_1;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_dgees_work( matrix_layout, jobvs, sort, select, n, a, lda,
                               sdim, wr, wi, vs, ldvs, work, lwork, bwork );
    
    LAPACKE_free( work );
exit_level_1:
    if( LAPACKE_lsame( sort, 's' ) ) {
        LAPACKE_free( bwork );
    }
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgees", info );
    }
    return info;
}

lapack_int LAPACKE_cgees_work( int matrix_layout, char jobvs, char sort,
                               LAPACK_C_SELECT1 select, lapack_int n,
                               lapack_complex_float* a, lapack_int lda,
                               lapack_int* sdim, lapack_complex_float* w,
                               lapack_complex_float* vs, lapack_int ldvs,
                               lapack_complex_float* work, lapack_int lwork,
                               float* rwork, lapack_logical* bwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgees( &jobvs, &sort, select, &n, a, &lda, sdim, w, vs, &ldvs,
                      work, &lwork, rwork, bwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldvs_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        lapack_complex_float* vs_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_cgees_work", info );
            return info;
        }
        if( ldvs < n ) {
            info = -11;
            LAPACKE_xerbla( "LAPACKE_cgees_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cgees( &jobvs, &sort, select, &n, a, &lda_t, sdim, w, vs,
                          &ldvs_t, work, &lwork, rwork, bwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            vs_t = (lapack_complex_float*)
                LAPACKE_malloc( sizeof(lapack_complex_float) *
                                ldvs_t * MAX(1,n) );
            if( vs_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        
        LAPACKE_cge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_cgees( &jobvs, &sort, select, &n, a_t, &lda_t, sdim, w, vs_t,
                      &ldvs_t, work, &lwork, rwork, bwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, vs_t, ldvs_t, vs, ldvs );
        }
        
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            LAPACKE_free( vs_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgees_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgees_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_C_SELECT1 select, lapack_int n,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_int* sdim, lapack_complex_float* w,
                          lapack_complex_float* vs, lapack_int ldvs )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_logical* bwork = NULL;
    float* rwork = NULL;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgees", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -6;
        }
    }
#endif
    
    if( LAPACKE_lsame( sort, 's' ) ) {
        bwork = (lapack_logical*)
            LAPACKE_malloc( sizeof(lapack_logical) * MAX(1,n) );
        if( bwork == NULL ) {
            info = LAPACK_WORK_MEMORY_ERROR;
            goto exit_level_0;
        }
    }
    rwork = (float*)LAPACKE_malloc( sizeof(float) * MAX(1,n) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_cgees_work( matrix_layout, jobvs, sort, select, n, a, lda,
                               sdim, w, vs, ldvs, &work_query, lwork, rwork,
                               bwork );
    if( info != 0 ) {
        goto exit_level_2;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_2;
    }
    
    info = LAPACKE_cgees_work( matrix_layout, jobvs, sort, select, n, a, lda,
                               sdim, w, vs, ldvs, work, lwork, rwork, bwork );
    
    LAPACKE_free( work );
exit_level_2:
    LAPACKE_free( rwork );
exit_level_1:
    if( LAPACKE_lsame( sort, 's' ) ) {
        LAPACKE_free( bwork );
    }
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgees", info );
    }
    return info;
}

lapack_int LAPACKE_zgees_work( int matrix_layout, char jobvs, char sort,
                               LAPACK_Z_SELECT1 select, lapack_int n,
                               lapack_complex_double* a, lapack_int lda,
                               lapack_int* sdim, lapack_complex_double* w,
                               lapack_complex_double* vs, lapack_int ldvs,
                               lapack_complex_double* work, lapack_int lwork,
                               double* rwork, lapack_logical* bwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgees( &jobvs, &sort, select, &n, a, &lda, sdim, w, vs, &ldvs,
                      work, &lwork, rwork, bwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_int ldvs_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        lapack_complex_double* vs_t = NULL;
        
        if( lda < n ) {
            info = -7;
            LAPACKE_xerbla( "LAPACKE_zgees_work", info );
            return info;
        }
        if( ldvs < n ) {
            info = -11;
            LAPACKE_xerbla( "LAPACKE_zgees_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zgees( &jobvs, &sort, select, &n, a, &lda_t, sdim, w, vs,
                          &ldvs_t, work, &lwork, rwork, bwork, &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            vs_t = (lapack_complex_double*)
                LAPACKE_malloc( sizeof(lapack_complex_double) *
                                ldvs_t * MAX(1,n) );
            if( vs_t == NULL ) {
                info = LAPACK_TRANSPOSE_MEMORY_ERROR;
                goto exit_level_1;
            }
        }
        
        LAPACKE_zge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_zgees( &jobvs, &sort, select, &n, a_t, &lda_t, sdim, w, vs_t,
                      &ldvs_t, work, &lwork, rwork, bwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, vs_t, ldvs_t, vs, ldvs );
        }
        
        if( LAPACKE_lsame( jobvs, 'v' ) ) {
            LAPACKE_free( vs_t );
        }
exit_level_1:
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgees_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgees_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_Z_SELECT1 select, lapack_int n,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_int* sdim, lapack_complex_double* w,
                          lapack_complex_double* vs, lapack_int ldvs )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_logical* bwork = NULL;
    double* rwork = NULL;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgees", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -6;
        }
    }
#endif
    
    if( LAPACKE_lsame( sort, 's' ) ) {
        bwork = (lapack_logical*)
            LAPACKE_malloc( sizeof(lapack_logical) * MAX(1,n) );
        if( bwork == NULL ) {
            info = LAPACK_WORK_MEMORY_ERROR;
            goto exit_level_0;
        }
    }
    rwork = (double*)LAPACKE_malloc( sizeof(double) * MAX(1,n) );
    if( rwork == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_1;
    }
    
    info = LAPACKE_zgees_work( matrix_layout, jobvs, sort, select, n, a, lda,
                               sdim, w, vs, ldvs, &work_query, lwork, rwork,
                               bwork );
    if( info != 0 ) {
        goto exit_level_2;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_2;
    }
    
    info = LAPACKE_zgees_work( matrix_layout, jobvs, sort, select, n, a, lda,
                               sdim, w, vs, ldvs, work, lwork, rwork, bwork );
    
    LAPACKE_free( work );
exit_level_2:
    LAPACKE_free( rwork );
exit_level_1:
    if( LAPACKE_lsame( sort, 's' ) ) {
        LAPACKE_free( bwork );
    }
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgees", info );
    }
    return info;
}

lapack_int LAPACKE_sgehrd_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, float* a, lapack_int lda,
                                float* tau, float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sgehrd( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_sgehrd_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sgehrd( &n, &ilo, &ihi, a, &lda_t, tau, work, &lwork,
                           &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_sge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_sgehrd( &n, &ilo, &ihi, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sgehrd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sgehrd_work", info );
    }
    return info;
}

lapack_int LAPACKE_sgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, float* a, lapack_int lda,
                           float* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sgehrd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_sgehrd_work( matrix_layout, n, ilo, ihi, a, lda, tau,
                                &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sgehrd_work( matrix_layout, n, ilo, ihi, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sgehrd", info );
    }
    return info;
}

lapack_int LAPACKE_dgehrd_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, double* a, lapack_int lda,
                                double* tau, double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dgehrd( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_dgehrd_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dgehrd( &n, &ilo, &ihi, a, &lda_t, tau, work, &lwork,
                           &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_dgehrd( &n, &ilo, &ihi, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dgehrd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dgehrd_work", info );
    }
    return info;
}

lapack_int LAPACKE_dgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, double* a, lapack_int lda,
                           double* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgehrd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_dgehrd_work( matrix_layout, n, ilo, ihi, a, lda, tau,
                                &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dgehrd_work( matrix_layout, n, ilo, ihi, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgehrd", info );
    }
    return info;
}

lapack_int LAPACKE_cgehrd_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, lapack_complex_float* a,
                                lapack_int lda, lapack_complex_float* tau,
                                lapack_complex_float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cgehrd( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_cgehrd_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cgehrd( &n, &ilo, &ihi, a, &lda_t, tau, work, &lwork,
                           &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_cge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_cgehrd( &n, &ilo, &ihi, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cgehrd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cgehrd_work", info );
    }
    return info;
}

lapack_int LAPACKE_cgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cgehrd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_cgehrd_work( matrix_layout, n, ilo, ihi, a, lda, tau,
                                &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cgehrd_work( matrix_layout, n, ilo, ihi, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cgehrd", info );
    }
    return info;
}

lapack_int LAPACKE_zgehrd_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, lapack_complex_double* a,
                                lapack_int lda, lapack_complex_double* tau,
                                lapack_complex_double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zgehrd( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_zgehrd_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zgehrd( &n, &ilo, &ihi, a, &lda_t, tau, work, &lwork,
                           &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_zgehrd( &n, &ilo, &ihi, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zgehrd_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zgehrd_work", info );
    }
    return info;
}

lapack_int LAPACKE_zgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zgehrd", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
    }
#endif
    
    info = LAPACKE_zgehrd_work( matrix_layout, n, ilo, ihi, a, lda, tau,
                                &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zgehrd_work( matrix_layout, n, ilo, ihi, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zgehrd", info );
    }
    return info;
}

lapack_int LAPACKE_sorghr_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, float* a, lapack_int lda,
                                const float* tau, float* work,
                                lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_sorghr( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_sorghr_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_sorghr( &n, &ilo, &ihi, a, &lda_t, tau, work, &lwork,
                           &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (float*)LAPACKE_malloc( sizeof(float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_sge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_sorghr( &n, &ilo, &ihi, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_sge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_sorghr_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_sorghr_work", info );
    }
    return info;
}

lapack_int LAPACKE_sorghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, float* a, lapack_int lda,
                           const float* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float* work = NULL;
    float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_sorghr", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_sge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_s_nancheck( n-1, tau, 1 ) ) {
            return -7;
        }
    }
#endif
    
    info = LAPACKE_sorghr_work( matrix_layout, n, ilo, ihi, a, lda, tau,
                                &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (float*)LAPACKE_malloc( sizeof(float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_sorghr_work( matrix_layout, n, ilo, ihi, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_sorghr", info );
    }
    return info;
}

lapack_int LAPACKE_dorghr_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, double* a, lapack_int lda,
                                const double* tau, double* work,
                                lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_dorghr( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_dorghr_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_dorghr( &n, &ilo, &ihi, a, &lda_t, tau, work, &lwork,
                           &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (double*)LAPACKE_malloc( sizeof(double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_dge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_dorghr( &n, &ilo, &ihi, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_dge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_dorghr_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_dorghr_work", info );
    }
    return info;
}

lapack_int LAPACKE_dorghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, double* a, lapack_int lda,
                           const double* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double* work = NULL;
    double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dorghr", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_dge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_d_nancheck( n-1, tau, 1 ) ) {
            return -7;
        }
    }
#endif
    
    info = LAPACKE_dorghr_work( matrix_layout, n, ilo, ihi, a, lda, tau,
                                &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    
    work = (double*)LAPACKE_malloc( sizeof(double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_dorghr_work( matrix_layout, n, ilo, ihi, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dorghr", info );
    }
    return info;
}

lapack_int LAPACKE_cunghr_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, lapack_complex_float* a,
                                lapack_int lda, const lapack_complex_float* tau,
                                lapack_complex_float* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_cunghr( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_float* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_cunghr_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_cunghr( &n, &ilo, &ihi, a, &lda_t, tau, work, &lwork,
                           &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_float*)
            LAPACKE_malloc( sizeof(lapack_complex_float) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_cge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_cunghr( &n, &ilo, &ihi, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_cge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_cunghr_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_cunghr_work", info );
    }
    return info;
}

lapack_int LAPACKE_cunghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_float* work = NULL;
    lapack_complex_float work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_cunghr", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_cge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_c_nancheck( n-1, tau, 1 ) ) {
            return -7;
        }
    }
#endif
    
    info = LAPACKE_cunghr_work( matrix_layout, n, ilo, ihi, a, lda, tau,
                                &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_C2INT( work_query );
    
    work = (lapack_complex_float*)
        LAPACKE_malloc( sizeof(lapack_complex_float) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_cunghr_work( matrix_layout, n, ilo, ihi, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_cunghr", info );
    }
    return info;
}

lapack_int LAPACKE_zunghr_work( int matrix_layout, lapack_int n, lapack_int ilo,
                                lapack_int ihi, lapack_complex_double* a,
                                lapack_int lda,
                                const lapack_complex_double* tau,
                                lapack_complex_double* work, lapack_int lwork )
{
    lapack_int info = 0;
    if( matrix_layout == LAPACK_COL_MAJOR ) {
        
        LAPACK_zunghr( &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
    } else if( matrix_layout == LAPACK_ROW_MAJOR ) {
        lapack_int lda_t = MAX(1,n);
        lapack_complex_double* a_t = NULL;
        
        if( lda < n ) {
            info = -6;
            LAPACKE_xerbla( "LAPACKE_zunghr_work", info );
            return info;
        }
        
        if( lwork == -1 ) {
            LAPACK_zunghr( &n, &ilo, &ihi, a, &lda_t, tau, work, &lwork,
                           &info );
            return (info < 0) ? (info - 1) : info;
        }
        
        a_t = (lapack_complex_double*)
            LAPACKE_malloc( sizeof(lapack_complex_double) * lda_t * MAX(1,n) );
        if( a_t == NULL ) {
            info = LAPACK_TRANSPOSE_MEMORY_ERROR;
            goto exit_level_0;
        }
        
        LAPACKE_zge_trans( matrix_layout, n, n, a, lda, a_t, lda_t );
        
        LAPACK_zunghr( &n, &ilo, &ihi, a_t, &lda_t, tau, work, &lwork, &info );
        if( info < 0 ) {
            info = info - 1;
        }
        
        LAPACKE_zge_trans( LAPACK_COL_MAJOR, n, n, a_t, lda_t, a, lda );
        
        LAPACKE_free( a_t );
exit_level_0:
        if( info == LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            LAPACKE_xerbla( "LAPACKE_zunghr_work", info );
        }
    } else {
        info = -1;
        LAPACKE_xerbla( "LAPACKE_zunghr_work", info );
    }
    return info;
}

lapack_int LAPACKE_zunghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau )
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_double* work = NULL;
    lapack_complex_double work_query;
    if( matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_zunghr", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if( LAPACKE_get_nancheck() ) {
        
        if( LAPACKE_zge_nancheck( matrix_layout, n, n, a, lda ) ) {
            return -5;
        }
        if( LAPACKE_z_nancheck( n-1, tau, 1 ) ) {
            return -7;
        }
    }
#endif
    
    info = LAPACKE_zunghr_work( matrix_layout, n, ilo, ihi, a, lda, tau,
                                &work_query, lwork );
    if( info != 0 ) {
        goto exit_level_0;
    }
    lwork = LAPACK_Z2INT( work_query );
    
    work = (lapack_complex_double*)
        LAPACKE_malloc( sizeof(lapack_complex_double) * lwork );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    
    info = LAPACKE_zunghr_work( matrix_layout, n, ilo, ihi, a, lda, tau, work,
                                lwork );
    
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_zunghr", info );
    }
    return info;
}


void* get_dgetrf_ptr(void) {
    return (void*)&LAPACKE_dgetrf;
}

void* get_sgetrf_ptr(void) {
    return (void*)&LAPACKE_sgetrf;
}

void* get_zgetrf_ptr(void) {
    return (void*)&LAPACKE_zgetrf;
}

void* get_cgetrf_ptr(void) {
    return (void*)&LAPACKE_cgetrf;
}
#else
#define HAVE_LAPACK_CONFIG_H
#define HAVE_STDINT_H
#include <stdint.h>
#define int32_t int32_t
#include <lapacke.h>

void* get_dgetrf_ptr(void) {
    return (void*)&LAPACKE_dgetrf;
}

void* get_sgetrf_ptr(void) {
    return (void*)&LAPACKE_sgetrf;
}

void* get_zgetrf_ptr(void) {
    return (void*)&LAPACKE_zgetrf;
}

void* get_cgetrf_ptr(void) {
    return (void*)&LAPACKE_cgetrf;
}
#endif
