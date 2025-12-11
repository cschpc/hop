#if defined(_HOP_EXTENSIONS_COMPLEX)
/*
 * Operator overloading for complex data types
 *   operator: +
 *   data types: cuDoubleComplex, cuFloatComplex, <T> for scalars
 */
__host__ __device__ __inline__
cuDoubleComplex operator+(cuDoubleComplex const &a, cuDoubleComplex const &b)
{
    return cuCadd(a, b);
}
__host__ __device__ __inline__
cuFloatComplex operator+(cuFloatComplex const &a, cuFloatComplex const &b)
{
    return cuCaddf(a, b);
}
__host__ __device__ __inline__
cuDoubleComplex operator+(cuFloatComplex const &a, cuDoubleComplex const &b)
{
    return cuCadd(cuComplexFloatToDouble(a), b);
}
__host__ __device__ __inline__
cuDoubleComplex operator+(cuDoubleComplex const &a, cuFloatComplex const &b)
{
    return cuCadd(a, cuComplexFloatToDouble(b));
}

template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator+(cuDoubleComplex const &a, T const &b)
{
    return cuCadd(a, make_cuDoubleComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator+(T const &a, cuDoubleComplex const &b)
{
    return cuCadd(make_cuDoubleComplex(a, 0), b);
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator+(cuFloatComplex const &a, T const &b)
{
    return cuCaddf(a, make_cuFloatComplex(b));
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator+(T const &a, cuFloatComplex const &b)
{
    return cuCaddf(make_cuFloatComplex(a), b);
}


/*
 * Operator overloading for complex data types
 *   operator: -
 *   data types: cuDoubleComplex, cuFloatComplex, <T> for scalars
 */
__host__ __device__ __inline__
cuDoubleComplex operator-(cuDoubleComplex const &a, cuDoubleComplex const &b)
{
    return cuCsub(a, b);
}
__host__ __device__ __inline__
cuFloatComplex operator-(cuFloatComplex const &a, cuFloatComplex const &b)
{
    return cuCsubf(a, b);
}
__host__ __device__ __inline__
cuDoubleComplex operator-(cuFloatComplex const &a, cuDoubleComplex const &b)
{
    return cuCsub(cuComplexFloatToDouble(a), b);
}
__host__ __device__ __inline__
cuDoubleComplex operator-(cuDoubleComplex const &a, cuFloatComplex const &b)
{
    return cuCsub(a, cuComplexFloatToDouble(b));
}

template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator-(cuDoubleComplex const &a, T const &b)
{
    return cuCsub(a, make_cuDoubleComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator-(T const &a, cuDoubleComplex const &b)
{
    return cuCsub(make_cuDoubleComplex(a, 0), b);
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator-(cuFloatComplex const &a, T const &b)
{
    return cuCsubf(a, make_cuFloatComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator-(T const &a, cuFloatComplex const &b)
{
    return cuCsubf(make_cuFloatComplex(a, 0), b);
}


/*
 * Operator overloading for complex data types
 *   operator: *
 *   data types: cuDoubleComplex, cuFloatComplex, <T> for scalars
 */
__host__ __device__ __inline__
cuDoubleComplex operator*(cuDoubleComplex const &a, cuDoubleComplex const &b)
{
    return cuCmul(a, b);
}
__host__ __device__ __inline__
cuFloatComplex operator*(cuFloatComplex const &a, cuFloatComplex const &b)
{
    return cuCmulf(a, b);
}
__host__ __device__ __inline__
cuDoubleComplex operator*(cuFloatComplex const &a, cuDoubleComplex const &b)
{
    return cuCmul(cuComplexFloatToDouble(a), b);
}
__host__ __device__ __inline__
cuDoubleComplex operator*(cuDoubleComplex const &a, cuFloatComplex const &b)
{
    return cuCmul(a, cuComplexFloatToDouble(b));
}

template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator*(cuDoubleComplex const &a, T const &b)
{
    return cuCmul(a, make_cuDoubleComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator*(T const &a, cuDoubleComplex const &b)
{
    return cuCmul(make_cuDoubleComplex(a, 0), b);
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator*(cuFloatComplex const &a, T const &b)
{
    return cuCmul(a, make_cuFloatComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator*(T const &a, cuFloatComplex const &b)
{
    return cuCmul(make_cuFloatComplex(a, 0), b);
}


/*
 * Operator overloading for complex data types
 *   operator: +=
 *   data types: cuDoubleComplex, cuFloatComplex, <T> for scalars
 */
__host__ __device__ __inline__
void operator+=(cuDoubleComplex &a, cuDoubleComplex const &b)
{
    a = cuCadd(a, b);
}
__host__ __device__ __inline__
void operator+=(cuFloatComplex &a, cuFloatComplex const &b)
{
    a = cuCaddf(a, b);
}
__host__ __device__ __inline__
void operator+=(cuFloatComplex &a, cuDoubleComplex const &b)
{
    a = cuCaddf(a, cuComplexDoubleToFloat(b));
}
__host__ __device__ __inline__
void operator+=(cuDoubleComplex &a, cuFloatComplex const &b)
{
    a = cuCadd(a, cuComplexFloatToDouble(b));
}

template <typename T>
__host__ __device__ __inline__
void operator+=(cuDoubleComplex &a, T const &b)
{
    a = cuCadd(a, make_cuDoubleComplex(b));
}
template <typename T>
__host__ __device__ __inline__
void operator+=(cuFloatComplex &a, T const &b)
{
    a = cuCadd(a, make_cuFloatComplex(b));
}
#endif
